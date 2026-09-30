"""
JSON-backed database of hand gesture samples.

gestures.json stores *real* captures only:
  {"label": "deaf", "vector": [63 floats], "source": "real"}

At load time we optionally expand each real sample into many mild synthetic
neighbors (rotation / noise / scale) so matching is robust without 1000 manual
shots. Matching uses k-NN majority vote over the expanded corpus.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from gesture_augment import expand_samples


DEFAULT_AUGMENT_PER_REAL = int(os.getenv("IGRA_AUGMENT_PER_REAL", "40"))
DEFAULT_MATCH_K = int(os.getenv("IGRA_MATCH_K", "7"))
# Absolute cap — rock/paper captures are often 1.5–3.5 apart live.
DEFAULT_MAX_DISTANCE = float(os.getenv("IGRA_MAX_DISTANCE", "3.2"))
# Winner must beat the next-best *other* label by this ratio (or raw gap).
DEFAULT_MARGIN_RATIO = float(os.getenv("IGRA_MARGIN_RATIO", "1.12"))
DEFAULT_OUTLIER_FACTOR = float(os.getenv("IGRA_OUTLIER_FACTOR", "1.8"))


class GestureDatabase:
    def __init__(self):
        # Real samples persisted to disk.
        self.samples: list[dict] = []
        # In-memory corpus used for matching (real + augmented).
        self.match_samples: list[dict] = []
        self.augment_per_real = DEFAULT_AUGMENT_PER_REAL
        self.match_k = DEFAULT_MATCH_K
        self.default_max_distance = DEFAULT_MAX_DISTANCE
        self.margin_ratio = DEFAULT_MARGIN_RATIO
        self.outlier_factor = DEFAULT_OUTLIER_FACTOR
        self.pruned_count = 0

    def add(self, label: str, vector: list, *, source: str = "real"):
        clean = (label or "").strip()
        if not clean:
            raise ValueError("label must be non-empty")
        self.samples.append(
            {
                "label": clean,
                "vector": list(vector),
                "source": source,
                "created_at": time.time(),
            }
        )
        self._rebuild_match_corpus()

    def real_samples(self) -> list[dict]:
        return [s for s in self.samples if s.get("source", "real") != "aug"]

    def _prune_outliers(self, real: list[dict]) -> list[dict]:
        """Drop real samples far from their per-label median (bad captures)."""
        by_label: dict[str, list[dict]] = {}
        for sample in real:
            by_label.setdefault(sample["label"], []).append(sample)

        kept: list[dict] = []
        pruned = 0
        for label, group in by_label.items():
            if len(group) <= 2:
                kept.extend(group)
                continue
            vectors = [s["vector"] for s in group]
            # Component-wise median is robust enough for landmark vectors.
            dim = len(vectors[0])
            median = []
            for i in range(dim):
                col = sorted(v[i] for v in vectors)
                mid = len(col) // 2
                if len(col) % 2:
                    median.append(col[mid])
                else:
                    median.append(0.5 * (col[mid - 1] + col[mid]))
            distances = [self._euclidean(s["vector"], median) for s in group]
            order = sorted(distances)
            # Robust scale: median distance to median pose.
            med_d = order[len(order) // 2] or 0.35
            limit = max(1.25, med_d * self.outlier_factor)
            for sample, dist in zip(group, distances):
                if dist <= limit:
                    kept.append(sample)
                else:
                    pruned += 1
        self.pruned_count = pruned
        return kept if kept else list(real)

    def _rebuild_match_corpus(self) -> None:
        real = self._prune_outliers(self.real_samples())
        if self.augment_per_real <= 1:
            self.match_samples = [
                {"label": s["label"], "vector": s["vector"], "source": s.get("source", "real")}
                for s in real
            ]
            return
        self.match_samples = expand_samples(real, per_real=self.augment_per_real)

    def nearest(self, vector: list, top_n: int = 5) -> list[tuple[str, float]]:
        """Return top-N (label, distance) hits for HUD/debug."""
        corpus = self.match_samples or self.samples
        if not corpus:
            return []
        scored = [
            (sample["label"], self._euclidean(vector, sample["vector"]))
            for sample in corpus
        ]
        scored.sort(key=lambda item: item[1])
        # unique labels preserving order
        out: list[tuple[str, float]] = []
        seen = set()
        for label, dist in scored:
            if label in seen:
                continue
            seen.add(label)
            out.append((label, dist))
            if len(out) >= top_n:
                break
        return out

    def match(self, vector: list, max_distance: float | None = None):
        """
        k-NN majority vote with margin vs the next-best other label.

        Absolute threshold alone fails for rock/paper when captures vary a lot;
        requiring the winner to beat the runner-up keeps o/y/u separated while
        still accepting looser same-label poses.

        Returns (label, score_distance) or (None, best_distance).
        """
        corpus = self.match_samples or self.samples
        if not corpus:
            return None, None

        cap = self.default_max_distance if max_distance is None else float(max_distance)

        scored: list[tuple[float, str]] = []
        for sample in corpus:
            distance = self._euclidean(vector, sample["vector"])
            scored.append((distance, sample["label"]))
        scored.sort(key=lambda item: item[0])

        best_distance = scored[0][0]
        best_label = scored[0][1]

        # Second-best *different* label distance (for margin test).
        second_distance = None
        for dist, label in scored[1:]:
            if label != best_label:
                second_distance = dist
                break

        k = max(1, min(self.match_k, len(scored)))
        # Neighbors within a band of the nearest get a vote.
        band = max(cap * 0.35, best_distance * 1.35, 0.75)
        voters = [item for item in scored[: max(k, 15)] if item[0] <= band][:k]
        if not voters:
            voters = scored[:1]

        votes = Counter(label for _, label in voters)
        winner, winner_votes = votes.most_common(1)[0]
        top = votes.most_common()
        if len(top) >= 2 and top[0][1] == top[1][1]:
            winner = best_label
            winner_votes = 1

        winner_distances = [d for d, label in voters if label == winner]
        mean_d = sum(winner_distances) / max(1, len(winner_distances))

        # Hard reject if even the nearest is absurdly far.
        if best_distance > cap:
            return None, best_distance

        # Margin: winner nearest must be meaningfully closer than next label.
        if second_distance is not None:
            margin_ok = second_distance >= max(
                best_distance * self.margin_ratio,
                best_distance + 0.15,
            )
        else:
            margin_ok = True

        # Strong absolute hit always wins.
        strong_hit = best_distance <= min(1.0, cap * 0.45)
        vote_ok = winner_votes >= max(2, (k + 1) // 2) or winner == best_label

        if strong_hit or (margin_ok and vote_ok):
            return winner, mean_d if winner == best_label else best_distance

        # Soft accept: nearest label with ok margin even if votes split.
        if margin_ok and best_distance <= cap * 0.85:
            return best_label, best_distance

        return None, best_distance

    @staticmethod
    def _euclidean(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def labels(self):
        return sorted({s["label"] for s in self.real_samples()})

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for s in self.real_samples():
            c[s["label"]] = c.get(s["label"], 0) + 1
        return c

    def save(self, path: str):
        # Persist real captures only — augmented vectors are rebuilt on load.
        payload = []
        for s in self.real_samples():
            payload.append(
                {
                    "label": s["label"],
                    "vector": s["vector"],
                    "source": "real",
                    **({"created_at": s["created_at"]} if "created_at" in s else {}),
                }
            )
        Path(path).write_text(json.dumps(payload, indent=2))

    def load(self, path: str, *, augment_per_real: int | None = None):
        file = Path(path)
        if file.exists():
            raw = json.loads(file.read_text())
            self.samples = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).strip()
                vector = item.get("vector")
                if not label or not isinstance(vector, list):
                    continue
                source = str(item.get("source", "real"))
                # Ignore any previously baked aug rows if someone saved them.
                if source == "aug":
                    continue
                row = {"label": label, "vector": vector, "source": "real"}
                if "created_at" in item:
                    row["created_at"] = item["created_at"]
                self.samples.append(row)
        if augment_per_real is not None:
            self.augment_per_real = int(augment_per_real)
        self._rebuild_match_corpus()
        return self
