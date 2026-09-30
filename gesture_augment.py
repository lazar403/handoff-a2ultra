"""
Synthesize nearby hand-pose vectors from a few real captures.

Real MediaPipe samples stay in gestures.json. At load/match time we expand
each real sample into many mild variants so you do not need 1000 manual shots.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np

from gesture_utils import MIDDLE_FINGER_MCP, WRIST, normalize_points


def _seed_for(label: str, vector: Sequence[float], index: int) -> int:
    raw = f"{label}|{index}|{len(vector)}|{round(float(vector[3]), 5)}".encode()
    return int(hashlib.md5(raw).hexdigest()[:8], 16)


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def augment_vector(
    vector: Sequence[float],
    *,
    count: int = 40,
    label: str = "",
    include_original: bool = True,
    noise_std: float = 0.025,
    max_rot_deg: float = 18.0,
    scale_jitter: float = 0.08,
    joint_jitter: float = 0.02,
) -> list[list[float]]:
    """
    Build `count` approximate neighbors around one real landmark vector.

    Transforms stay small on purpose: large random poses would invent fake
    gestures and pollute matching.
    """
    base = np.asarray(vector, dtype=np.float32).reshape(-1, 3)
    if base.shape != (21, 3):
        raise ValueError(f"expected 21x3 landmarks, got {base.shape}")

    out: list[list[float]] = []
    if include_original:
        out.append(base.flatten().tolist())

    target = max(0, int(count) - (1 if include_original else 0))
    max_rot = np.deg2rad(max_rot_deg)

    for i in range(target):
        rng = np.random.default_rng(_seed_for(label, vector, i))
        rx, ry, rz = rng.uniform(-max_rot, max_rot, size=3)
        rot = _rotation_matrix(float(rx), float(ry), float(rz))
        points = base @ rot.T

        # Mild non-uniform scale (camera angle / hand size drift).
        scales = rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter, size=3).astype(np.float32)
        points *= scales

        # Per-joint wobble + global noise.
        points += rng.normal(0.0, joint_jitter, size=points.shape).astype(np.float32)
        points += rng.normal(0.0, noise_std, size=points.shape).astype(np.float32)

        # Occasional mirror on X (left/right hand-ish variation). Keep rare.
        if rng.random() < 0.15:
            points[:, 0] *= -1.0

        points = normalize_points(points)
        out.append(points.flatten().tolist())

    return out


def expand_samples(
    samples: Iterable[dict],
    *,
    per_real: int = 40,
    real_only_key: str = "source",
) -> list[dict]:
    """
    Expand a list of real samples into real + synthetic match corpus.

    Samples already tagged source=\"aug\" are passed through unchanged.
    """
    expanded: list[dict] = []
    for sample in samples:
        label = str(sample.get("label", "")).strip()
        vector = sample.get("vector") or []
        source = str(sample.get(real_only_key, "real"))
        if source == "aug":
            expanded.append(sample)
            continue
        for idx, vec in enumerate(
            augment_vector(vector, count=per_real, label=label, include_original=True)
        ):
            expanded.append(
                {
                    "label": label,
                    "vector": vec,
                    "source": "real" if idx == 0 else "aug",
                    "parent": sample.get("id"),
                }
            )
    return expanded
