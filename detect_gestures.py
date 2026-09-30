"""
Recognize hand gestures live against gestures.json and show the label.

Uses the shared HandGestureDetector + GestureDatabase (augmented k-NN).

Run:
    python detect_gestures.py

Press 'q' to quit.
"""

from __future__ import annotations

import os

import cv2

from gesture_database import GestureDatabase
from gesture_utils import landmarks_to_vector
from hand_gesture_detector import HandGestureDetector

DB_PATH = os.getenv("IGRA_GESTURE_DB", "gestures.json")
STABLE_FRAMES = int(os.getenv("IGRA_STABLE_FRAMES", "3"))
_MAX = os.getenv("IGRA_MAX_DISTANCE")
MAX_DISTANCE = float(_MAX) if _MAX else None


def main():
    db = GestureDatabase().load(DB_PATH)
    if not db.real_samples():
        print(f"No samples in {DB_PATH}. Run collect_gestures.py first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not access the camera.")
        return

    detector = HandGestureDetector(max_hands=1)

    counts = db.counts()
    print(
        f"Real {len(db.real_samples())} → match corpus {len(db.match_samples)} "
        f"(aug x{db.augment_per_real}/sample, k={db.match_k}, "
        f"cap={db.default_max_distance}, pruned={db.pruned_count})"
    )
    print(f"Labels: {', '.join(f'{k}={v}' for k, v in sorted(counts.items()))}")
    print("HUD: green=accepted, yellow=~nearest guess, red=no hand")
    print("Press 'q' to quit.")

    pending_label = None
    pending_count = 0
    stable_label = None
    stable_distance = None

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        results = detector.find_hands(frame)
        detector.draw_landmarks(frame, results)

        display_label = "no hand"
        display_dist = ""
        color = (0, 0, 255)
        sub = ""

        if results.multi_hand_landmarks:
            vector = landmarks_to_vector(results.multi_hand_landmarks[0])
            nearest = db.nearest(vector, top_n=3)
            label, distance = db.match(vector, max_distance=MAX_DISTANCE)

            if nearest:
                sub = "  ".join(f"{lab}:{dist:.2f}" for lab, dist in nearest)

            if label is None:
                pending_label = None
                pending_count = 0
                stable_label = None
                stable_distance = distance
                if nearest:
                    display_label = f"~{nearest[0][0]}"
                    display_dist = f"  d={nearest[0][1]:.2f}"
                else:
                    display_label = "?"
                    if distance is not None:
                        display_dist = f"  d={distance:.2f}"
                color = (0, 165, 255)
            else:
                if label == pending_label:
                    pending_count += 1
                else:
                    pending_label = label
                    pending_count = 1

                if pending_count >= STABLE_FRAMES:
                    stable_label = label
                    stable_distance = distance

                if stable_label is not None:
                    display_label = stable_label
                    display_dist = f"  d={stable_distance:.2f}"
                    color = (0, 255, 0)
                else:
                    display_label = f"...{label}"
                    display_dist = f"  d={distance:.2f}"
                    color = (0, 255, 255)
        else:
            pending_label = None
            pending_count = 0
            stable_label = None
            stable_distance = None

        cv2.putText(
            frame,
            f"{display_label}{display_dist}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            color,
            3,
            cv2.LINE_AA,
        )
        if sub:
            cv2.putText(
                frame,
                sub,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"db: {', '.join(db.labels())}   q: quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Detect Gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
