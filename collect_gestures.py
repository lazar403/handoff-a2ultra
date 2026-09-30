"""
Record hand gesture samples into gestures.json.

Controls:
  c - capture the current hand pose, then type a label in the terminal
      (e.g. rock, paper, scissors, deaf, y, o, u)
  q - quit and save

Tips:
  - 5–15 real captures per label from slightly different angles is enough.
  - Matching expands each real sample into ~40 mild variants automatically
    (see gesture_augment.py) — you do NOT need 1000 manual shots.
  - Prefer stable label names: deaf (finger on ear), rock, paper, scissors.

Run:
    python collect_gestures.py
"""

import cv2

from gesture_database import GestureDatabase
from gesture_utils import landmarks_to_vector
from hand_gesture_detector import HandGestureDetector

DB_PATH = "gestures.json"

# Common typos / short forms -> canonical labels
LABEL_ALIASES = {
    "de": "deaf",
    "deaf_mute": "deaf",
    "gluv": "deaf",
    "gluvonem": "deaf",
    "r": "rock",
    "kamen": "rock",
    "p": "paper",
    "papir": "paper",
    "s": "scissors",
    "makaze": "scissors",
}


def canonicalize_label(raw: str) -> str:
    label = (raw or "").strip().lower().replace(" ", "_")
    return LABEL_ALIASES.get(label, label)


def print_stats(db: GestureDatabase) -> None:
    counts = db.counts()
    real_n = len(db.real_samples())
    match_n = len(db.match_samples)
    print(
        f"Real samples: {real_n} | match corpus (with aug): {match_n} "
        f"| per-real aug: {db.augment_per_real}"
    )
    if counts:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"Per label: {detail}")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not access the camera.")
        return

    detector = HandGestureDetector(max_hands=1)
    db = GestureDatabase().load(DB_PATH)

    print_stats(db)
    print("Press 'c' to capture, 'q' to quit. Labels auto-normalized (de→deaf, …).")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        results = detector.find_hands(frame)
        detector.draw_landmarks(frame, results)

        counts = db.counts()
        hud = "c: capture   q: quit"
        if counts:
            hud = " | ".join(f"{k}:{v}" for k, v in sorted(counts.items())) + "   c/q"
        cv2.putText(frame, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("Collect Gestures", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if not results.multi_hand_landmarks:
                print("No hand detected, try again.")
                continue

            vector = landmarks_to_vector(results.multi_hand_landmarks[0])
            label = canonicalize_label(input("Label for this gesture: "))
            if label:
                db.add(label, vector)
                db.save(DB_PATH)
                print(
                    f"Saved real '{label}'. "
                    f"real={db.counts().get(label, 0)} "
                    f"match_corpus≈{len(db.match_samples)}"
                )

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    db.save(DB_PATH)
    print_stats(db)
    print(f"Done. Real samples saved to {DB_PATH}.")


if __name__ == "__main__":
    main()
