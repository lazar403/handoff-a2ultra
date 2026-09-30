"""
Hand Gesture Detector
----------------------
Opens the Mac's webcam and detects hands + hand landmarks in real time
using MediaPipe. This is the shared building block (HandGestureDetector)
used by collect_gestures.py and detect_gestures.py.

Run this file directly for a plain hand-tracking demo with no gesture
matching. For the full pipeline see:
  - collect_gestures.py  (record labeled gestures into a database)
  - detect_gestures.py   (recognize gestures live and print the word)

Requirements:
    pip install opencv-python mediapipe

Run:
    python hand_gesture_detector.py

Press 'q' to quit.

Backends:
  - legacy mp.solutions.hands (mediapipe <= 0.10.21, e.g. laptop venv)
  - Tasks HandLandmarker (mediapipe >= 0.10.30 / 1.x, e.g. A2 system python,
    where mp.solutions no longer exists). Needs models/hand_landmarker.task
    or IGRA_HAND_MODEL pointing at one.
Both return an object with .multi_hand_landmarks[i].landmark[j].x/y/z.
"""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp

DEFAULT_HAND_MODEL = os.getenv(
    "IGRA_HAND_MODEL",
    str(Path(__file__).resolve().parent / "models" / "hand_landmarker.task"),
)


class HandGestureDetector:
    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.7):
        self._legacy = hasattr(mp, "solutions")
        if self._legacy:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles
            return

        from mediapipe.tasks.python import BaseOptions, vision

        model = Path(DEFAULT_HAND_MODEL)
        if not model.is_file():
            raise FileNotFoundError(
                f"mediapipe {mp.__version__} has no mp.solutions; Tasks backend needs "
                f"a hand landmarker model at {model} (set IGRA_HAND_MODEL)"
            )
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        )
        self._connections = [
            (c.start, c.end)
            for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
        ]
        self._last_ts_ms = -1

    def find_hands(self, frame):
        """Detect hands in a BGR frame. Returns the MediaPipe results object."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._legacy:
            return self.hands.process(rgb_frame)

        # VIDEO mode requires strictly increasing timestamps.
        ts_ms = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.hands.detect_for_video(image, ts_ms)
        hands = [SimpleNamespace(landmark=lms) for lms in result.hand_landmarks]
        return SimpleNamespace(multi_hand_landmarks=hands or None)

    def draw_landmarks(self, frame, results):
        """Draw hand landmarks + connections on the frame in place."""
        if not results.multi_hand_landmarks:
            return
        if self._legacy:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
            return

        h, w = frame.shape[:2]
        for hand_landmarks in results.multi_hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            for a, b in self._connections:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, 3, (0, 0, 255), -1, cv2.LINE_AA)

    def close(self):
        self.hands.close()


def main():
    cap = cv2.VideoCapture(0)  # 0 = default camera (built-in MacBook webcam)

    if not cap.isOpened():
        print("Error: could not access the camera. Check macOS camera permissions "
              "for your terminal/IDE in System Settings > Privacy & Security > Camera.")
        return

    detector = HandGestureDetector()

    print("Camera started. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror image, feels more natural
        results = detector.find_hands(frame)
        detector.draw_landmarks(frame, results)

        cv2.imshow("Hand Gesture Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()