"""
Turns a MediaPipe hand_landmarks object into a fixed-size feature vector
that's independent of where the hand is on screen or how close it is to
the camera — so the same gesture produces (roughly) the same vector no
matter where or how it's performed.
"""

import numpy as np

WRIST = 0
MIDDLE_FINGER_MCP = 9  # knuckle at the base of the middle finger


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Wrist-origin + middle-MCP scale normalization for a (21, 3) array."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3).copy()
    pts -= pts[WRIST]
    scale = float(np.linalg.norm(pts[MIDDLE_FINGER_MCP]))
    if scale > 1e-6:
        pts /= scale
    return pts


def landmarks_to_vector(hand_landmarks) -> list:
    """
    Convert 21 hand landmarks (each with x, y, z) into a flat list of
    63 floats, normalized for position and scale.
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    return normalize_points(points).flatten().tolist()
