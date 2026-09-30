"""
Open a camera by OpenCV index or path (e.g. IGRA_CAMERA=0, /dev/video4, rtsp://...).

Robot cameras that are not plain OpenCV devices (e.g. ROS 2 image topics)
plug in with set_camera_opener(fn), where fn(device) returns a capture-like
object exposing read() / isOpened() / release().
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Optional

logger = logging.getLogger("igra.camera")

_opener: Optional[Callable[[str], Any]] = None


def set_camera_opener(fn: Optional[Callable[[str], Any]]) -> None:
    """Use fn(device) to open cameras; None restores plain OpenCV."""
    global _opener
    _opener = fn


def normalize_camera_id(value: str | int | None) -> str:
    raw = str(value if value is not None else "0").strip()
    return raw or "0"


def open_camera(device: str | int | None, *, startup_timeout: float = 10.0) -> tuple[Any, str, str]:
    """
    Open camera. Returns (capture, resolved_id, backend) where backend is
    'custom' or 'opencv'.
    """
    requested = normalize_camera_id(device)
    logger.info("Opening IGRA camera %r", requested)
    if _opener is not None:
        cap = _opener(requested)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {requested!r} did not open")
        return cap, requested, "custom"
    return _open_opencv(requested)


def _open_opencv(requested: str) -> tuple[Any, str, str]:
    import cv2

    target: int | str = int(requested) if requested.isdigit() else requested
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0
    cap = cv2.VideoCapture(target, backend)
    if not cap.isOpened() and backend:
        cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {requested!r} with OpenCV")
    logger.info("IGRA camera OpenCV ready: %s", requested)
    return cap, requested, "opencv"
