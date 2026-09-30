"""
Text display for the deaf path (e.g. "Ponovo" on a robot face screen).

The game never blocks on it: every call runs on a background thread and
failures only get logged. Plug a real display in with set_display(show, warm),
where show(text, duration_s) renders text and warm() pre-opens the link.
Without one, text is only logged.

IGRA_HEAD_SCREEN=0 turns it off.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional

logger = logging.getLogger("igra.head_screen")

ENABLED = os.getenv("IGRA_HEAD_SCREEN", "1").strip() not in {"0", "false", "False", "no"}

_show_fn: Optional[Callable[[str, float], None]] = None
_warm_fn: Optional[Callable[[], None]] = None


def set_display(
    show: Optional[Callable[[str, float], None]],
    warm: Optional[Callable[[], None]] = None,
) -> None:
    global _show_fn, _warm_fn
    _show_fn = show
    _warm_fn = warm


def _run_in_background(target, *args) -> None:
    if not ENABLED:
        return

    def _safe() -> None:
        try:
            target(*args)
        except Exception as exc:
            logger.warning("Head screen failed: %s", exc)

    threading.Thread(target=_safe, name="igra-head-screen", daemon=True).start()


def _show(text: str, duration_s: float) -> None:
    if _show_fn is None:
        logger.info("Head screen (no display): %r", text)
        return
    _show_fn(text, duration_s)


def flash(text: str, duration_s: float = 2.0) -> None:
    """Show text on the display, then restore its default content."""
    _run_in_background(_show, text, duration_s)


def warm_up() -> None:
    """Pre-open the display link so the first flash is fast."""
    if _warm_fn is not None:
        _run_in_background(_warm_fn)
