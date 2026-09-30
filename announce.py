"""
Speak a short phrase (default: RADI).

Default output is host TTS (macOS `say` / espeak). Any other voice (robot
speaker, cloud TTS, ...) plugs in with set_speaker(fn), where fn(text) -> bool.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
from typing import Callable, Optional

logger = logging.getLogger("igra.announce")

_last_spoken_at = 0.0
_last_text = ""
_speaker: Optional[Callable[[str], bool]] = None


def set_speaker(fn: Optional[Callable[[str], bool]]) -> None:
    """Route announcements through fn(text) -> bool; None restores local TTS."""
    global _speaker
    _speaker = fn


def _cooldown_ok(text: str, cooldown_s: float) -> bool:
    global _last_spoken_at, _last_text
    now = time.time()
    if text == _last_text and (now - _last_spoken_at) < cooldown_s:
        return False
    _last_spoken_at = now
    _last_text = text
    return True


def speak_local(text: str) -> bool:
    """Best-effort local TTS on the host."""
    phrase = (text or "").strip()
    if not phrase:
        return False

    system = platform.system().lower()
    try:
        if system == "darwin" and shutil.which("say"):
            subprocess.Popen(
                ["say", phrase],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Local say: %s", phrase)
            return True
        if shutil.which("espeak-ng"):
            subprocess.Popen(
                ["espeak-ng", phrase],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Local espeak-ng: %s", phrase)
            return True
        if shutil.which("espeak"):
            subprocess.Popen(
                ["espeak", phrase],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Local espeak: %s", phrase)
            return True
    except Exception as exc:
        logger.warning("Local TTS failed: %s", exc)
        return False

    logger.info("No local TTS binary; logged only: %s", phrase)
    return False


def announce(text: str = "RADI", *, cooldown_s: float | None = None) -> dict:
    """
    Announce detection success.

    Uses the speaker set via set_speaker(), falling back to local TTS if it
    is missing or fails.
    """
    phrase = (text or "RADI").strip() or "RADI"
    cooldown = float(
        cooldown_s
        if cooldown_s is not None
        else os.getenv("IGRA_ANNOUNCE_COOLDOWN_S", "2.5")
    )
    if not _cooldown_ok(phrase, cooldown):
        return {"spoken": False, "reason": "cooldown", "text": phrase}

    custom_ok = False
    local_ok = False
    if _speaker is not None:
        try:
            custom_ok = bool(_speaker(phrase))
        except Exception as exc:
            logger.warning("Custom speaker failed: %s", exc)
    if not custom_ok:
        local_ok = speak_local(phrase)

    return {
        "spoken": bool(custom_ok or local_ok),
        "custom": custom_ok,
        "local": local_ok,
        "text": phrase,
    }


# Kept for callers from the camera thread; announce() is already synchronous.
announce_sync = announce
