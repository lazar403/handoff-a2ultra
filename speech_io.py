"""
Speech I/O for the RPS invite.

speak() goes through announce (custom speaker or local TTS).

listen() returns the next finalized user utterance. Plug any STT in with
set_listener(fn), where fn(timeout_s) -> str | None. Without one it reads a
typed line from the terminal, so the whole speech path runs on a laptop.
"""

from __future__ import annotations

import re
import select
import sys
from typing import Callable, Optional

from announce import announce_sync

_listener: Optional[Callable[[float], Optional[str]]] = None


def set_listener(fn: Optional[Callable[[float], Optional[str]]]) -> None:
    """Route listen() through fn(timeout_s); None restores terminal input."""
    global _listener
    _listener = fn


def speak(text: str) -> None:
    phrase = (text or "").strip()
    if not phrase:
        return
    print(f"[ROBOT KAŽE] {phrase}")
    try:
        announce_sync(phrase, cooldown_s=0.0)
    except Exception as exc:
        print(f"[speak] {exc}")


def _console_listen(timeout: float) -> Optional[str]:
    if not sys.stdin or not sys.stdin.isatty():
        return None
    print(f"[STT] upiši odgovor (≤{timeout:.0f}s): ", end="", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        print()
        return None
    text = sys.stdin.readline().strip()
    return text or None


def listen(timeout: float, *, after_ts: float | None = None) -> Optional[str]:
    """
    Wait for a new user utterance; None on timeout.

    after_ts is accepted for callers that mark when the robot stopped talking
    (to skip its own TTS echo); buffering listeners can honor it themselves.
    """
    timeout = max(0.5, float(timeout))
    fn = _listener or _console_listen
    try:
        text = fn(timeout)
    except Exception as exc:
        print(f"[STT] listener failed: {exc}")
        return None
    if text:
        print(f"[STT] ← {text}")
        return text
    print("[STT] timeout")
    return None


_YES = re.compile(
    r"\b(da|može|moze|hajde|ajde|ajdemo|yes|yeah|ok|okay|želim|zelim|hoću|hocu)\b",
    re.I,
)
_NO = re.compile(r"\b(ne|no|nemoj|neću|necu|stop|odustajem)\b", re.I)


def is_yes(text: str | None) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if _NO.search(t) and not _YES.search(t):
        return False
    return bool(_YES.search(t)) or t in {"da", "yes", "ok", "hajde", "moze", "može"}


def is_no(text: str | None) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if is_yes(t) and not t.startswith("ne"):
        return False
    return bool(_NO.search(t)) or t in {"ne", "no"}


def parse_name(text: str | None) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(
        r"^(zovem se|ja sam|ime mi je|moje ime je|my name is|i am|i'm)\s+",
        "",
        t,
        flags=re.I,
    )
    t = t.strip(" .,!?:;\"'")
    if not t or is_yes(t) or is_no(t):
        return None
    parts = [p for p in re.split(r"\s+", t) if p]
    if not parts:
        return None
    name = " ".join(parts[:3])
    return name[:40] if len(name) >= 2 else None
