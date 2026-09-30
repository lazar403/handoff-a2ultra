"""
Physical approach / face-player.

Two motion primitives, supplied by the robot integration via set_mover():
  turn(angle_rad)      — rotate in place (+ = left/CCW)
  forward(distance_m)  — walk forward relative

Without a mover (or with IGRA_NAV_EXECUTE=0, the default) every step is
dry-run: logged, no motion.

Safety:
  IGRA_NAV_EXECUTE=0 (default) → dry-run only
  IGRA_NAV_EXECUTE=1           → real motion (E-stop recommended)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("igra.approach")

# Defaults tuned for short face-to-face approach in a demo booth
DEFAULT_FORWARD_M = float(os.getenv("IGRA_APPROACH_DISTANCE_M", "0.35"))
DEFAULT_MAX_TURN_RAD = float(os.getenv("IGRA_APPROACH_MAX_TURN_RAD", str(math.radians(45))))
DEFAULT_FOV_RAD = float(os.getenv("IGRA_CAMERA_FOV_RAD", str(math.radians(110))))  # fisheye-ish
NAV_EXECUTE = os.getenv("IGRA_NAV_EXECUTE", "0").strip() in {"1", "true", "True", "yes"}

_turn_fn: Optional[Callable[[float], None]] = None
_forward_fn: Optional[Callable[[float], None]] = None
_after_fn: Optional[Callable[[], None]] = None


def set_mover(
    turn: Optional[Callable[[float], None]],
    forward: Optional[Callable[[float], None]],
    after: Optional[Callable[[], None]] = None,
) -> None:
    """
    Register motion primitives. Each raises on failure.
    after() runs once the approach finishes (e.g. switch back to arm mode).
    """
    global _turn_fn, _forward_fn, _after_fn
    _turn_fn, _forward_fn, _after_fn = turn, forward, after


@dataclass
class ApproachResult:
    ok: bool
    dry_run: bool
    turned_rad: float = 0.0
    forward_m: float = 0.0
    detail: str = ""
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "turned_rad": self.turned_rad,
            "turned_deg": math.degrees(self.turned_rad),
            "forward_m": self.forward_m,
            "detail": self.detail,
            "error": self.error,
        }


def bearing_from_norm_x(x_norm: float, fov_rad: float = DEFAULT_FOV_RAD) -> float:
    """
    Convert normalized image x (0=left, 1=right) to yaw command.
    Person on the right → turn right (negative yaw; + = left/CCW).
    """
    x = max(0.0, min(1.0, float(x_norm)))
    # center 0.5 → 0; left (0) → +fov/2 (CCW); right (1) → -fov/2
    return (0.5 - x) * fov_rad


def bearing_from_landmarks(hand_landmarks, fov_rad: float = DEFAULT_FOV_RAD) -> Optional[float]:
    """Average landmark x as person/hand bearing proxy."""
    try:
        xs = [lm.x for lm in hand_landmarks.landmark]
        if not xs:
            return None
        return bearing_from_norm_x(sum(xs) / len(xs), fov_rad=fov_rad)
    except Exception:
        return None


def _can_execute(execute: bool | None) -> bool:
    execute = NAV_EXECUTE if execute is None else bool(execute)
    return execute and _turn_fn is not None and _forward_fn is not None


def spin_turn(angle_rad: float, *, execute: bool | None = None) -> ApproachResult:
    execute = _can_execute(execute)
    angle = max(-DEFAULT_MAX_TURN_RAD, min(DEFAULT_MAX_TURN_RAD, float(angle_rad)))
    if abs(angle) < math.radians(3):
        return ApproachResult(ok=True, dry_run=not execute, turned_rad=0.0, detail="turn skipped (small)")

    label = f"SpinTurn {angle:.3f} rad ({math.degrees(angle):.1f} deg)"
    if not execute:
        logger.info("[dry-run] %s", label)
        return ApproachResult(ok=True, dry_run=True, turned_rad=angle, detail=label)

    try:
        _turn_fn(angle)
        return ApproachResult(ok=True, dry_run=False, turned_rad=angle, detail=label)
    except Exception as exc:
        logger.exception("spin_turn failed")
        return ApproachResult(ok=False, dry_run=False, turned_rad=angle, error=str(exc))


def move_forward(distance_m: float, *, execute: bool | None = None) -> ApproachResult:
    execute = _can_execute(execute)
    distance = max(0.0, min(1.5, float(distance_m)))
    if distance < 0.05:
        return ApproachResult(ok=True, dry_run=not execute, forward_m=0.0, detail="forward skipped (small)")

    label = f"MoveForward {distance:.2f} m"
    if not execute:
        logger.info("[dry-run] %s", label)
        return ApproachResult(ok=True, dry_run=True, forward_m=distance, detail=label)

    try:
        _forward_fn(distance)
        return ApproachResult(ok=True, dry_run=False, forward_m=distance, detail=label)
    except Exception as exc:
        logger.exception("move_forward failed")
        return ApproachResult(ok=False, dry_run=False, forward_m=distance, error=str(exc))


def approach_player(
    *,
    bearing_rad: float | None = None,
    distance_m: float | None = None,
    execute: bool | None = None,
) -> dict[str, Any]:
    """
    Face the player (spin) then take a short step forward.

    bearing_rad: yaw to apply before walking (from camera), None → no turn
    distance_m: forward step, default IGRA_APPROACH_DISTANCE_M
    """
    execute = _can_execute(execute)
    distance = DEFAULT_FORWARD_M if distance_m is None else float(distance_m)

    steps: list[dict[str, Any]] = []
    turned = 0.0

    if bearing_rad is not None and abs(bearing_rad) >= math.radians(3):
        turn_res = spin_turn(bearing_rad, execute=execute)
        steps.append({"action": "spin", **turn_res.as_dict()})
        turned = turn_res.turned_rad
        if not turn_res.ok and execute:
            return {
                "ok": False,
                "execute": execute,
                "steps": steps,
                "error": turn_res.error or "spin failed",
            }

    fwd = move_forward(distance, execute=execute)
    steps.append({"action": "forward", **fwd.as_dict()})

    if execute and _after_fn is not None:
        try:
            _after_fn()
            steps.append({"action": "after", "ok": True})
        except Exception as exc:
            steps.append({"action": "after", "ok": False, "error": str(exc)})

    ok = all(s.get("ok", True) for s in steps)
    return {
        "ok": ok,
        "execute": execute,
        "turned_rad": turned,
        "forward_m": fwd.forward_m,
        "steps": steps,
        "error": None if ok else next((s.get("error") for s in steps if s.get("error")), "approach failed"),
    }
