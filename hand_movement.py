"""
Lab demo: arm raise + 3 pumps + hold a hand pose.

Prefer robot_arm.RobotArm from game code. This file is only a quick hardware check.
"""

from __future__ import annotations

import os
import sys

from robot_arm import Gesture, RobotArm


def main(sign: str = "rock") -> None:
    dry = os.getenv("IGRA_ARM_DRY_RUN", "0").strip() in {"1", "true", "True"}
    arm = RobotArm(dry_run=dry)
    gesture = Gesture.from_label(sign) or Gesture.ROCK
    try:
        arm.start()
        arm.move_arm_pregame()
        arm.show_gesture(gesture)
        arm.rest()
        print("Gotovo.")
    except KeyboardInterrupt:
        print("Prekinuto - home.")
        try:
            arm.rest()
        except Exception:
            pass
    finally:
        arm.running = False


if __name__ == "__main__":
    sign = sys.argv[1] if len(sys.argv) > 1 else "rock"
    main(sign)
