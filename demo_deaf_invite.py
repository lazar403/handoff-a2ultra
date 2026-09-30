"""
Hardware check for deaf-mode signs.

  python demo_deaf_invite.py          # invite: YOU → ROCK
  python demo_deaf_invite.py win      # verdict: YOU → thumb up
  python demo_deaf_invite.py lose     # verdict: YOU → thumb down

  IGRA_ARM_DRY_RUN=1 ...              # print-only, no robot
"""

from __future__ import annotations

import os
import sys

from robot_arm import Result, RobotArm

VERDICTS = {"win": Result.WIN, "lose": Result.LOSE}


def main(sign: str = "invite") -> None:
    dry = os.getenv("IGRA_ARM_DRY_RUN", "0").strip() in {"1", "true", "True"}
    arm = RobotArm(dry_run=dry)
    try:
        arm.start()
        if sign in VERDICTS:
            arm.raise_arm()
            arm.show_verdict(VERDICTS[sign])
        else:
            print("Showing deaf-mode invite: YOU then ROCK")
            arm.ask_play_gesture(repeats=1)
        arm.rest()
        print("Done.")
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            arm.rest()
        except Exception:
            pass
    finally:
        arm.running = False


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "invite")
