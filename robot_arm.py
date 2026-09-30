"""
Agibot A2 right-arm + hand control for RPS (HTTP-RPC).

This is the *output* path (robot moves).
Vision/gesture recognition stays in gesture_* / hand_gesture_detector / camera_source.
"""

from __future__ import annotations

import math
import os
import time
from enum import Enum
from typing import Optional

import requests


class Gesture(str, Enum):
    PAPER = "paper"
    ROCK = "rock"
    SCISSORS = "scissors"

    # Serbian aliases used in older scripts / speech
    @classmethod
    def from_label(cls, label: str | None) -> Optional["Gesture"]:
        if not label:
            return None
        key = str(label).strip().lower()
        mapping = {
            "paper": cls.PAPER,
            "papir": cls.PAPER,
            "rock": cls.ROCK,
            "kamen": cls.ROCK,
            "scissors": cls.SCISSORS,
            "makaze": cls.SCISSORS,
        }
        return mapping.get(key)


class Result(str, Enum):
    WIN = "win"  # user wins
    LOSE = "lose"  # user loses
    DRAW = "draw"


BEATS: dict[Gesture, Gesture] = {
    Gesture.PAPER: Gesture.ROCK,
    Gesture.ROCK: Gesture.SCISSORS,
    Gesture.SCISSORS: Gesture.PAPER,
}


def judge(user: Gesture, robot: Gesture) -> Result:
    """Result from the *user's* point of view."""
    if user == robot:
        return Result.DRAW
    return Result.WIN if BEATS[user] == robot else Result.LOSE


class RobotArm:
    """Right arm + hand on A2 via McMotion / McAction / SetHandCommand."""

    MC_URL = os.getenv(
        "IGRA_MC_URL",
        "http://192.168.100.100:56322/rpc/aimdk.protocol",
    )
    CONTROL_SOURCE = "ControlSource_SAFE"
    TARGET_ACTION = "McAction_RL_LOCOMOTION_ARM_EXT_PLANNING_MOVE"

    ROUTE_TO_TARGET = {
        "McAction_RL_WHOLE_BODY_EXT_JOINT_SERVO": [
            "McAction_RL_LOCOMOTION_DEFAULT",
            TARGET_ACTION,
        ],
        "McAction_RL_WHOLE_BODY_EXT_ONLINE_PLANNING": [
            "McAction_RL_LOCOMOTION_DEFAULT",
            TARGET_ACTION,
        ],
        "McAction_RL_LOCOMOTION_ARM_EXT_JOINT_SERVO": [
            "McAction_RL_LOCOMOTION_DEFAULT",
            TARGET_ACTION,
        ],
        "McAction_RL_LOCOMOTION_EXT_ONLINE_PLANNING": [
            "McAction_RL_LOCOMOTION_DEFAULT",
            TARGET_ACTION,
        ],
        "McAction_RL_LOCOMOTION_DEFAULT": [TARGET_ACTION],
        "McAction_RL_JOINT_DEFAULT": [TARGET_ACTION],
    }
    HOP_PAUSE = 3.0

    VEL = float(os.getenv("IGRA_ARM_VEL", "1.0"))
    ACC = float(os.getenv("IGRA_ARM_ACC", "1.0"))

    SHOULDER_90 = 1.5708
    ELBOW_UP = 1.50
    ELBOW_DOWN = 0.40
    RIGHT_WRIST_ROLL = 1.6
    # Thumbs down = the thumbs_up pose with the forearm rolled half a turn.
    THUMB_DOWN_WRIST_ROLL = RIGHT_WRIST_ROLL - math.pi
    SHAKES = 3
    STEP_WAIT = 1.5
    SIGN_HOLD = 3.0  # hold R/P/S pose ~3s (the "animation" window)

    LEFT_HOME = [0.0, 1.26, 0.0, -0.03, 0.0, 0.0, 0.0]
    RIGHT_HOME = [0.0, -1.26, 0.0, 0.03, 0.0, 0.0, 0.0]

    BOUNDS = {
        "j1": (-2.91, 2.91),
        "right_j4": (0.03, 2.00),
        "left_j4": (-2.00, -0.03),
        "wrist": (-2.00, 2.00),
    }

    HAND_SIDE = "right"
    POS_KEYS = [
        "thumb_pos_0",
        "thumb_pos_1",
        "index_pos",
        "middle_pos",
        "ring_pos",
        "pinky_pos",
    ]
    TOQ_KEYS = [
        "thumb_toq_0",
        "thumb_toq_1",
        "index_toq",
        "middle_toq",
        "ring_toq",
        "pinky_toq",
    ]
    MAX_POS = 2000

    HAND_TOQ = 6000
    # Pause between closing the fingers and closing the thumb over them.
    THUMB_DELAY = 0.4

    # Hand finger poses (SetHandCommand), order matches POS_KEYS:
    # thumb_0, thumb_1, index, middle, ring, pinky
    POSES: dict[str, tuple[list[int], list[int]]] = {
        "open": ([0, 0, 0, 0, 0, 0], [HAND_TOQ] * 6),
        "close": ([2000, 2000, 2000, 2000, 2000, 2000], [HAND_TOQ] * 6),
        "point": ([2000, 2000, 0, 2000, 2000, 2000], [HAND_TOQ] * 6),
        "question": ([2000, 2000, 1000, 2000, 2000, 2000], [HAND_TOQ] * 6),
        "scissors": ([2000, 2000, 0, 0, 2000, 2000], [HAND_TOQ] * 6),
        "thumbs_up": ([0, 0, 2000, 2000, 2000, 2000], [HAND_TOQ] * 6),
    }

    GESTURE_POSE: dict[Gesture, str] = {
        Gesture.PAPER: "open",
        Gesture.ROCK: "close",
        Gesture.SCISSORS: "scissors",
    }

    # Deaf-mode verdict from the user's POV: thumb up if they won, down if they lost.
    VERDICT_WRIST_ROLL: dict[Result, float] = {
        Result.WIN: RIGHT_WRIST_ROLL,
        Result.LOSE: THUMB_DOWN_WRIST_ROLL,
    }
    VERDICT_POINT_HOLD = 1.5

    def __init__(self, *, dry_run: bool | None = None) -> None:
        if dry_run is None:
            dry_run = os.getenv("IGRA_ARM_DRY_RUN", "0").strip() in {"1", "true", "True"}
        self.dry_run = dry_run
        self.running = False
        # Unknown until the first move, so the first rest() always goes home.
        self._at_home = False

    @property
    def hand_url(self) -> str:
        return f"{self.MC_URL}.McMotionService/SetHandCommand"

    def start(self) -> None:
        self.ensure_action()
        self.set_hand_pose("open")
        self.running = True

    def stop(self) -> None:
        if not self.running:
            return
        try:
            self.rest()
        finally:
            self.running = False

    def _header(self) -> dict:
        now = time.time()
        return {
            "timestamp": {"seconds": int(now), "nanos": int((now % 1) * 1e9)},
            "control_source": self.CONTROL_SOURCE,
        }

    def _rpc(self, service: str, method: str, body: dict) -> dict:
        if self.dry_run:
            print(f"[DRY ARM RPC] {service}/{method}")
            return {"info": {"current_action": self.TARGET_ACTION, "status": "McActionStatus_RUNNING"}}
        r = requests.post(f"{self.MC_URL}.{service}/{method}", json=body, timeout=5)
        r.raise_for_status()
        return r.json()

    def get_action(self) -> tuple[str, str]:
        info = self._rpc("McActionService", "GetAction", {"header": self._header()}).get("info", {})
        return info.get("current_action", ""), info.get("status", "")

    def set_action(self, action: str, wait: float = 15.0) -> None:
        print("SetAction ->", action)
        self._rpc(
            "McActionService",
            "SetAction",
            {"header": self._header(), "command": {"action": action, "ext_action": ""}},
        )
        if self.dry_run:
            return
        deadline = time.monotonic() + wait
        cur, status = "", ""
        while time.monotonic() < deadline:
            cur, status = self.get_action()
            if cur == action and status in ("McActionStatus_RUNNING", "McActionStatus_DONE"):
                return
            time.sleep(0.5)
        raise RuntimeError(f"SetAction({action}) failed, robot still {cur}/{status}")

    def ensure_action(self) -> None:
        cur, status = self.get_action()
        print("GetAction:", cur, status)
        if cur == self.TARGET_ACTION or self.dry_run:
            return
        if cur not in self.ROUTE_TO_TARGET:
            raise RuntimeError(f"No route from {cur} to {self.TARGET_ACTION}")
        hops = self.ROUTE_TO_TARGET[cur]
        for i, hop in enumerate(hops):
            self.set_action(hop)
            if i < len(hops) - 1:
                time.sleep(self.HOP_PAUSE)

    def _validate(self, left: list[float], right: list[float]) -> None:
        for arm, j in (("left", left), ("right", right)):
            lo, hi = self.BOUNDS["j1"]
            if not lo <= j[0] <= hi:
                raise ValueError(f"{arm} j1={j[0]} out of range")
            lo, hi = self.BOUNDS[f"{arm}_j4"]
            if not lo <= j[3] <= hi:
                raise ValueError(f"{arm} j4={j[3]} out of range")
            lo, hi = self.BOUNDS["wrist"]
            for k in (4, 5, 6):
                if not lo <= j[k] <= hi:
                    raise ValueError(f"{arm} j{k + 1}={j[k]} out of range")

    def move(self, left: list[float], right: list[float], wait: float | None = None) -> None:
        if wait is None:
            wait = self.STEP_WAIT
        self._validate(left, right)
        self.ensure_action()
        body = {
            "header": self._header(),
            "group": "McPlanningGroup_DUAL_ARM",
            "mode": "McPlanningMode_DEFAULT",
            "target": {"type": "JOINT", "joints": left + right},
            "param": {"velocity_scale": self.VEL, "acceleration_scale": self.ACC},
        }
        print("PlanningMove:", self._rpc("McMotionService", "PlanningMove", body))
        self._at_home = left == self.LEFT_HOME and right == self.RIGHT_HOME
        time.sleep(wait)

    def _right_arm(self, shoulder: float, elbow: float, wrist_roll: float | None = None) -> list[float]:
        j = self.RIGHT_HOME.copy()
        j[0] = shoulder
        j[3] = elbow
        j[4] = self.RIGHT_WRIST_ROLL if wrist_roll is None else wrist_roll
        return j

    def raise_arm(self) -> None:
        self.move(self.LEFT_HOME, self._right_arm(self.SHOULDER_90, self.ELBOW_DOWN), wait=4.0)

    def go_home(self) -> None:
        self.move(self.LEFT_HOME, self.RIGHT_HOME, wait=4.0)

    def rest(self) -> None:
        """Default state: open hand, arms down beside the body."""
        self.set_hand_pose("open")
        if not self._at_home:
            self.go_home()

    def move_arm_pregame(self) -> None:
        """Fist + raise + 3 forearm pumps (~countdown animation)."""
        self.set_hand_pose("close")
        self.raise_arm()
        for i in range(self.SHAKES):
            print(f"{i + 1}...")
            self.move(self.LEFT_HOME, self._right_arm(self.SHOULDER_90, self.ELBOW_UP))
            self.move(self.LEFT_HOME, self._right_arm(self.SHOULDER_90, self.ELBOW_DOWN))

    def send_hand(self, pos: list[int], toq: list[int], side: Optional[str] = None) -> bool:
        side = side or self.HAND_SIDE
        pos_f = [float(max(0, min(self.MAX_POS, p))) for p in pos]
        now = time.time()
        payload = {
            "header": {
                "timestamp": {
                    "seconds": int(now),
                    "nanos": int((now % 1) * 1e9),
                    "ms_since_epoch": int(now * 1000),
                },
                "control_source": self.CONTROL_SOURCE,
            },
            "data": {
                side: {
                    "agi_hand": {
                        "finger": {
                            "pos": dict(zip(self.POS_KEYS, pos_f)),
                            "toq": dict(zip(self.TOQ_KEYS, map(float, toq))),
                        }
                    }
                }
            },
        }
        if self.dry_run:
            print(f"[DRY HAND] {side} pos={pos_f}")
            return True
        r = requests.post(self.hand_url, json=payload, timeout=5)
        print(side, r.status_code, r.text[:200])
        return r.ok

    def set_hand_pose(self, name: str) -> None:
        """Apply a pose; a closing thumb goes last so it lands over the fingers."""
        pos, toq = self.POSES[name]
        if pos[0] > 0 or pos[1] > 0:
            self.send_hand([0, 0] + pos[2:], toq)
            if not self.dry_run:
                time.sleep(self.THUMB_DELAY)
        self.send_hand(pos, toq)

    def show_gesture(self, gesture: Gesture, hold: float | None = None) -> None:
        if hold is None:
            hold = self.SIGN_HOLD
        print(f"Robot shows: {gesture.value}")
        self.set_hand_pose(self.GESTURE_POSE[gesture])
        time.sleep(hold)

    def show_you(self, hold: float = 2.0) -> None:
        """YOU = index finger pointed at the player (raised arm faces them)."""
        print("Robot hand: YOU (point)")
        self.set_hand_pose("point")
        time.sleep(hold)

    def show_rock_sign(self, hold: float = 2.0) -> None:
        """Closed fist = ROCK / play RPS emblem."""
        print("Robot hand: ROCK")
        self.set_hand_pose("close")
        time.sleep(hold)

    def ask_play_gesture(self, *, repeats: int = 1) -> None:
        """
        Deaf-mode invite:
          raise arm → YOU (point) → ROCK (fist)
        Meaning: "You? Rock-paper-scissors?"
        """
        repeats = max(1, int(repeats))
        self.ensure_action()
        self.raise_arm()
        for i in range(repeats):
            if repeats > 1:
                print(f"Invite sign cycle {i + 1}/{repeats}")
            self.show_you(hold=2.2)
            self.show_rock_sign(hold=2.2)
            # brief open between repeats so the change is visible
            if i + 1 < repeats:
                self.set_hand_pose("open")
                time.sleep(0.6)

    def show_verdict(self, result: Result, hold: float | None = None) -> None:
        """
        Deaf-mode result: YOU (point at player), then thumb up (player won)
        or thumb down (player lost), other fingers clenched.
        """
        if hold is None:
            hold = self.SIGN_HOLD
        wrist_roll = self.VERDICT_WRIST_ROLL[result]
        print(f"Robot verdict: YOU → thumb {'up' if result == Result.WIN else 'down'}")
        self.set_hand_pose("point")
        # Rolling the forearm keeps the index aimed at the player, so the
        # wrist is set for the thumb while YOU is still showing.
        self.move(
            self.LEFT_HOME,
            self._right_arm(self.SHOULDER_90, self.ELBOW_DOWN, wrist_roll),
            wait=self.VERDICT_POINT_HOLD,
        )
        self.set_hand_pose("thumbs_up")
        time.sleep(hold)

    def show_again(self, hold: float = 1.5) -> None:
        """Deaf-mode "again" (draw or unrecognized move): open hand."""
        print("Robot hand: AGAIN (open)")
        self.set_hand_pose("open")
        time.sleep(hold)
