"""
Papir-kamen-makaze state machine.

Invite (speech):
  1) pitaj za igru
  2) STT: da/ne  (speech_io.listen)
  3) pitaj ime
  4) STT: ime
  5) igraj runde dok ne bude 1 pobeda (nereseno → ponovi)

Vision/deaf path remains as fallback when no verbal answer.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections import Counter
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from camera_source import open_camera
import head_screen
from gesture_database import GestureDatabase
from gesture_utils import landmarks_to_vector
from hand_gesture_detector import HandGestureDetector
from leaderboard import LeaderboardStore
from robot_arm import Gesture, Result, RobotArm, judge
from speech_io import is_no, is_yes, listen, parse_name, speak

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = os.getenv("IGRA_GESTURE_DB", str(ROOT / "gestures.json"))
DEFAULT_CAMERA = os.getenv("IGRA_CAMERA", "0")
DEFAULT_LEADERBOARD = os.getenv(
    "IGRA_LEADERBOARD_DB", str(ROOT / "data" / "igra_leaderboard.db")
)
ROBOT_ID = os.getenv("IGRA_ROBOT_ID", "agibot-a2-ultra")

DEAF_LABELS = {"deaf", "gluvonem", "de", "gluv"}
# Hand labels accepted as "yes, I'll play" in the sign-invite path. There is no
# dedicated yes sign: the player answers the robot's ROCK invite with a fist or Y.
YES_LABELS = {"y", "rock", "kamen"}
RPS_LABELS = {"rock", "kamen", "paper", "papir", "scissors", "makaze"}


class State(Enum):
    ASK = auto()
    WAIT_ANSWER = auto()
    CHECK_CAMERA = auto()
    DETECT_DEAF = auto()
    ASK_GESTURE = auto()
    ASK_NAME = auto()
    WAIT_NAME = auto()
    PLAY = auto()
    RESULT = auto()
    END = auto()


class Vision:
    def __init__(
        self,
        camera: str | int | None = None,
        show_window: bool = False,
        db_path: str | None = None,
    ) -> None:
        self.camera = camera if camera is not None else DEFAULT_CAMERA
        self.show_window = show_window
        self.db_path = db_path or DEFAULT_DB
        self.cap = None
        self.backend = "opencv"
        self.resolved = "0"
        self.detector: Optional[HandGestureDetector] = None
        self.db = GestureDatabase()

    def start(self) -> None:
        self.db.load(self.db_path)
        print(
            f"Vision DB real={len(self.db.real_samples())} "
            f"match={len(self.db.match_samples)} labels={self.db.labels()}"
        )
        self.cap, self.resolved, self.backend = open_camera(self.camera, startup_timeout=12.0)
        self.detector = HandGestureDetector(max_hands=1)

    def stop(self) -> None:
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.show_window:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass

    def _read_frame(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if self.backend == "opencv" and str(self.resolved) in {"0", "1"}:
            import cv2

            frame = cv2.flip(frame, 1)
        return frame

    def read_label(self) -> tuple[bool, Optional[str]]:
        import cv2

        frame = self._read_frame()
        if frame is None or self.detector is None:
            return False, None
        results = self.detector.find_hands(frame)
        if self.show_window:
            self.detector.draw_landmarks(frame, results)
            cv2.imshow("RPS Vision", frame)
            cv2.waitKey(1)
        if not results.multi_hand_landmarks:
            return False, None
        vector = landmarks_to_vector(results.multi_hand_landmarks[0])
        label, _ = self.db.match(vector)
        return True, label

    def _collect(self, duration: float) -> tuple[int, Counter]:
        hand_frames = 0
        labels: Counter = Counter()
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            has_hand, label = self.read_label()
            if has_hand:
                hand_frames += 1
            if label:
                labels[label.lower()] += 1
        return hand_frames, labels

    def is_showing_something(self, duration: float = 3.0, min_frames: int = 5) -> bool:
        hand_frames, _ = self._collect(duration)
        return hand_frames >= min_frames

    def read_sign(self, duration: float = 3.0, min_hits: int = 5) -> Optional[str]:
        """Dominant intent sign over the window: "deaf", "yes" (fist) or None."""
        _, labels = self._collect(duration)
        deaf_hits = sum(labels[l] for l in DEAF_LABELS)
        yes_hits = sum(labels[l] for l in YES_LABELS)
        if deaf_hits >= min_hits and deaf_hits >= yes_hits:
            return "deaf"
        if yes_hits >= min_hits:
            return "yes"
        return None

    def wait_for_yes(self, timeout: float = 5.0, min_hits: int = 5) -> bool:
        hits = 0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, label = self.read_label()
            if label and label.lower() in YES_LABELS:
                hits += 1
                if hits >= min_hits:
                    return True
        return False

    def detect_gesture(self, duration: float = 3.0) -> Optional[Gesture]:
        _, labels = self._collect(duration)
        rps_hits: Counter = Counter()
        for lab, n in labels.items():
            if lab in RPS_LABELS:
                g = Gesture.from_label(lab)
                if g:
                    rps_hits[g] += n
        if not rps_hits:
            return None
        return rps_hits.most_common(1)[0][0]


class RPSGame:
    """
    Speech path: ASK → WAIT_ANSWER(STT da) → ASK_NAME → WAIT_NAME(STT) → PLAY*
    PLAY* repeats on draw or unrecognized move until one side wins once.
    """

    ANSWER_TIMEOUT = float(os.getenv("IGRA_ANSWER_TIMEOUT", "8"))
    NAME_TIMEOUT = float(os.getenv("IGRA_NAME_TIMEOUT", "10"))
    # After YOU→ROCK, give the player time to answer with a hand sign.
    GESTURE_ANSWER_TIMEOUT = float(os.getenv("IGRA_GESTURE_ANSWER_TIMEOUT", "10"))
    MAX_ROUNDS = int(os.getenv("IGRA_MAX_ROUNDS", "12"))  # safety

    def __init__(
        self,
        arm: RobotArm,
        vision: Vision,
        *,
        player_name: str = "player",
        leaderboard: LeaderboardStore | None = None,
    ) -> None:
        self.arm = arm
        self.vision = vision
        self.player_name = player_name
        self.leaderboard = leaderboard or LeaderboardStore(DEFAULT_LEADERBOARD)
        self.state = State.ASK
        self.deaf = False
        self.robot_choice: Optional[Gesture] = None
        self.user_choice: Optional[Gesture] = None
        self.result: Optional[Result] = None
        self.round_index = 0
        self._listen_after = 0.0

    def say(self, text: str) -> None:
        if not self.deaf:
            speak(text)
            # mark time so STT ignores our own TTS echo if mis-tagged
            self._listen_after = time.time() + 0.4

    def _listen(self, timeout: float) -> Optional[str]:
        return listen(timeout, after_ts=self._listen_after)

    def _ask(self) -> State:
        self.say("Da li želiš da igramo papir-kamen-makaze? Reci da, ili pokaži pesnicu.")
        return State.WAIT_ANSWER

    def _wait_answer(self) -> State:
        answer = self._listen(self.ANSWER_TIMEOUT)
        if answer is None:
            # no speech → maybe signing
            return State.CHECK_CAMERA
        if is_yes(answer):
            print(f"Prihvaćeno: {answer!r}")
            return State.ASK_NAME
        if is_no(answer):
            self.say("U redu, možda drugi put.")
            return State.END
        self.say("Nisam razumeo. Reci da ili ne.")
        return State.ASK

    def _check_camera(self) -> State:
        return State.DETECT_DEAF if self.vision.is_showing_something() else State.ASK

    def _detect_deaf(self) -> State:
        sign = self.vision.read_sign()
        self.deaf = sign == "deaf"
        print("Komunikacija:", "GEST" if self.deaf else "GOVOR", f"(znak: {sign})")
        if sign == "yes":
            # Voice answer never arrived, but the player showed a fist: play
            # without asking a name, since speech is evidently not getting through.
            print("Pristanak pesnicom → PLAY")
            return State.PLAY
        if self.deaf:
            # Do not speak further — hand channel only.
            print("Detected DEAF sign → robot will ask with YOU + ROCK hand poses")
            # Warm the screen link during the invite so "Ponovo" shows without lag.
            head_screen.warm_up()
            return State.ASK_GESTURE
        # no speech, no sign → ask again
        return State.ASK

    def _ask_gesture(self) -> State:
        """
        Deaf path — robot *physically* shows invite:
          1) raise right arm
          2) YOU  = point at player
          3) ROCK = fist (play RPS?)
        Then wait for player assent: rock / y from gestures.json.
        """
        print("=== DEAF INVITE: arm up → YOU → ROCK ===")
        try:
            # One clear cycle; second repeat if first assent window is empty.
            self.arm.ask_play_gesture(repeats=1)
            if self.vision.wait_for_yes(self.GESTURE_ANSWER_TIMEOUT):
                return self._deaf_yes_to_play()

            print("No assent — repeating YOU → ROCK once more")
            self.arm.ask_play_gesture(repeats=1)
            if self.vision.wait_for_yes(self.GESTURE_ANSWER_TIMEOUT):
                return self._deaf_yes_to_play()
        except Exception as exc:
            print(f"Deaf invite arm/hand failed: {exc}")
            # Still allow play if they keep signaling yes without robot motion
            if self.vision.wait_for_yes(3.0):
                return self._deaf_yes_to_play()

        print("Deaf invite: declined / timeout")
        try:
            self.arm.rest()
        except Exception:
            pass
        return State.END

    def _deaf_yes_to_play(self) -> State:
        if not self.player_name or self.player_name in {"player", "igrač", "igrac"}:
            self.player_name = "igrač"
        # Brief open hand = OK, then play
        try:
            self.arm.set_hand_pose("open")
            time.sleep(0.8)
        except Exception:
            pass
        print(f"Deaf yes accepted; player={self.player_name} → PLAY")
        return State.PLAY

    def _ask_name(self) -> State:
        # Only used on speech path after verbal "da"
        self.say("Kako se zoveš?")
        return State.WAIT_NAME

    def _wait_name(self) -> State:
        raw = self._listen(self.NAME_TIMEOUT)
        name = parse_name(raw)
        if name:
            self.player_name = name
            self.say(f"Drago mi je, {self.player_name}. Idemo!")
            return State.PLAY
        if raw:
            # use raw-ish if parse failed but something was said
            maybe = raw.strip().split()[0][:40]
            if len(maybe) >= 2 and not is_yes(maybe) and not is_no(maybe):
                self.player_name = maybe.capitalize()
                self.say(f"U redu, {self.player_name}. Idemo!")
                return State.PLAY
        self.say("Nisam čuo ime. Kako se zoveš?")
        # one soft retry then default
        raw2 = self._listen(self.NAME_TIMEOUT)
        name2 = parse_name(raw2) or (raw2.strip().split()[0] if raw2 and raw2.strip() else None)
        if name2 and len(name2) >= 2:
            self.player_name = name2[:40]
            self.say(f"Super, {self.player_name}. Idemo!")
        else:
            self.player_name = self.player_name or "igrač"
            self.say("U redu, krećemo.")
        return State.PLAY

    def _play(self) -> State:
        self.round_index += 1
        if self.round_index > self.MAX_ROUNDS:
            self.say("Previše ponovljenih rundi. Prekidamo.")
            return State.END

        print(f"=== Runda {self.round_index} ===")
        self.say("Kamen, papir, makaze!")
        self.arm.move_arm_pregame()
        self.robot_choice = random.choice(list(Gesture))

        detected: dict[str, Optional[Gesture]] = {}
        watcher = threading.Thread(
            target=lambda: detected.update(
                user=self.vision.detect_gesture(self.arm.SIGN_HOLD)
            ),
            daemon=True,
        )
        watcher.start()
        self.arm.show_gesture(self.robot_choice)
        watcher.join()
        self.user_choice = detected.get("user")
        print(
            f"Robot: {self.robot_choice.value}, "
            f"korisnik: {self.user_choice.value if self.user_choice else 'nije prepoznat'}"
        )
        return State.RESULT

    def _replay_round(self, spoken: str) -> State:
        """Signal "again" on the player's channel and go back to PLAY."""
        if self.deaf:
            head_screen.flash("Ponovo")
            self.arm.show_again()
        else:
            self.say(spoken)
        return State.PLAY

    def _result(self) -> State:
        assert self.robot_choice is not None
        if self.user_choice is None:
            # Unrecognized move must not count as a robot win.
            return self._replay_round("Ne vidim lepo, hajmo ponovo.")

        self.result = judge(self.user_choice, self.robot_choice)

        if self.result == Result.DRAW:
            # first-to-1: replay
            return self._replay_round("Nerešeno! Idemo ponovo.")

        # decisive
        if self.deaf:
            self.arm.show_verdict(self.result)
            self.arm.rest()
        else:
            if self.result == Result.WIN:
                self.say(f"Pobedio si, {self.player_name}!")
            else:
                self.say(f"Izgubio si, {self.player_name}!")

        winner = "human" if self.result == Result.WIN else "robot"
        try:
            self.leaderboard.record_match(
                player_name=self.player_name,
                robot_id=ROBOT_ID,
                winner=winner,
                human_choice=self.user_choice.value if self.user_choice else None,
                robot_choice=self.robot_choice.value,
                comm_mode="sign" if self.deaf else "speech",
            )
            print(f"[leaderboard] {self.player_name} → {winner}")
        except Exception as exc:
            print(f"[leaderboard] {exc}")

        self.arm.rest()
        return State.END

    def run(self) -> Optional[Result]:
        handlers = {
            State.ASK: self._ask,
            State.WAIT_ANSWER: self._wait_answer,
            State.CHECK_CAMERA: self._check_camera,
            State.DETECT_DEAF: self._detect_deaf,
            State.ASK_GESTURE: self._ask_gesture,
            State.ASK_NAME: self._ask_name,
            State.WAIT_NAME: self._wait_name,
            State.PLAY: self._play,
            State.RESULT: self._result,
        }
        self.arm.start()
        self.vision.start()
        try:
            while self.state != State.END:
                print(f"--> {self.state.name}")
                self.state = handlers[self.state]()
        finally:
            self.vision.stop()
            self.arm.stop()
        return self.result


def main() -> None:
    dry = os.getenv("IGRA_ARM_DRY_RUN", "0").strip() in {"1", "true", "True"}
    show = os.getenv("IGRA_SHOW_WINDOW", "0").strip() not in {"0", "false", "False"}
    game = RPSGame(
        RobotArm(dry_run=dry),
        Vision(show_window=show),
        player_name=os.getenv("IGRA_PLAYER_NAME", "igrač"),
    )
    try:
        result = game.run()
        print("Rezultat:", None if result is None else result.value)
        print("Igrač:", game.player_name)
    except KeyboardInterrupt:
        print("Prekinuto.")


if __name__ == "__main__":
    main()
