"""
IGRA logic tests that run without the robot (arm in dry-run, fakes for
camera, speech and speaker).

Run from the project root:
    python -m unittest discover tests -v
"""

from __future__ import annotations

import math
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import announce  # noqa: E402
import camera_source  # noqa: E402
import game_host  # noqa: E402
import rps_game  # noqa: E402
import speech_io  # noqa: E402
from robot_arm import BEATS, Gesture, Result, RobotArm, judge  # noqa: E402
from rps_game import RPSGame, State  # noqa: E402


class RecordingArm(RobotArm):
    """Dry-run arm that records hand poses and arm moves instead of sleeping."""

    def __init__(self) -> None:
        super().__init__(dry_run=True)
        self.events: list[tuple] = []

    def send_hand(self, pos, toq, side=None) -> bool:
        self.events.append(("hand", list(pos)))
        return True

    def _rpc(self, service, method, body):
        if method == "PlanningMove":
            right = body["target"]["joints"][7:]
            self.events.append(("move", "home" if right == self.RIGHT_HOME else right[4]))
        return {"info": {"current_action": self.TARGET_ACTION, "status": "McActionStatus_RUNNING"}}


class FakeLeaderboard:
    def __init__(self) -> None:
        self.matches: list[dict] = []

    def record_match(self, **kwargs) -> None:
        self.matches.append(kwargs)


def pose(name: str) -> list[int]:
    return RobotArm.POSES[name][0]


class JudgeTests(unittest.TestCase):
    def test_every_combination(self) -> None:
        for user in Gesture:
            for robot in Gesture:
                expected = (
                    Result.DRAW if user == robot
                    else Result.WIN if BEATS[user] == robot
                    else Result.LOSE
                )
                self.assertEqual(judge(user, robot), expected, (user, robot))

    def test_rock_beats_scissors(self) -> None:
        self.assertEqual(judge(Gesture.ROCK, Gesture.SCISSORS), Result.WIN)
        self.assertEqual(judge(Gesture.SCISSORS, Gesture.ROCK), Result.LOSE)


class HandPoseTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch("robot_arm.time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.arm = RecordingArm()

    def test_rps_poses(self) -> None:
        self.assertEqual(pose("open"), [0, 0, 0, 0, 0, 0])  # paper
        self.assertEqual(pose("close"), [2000] * 6)  # rock
        self.assertEqual(pose("scissors"), [2000, 2000, 0, 0, 2000, 2000])
        self.assertEqual(pose("point"), [2000, 2000, 0, 2000, 2000, 2000])  # YOU
        self.assertEqual(pose("thumbs_up"), [0, 0, 2000, 2000, 2000, 2000])

    def test_closing_thumb_goes_last(self) -> None:
        self.arm.set_hand_pose("close")
        self.assertEqual(
            self.arm.events,
            [("hand", [0, 0, 2000, 2000, 2000, 2000]), ("hand", [2000] * 6)],
        )

    def test_open_thumb_is_one_command(self) -> None:
        self.arm.set_hand_pose("thumbs_up")
        self.assertEqual(self.arm.events, [("hand", pose("thumbs_up"))])


class VerdictTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch("robot_arm.time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.arm = RecordingArm()

    def _moves(self) -> list:
        return [value for kind, value in self.arm.events if kind == "move"]

    def test_win_is_you_then_thumb_up(self) -> None:
        self.arm.show_verdict(Result.WIN)
        hands = [value for kind, value in self.arm.events if kind == "hand"]
        self.assertEqual(hands[1], pose("point"))
        self.assertEqual(hands[-1], pose("thumbs_up"))
        self.assertEqual(self._moves(), [RobotArm.RIGHT_WRIST_ROLL])

    def test_lose_rolls_wrist_half_turn(self) -> None:
        self.arm.show_verdict(Result.LOSE)
        (roll,) = self._moves()
        self.assertAlmostEqual(roll, RobotArm.RIGHT_WRIST_ROLL - math.pi)
        lo, hi = RobotArm.BOUNDS["wrist"]
        self.assertTrue(lo <= roll <= hi)

    def test_wrist_rolls_while_pointing(self) -> None:
        """The wrist turns before the thumb pose, so thumb-up never flips to down."""
        self.arm.show_verdict(Result.LOSE)
        kinds = [kind for kind, _ in self.arm.events]
        self.assertLess(kinds.index("move"), len(kinds) - 1)
        self.assertEqual(self.arm.events[-1], ("hand", pose("thumbs_up")))

    def test_rest_goes_home_once(self) -> None:
        self.arm.rest()
        self.arm.rest()
        self.assertEqual(self._moves(), ["home"])
        self.assertEqual(self.arm.events[-1], ("hand", pose("open")))

    def test_rest_after_raise_goes_home_again(self) -> None:
        self.arm.rest()
        self.arm.raise_arm()
        self.arm.rest()
        self.assertEqual(self._moves().count("home"), 2)


class ResultTests(unittest.TestCase):
    def setUp(self) -> None:
        for target in ("robot_arm.time.sleep", "rps_game.head_screen.warm_up"):
            patcher = mock.patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)
        speak = mock.patch("rps_game.speak")
        self.spoken = speak.start()
        self.addCleanup(speak.stop)
        flash = mock.patch("rps_game.head_screen.flash")
        self.flash = flash.start()
        self.addCleanup(flash.stop)

        self.arm = RecordingArm()
        self.leaderboard = FakeLeaderboard()
        self.game = RPSGame(self.arm, vision=None, player_name="Ana", leaderboard=self.leaderboard)
        self.game.robot_choice = Gesture.ROCK

    def said(self) -> list[str]:
        return [c.args[0] for c in self.spoken.call_args_list]

    def test_unrecognized_move_replays_without_scoring(self) -> None:
        self.game.user_choice = None
        self.assertIs(self.game._result(), State.PLAY)
        self.assertEqual(self.said(), ["Ne vidim lepo, hajmo ponovo."])
        self.assertEqual(self.leaderboard.matches, [])

    def test_deaf_unrecognized_shows_ponovo_silently(self) -> None:
        self.game.deaf = True
        self.game.user_choice = None
        self.assertIs(self.game._result(), State.PLAY)
        self.flash.assert_called_once_with("Ponovo")
        self.assertEqual(self.said(), [])
        self.assertEqual(self.arm.events[-1], ("hand", pose("open")))

    def test_draw_replays(self) -> None:
        self.game.user_choice = Gesture.ROCK
        self.assertIs(self.game._result(), State.PLAY)
        self.assertEqual(self.leaderboard.matches, [])

    def test_win_scores_player_and_rests(self) -> None:
        self.game.user_choice = Gesture.PAPER
        self.assertIs(self.game._result(), State.END)
        self.assertEqual(self.said(), ["Pobedio si, Ana!"])
        (match,) = self.leaderboard.matches
        self.assertEqual((match["winner"], match["comm_mode"]), ("human", "speech"))
        self.assertEqual(self.arm.events[-2:], [("hand", pose("open")), ("move", "home")])

    def test_deaf_loss_signs_verdict_then_rests(self) -> None:
        self.game.deaf = True
        self.game.user_choice = Gesture.SCISSORS
        self.assertIs(self.game._result(), State.END)
        self.assertEqual(self.said(), [])
        moves = [value for kind, value in self.arm.events if kind == "move"]
        self.assertAlmostEqual(moves[0], RobotArm.RIGHT_WRIST_ROLL - math.pi)
        self.assertEqual(moves[-1], "home")
        (match,) = self.leaderboard.matches
        self.assertEqual((match["winner"], match["comm_mode"]), ("robot", "sign"))

    def test_round_limit_ends_game(self) -> None:
        self.game.round_index = RPSGame.MAX_ROUNDS
        self.assertIs(self.game._play(), State.END)


class SignFallbackTests(unittest.TestCase):
    """No voice answer: the camera decides between deaf path, fist-yes and asking again."""

    def setUp(self) -> None:
        patcher = mock.patch("rps_game.head_screen.warm_up")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.vision = rps_game.Vision.__new__(rps_game.Vision)
        self.game = RPSGame(RecordingArm(), self.vision, leaderboard=FakeLeaderboard())

    def _signs(self, **counts: int) -> State:
        self.vision._collect = lambda duration: (20, Counter(counts))
        return self.game._detect_deaf()

    def test_fist_means_yes(self) -> None:
        self.assertIs(self._signs(rock=12), State.PLAY)
        self.assertFalse(self.game.deaf)

    def test_deaf_sign_starts_sign_invite(self) -> None:
        self.assertIs(self._signs(deaf=9, rock=2), State.ASK_GESTURE)
        self.assertTrue(self.game.deaf)

    def test_other_hand_asks_again(self) -> None:
        self.assertIs(self._signs(paper=9), State.ASK)

    def test_too_few_frames_asks_again(self) -> None:
        self.assertIs(self._signs(rock=2), State.ASK)


class ListenTests(unittest.TestCase):
    def tearDown(self) -> None:
        speech_io.set_listener(None)

    def test_uses_plugged_listener(self) -> None:
        speech_io.set_listener(lambda timeout: "da")
        self.assertEqual(speech_io.listen(3.0), "da")

    def test_timeout_is_none(self) -> None:
        speech_io.set_listener(lambda timeout: None)
        self.assertIsNone(speech_io.listen(0.5))

    def test_failing_listener_is_none(self) -> None:
        def broken(timeout):
            raise OSError("mic gone")

        speech_io.set_listener(broken)
        self.assertIsNone(speech_io.listen(0.5))


class SpeechParsingTests(unittest.TestCase):
    def test_yes_and_no(self) -> None:
        for text in ("da", "Da, hajde!", "može", "ok"):
            self.assertTrue(speech_io.is_yes(text), text)
        for text in ("ne", "Ne želim", "nemoj"):
            self.assertTrue(speech_io.is_no(text), text)

    def test_parse_name(self) -> None:
        self.assertEqual(speech_io.parse_name("Zovem se Marko."), "Marko")
        self.assertEqual(speech_io.parse_name("Ja sam Ana Petrović"), "Ana Petrović")
        self.assertIsNone(speech_io.parse_name("da"))


class AnnounceTests(unittest.TestCase):
    """The robot must say each line once: plugged speaker, local TTS only as fallback."""

    def tearDown(self) -> None:
        announce.set_speaker(None)

    def _announce(self, *, speaker_ok: bool | None) -> tuple[dict, mock.Mock]:
        if speaker_ok is not None:
            announce.set_speaker(lambda text: speaker_ok)
        with mock.patch.object(announce, "speak_local", mock.Mock(return_value=True)) as local:
            result = announce.announce("test", cooldown_s=0)
        return result, local

    def test_speaker_only_when_it_works(self) -> None:
        result, local = self._announce(speaker_ok=True)
        self.assertTrue(result["custom"])
        local.assert_not_called()

    def test_local_voice_is_fallback(self) -> None:
        result, local = self._announce(speaker_ok=False)
        local.assert_called_once()
        self.assertTrue(result["spoken"])

    def test_no_speaker_uses_local(self) -> None:
        _, local = self._announce(speaker_ok=None)
        local.assert_called_once()


class CameraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.opened: list = []
        opened = self.opened

        class FakeCapture:
            def __init__(self, target, *args) -> None:
                opened.append(target)

            def isOpened(self) -> bool:
                return True

        self.FakeCapture = FakeCapture
        patcher = mock.patch("cv2.VideoCapture", FakeCapture)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(camera_source.set_camera_opener, None)

    def test_laptop_index_uses_opencv(self) -> None:
        _, resolved, backend = camera_source.open_camera("0")
        self.assertEqual((resolved, backend), ("0", "opencv"))
        self.assertEqual(self.opened, [0])

    def test_dev_path_uses_opencv(self) -> None:
        _, _, backend = camera_source.open_camera("/dev/video4")
        self.assertEqual(backend, "opencv")
        self.assertEqual(self.opened[0], "/dev/video4")

    def test_plugged_opener_wins(self) -> None:
        camera_source.set_camera_opener(lambda device: self.FakeCapture(f"robot:{device}"))
        _, resolved, backend = camera_source.open_camera("chest")
        self.assertEqual((resolved, backend), ("chest", "custom"))
        self.assertEqual(self.opened, ["robot:chest"])


class PresenceTests(unittest.TestCase):
    def _person(self, status: dict):
        host = game_host.GameHost(leaderboard=FakeLeaderboard(), dry_run_arm=True)
        with mock.patch.object(host, "_vision_snapshot", return_value=status):
            return host._current_person()

    def test_running_vision_is_not_a_person(self) -> None:
        self.assertIsNone(self._person({"active": True, "person_present": False}))

    def test_present_unknown_person(self) -> None:
        slot = self._person({"active": True, "person_present": True, "identity": {}})
        self.assertEqual((slot.name, slot.source), ("", "unknown"))

    def test_known_face_skips_name(self) -> None:
        slot = self._person({"person_present": True, "identity": {"status": "known", "name": "Ana", "face_id": "f1"}})
        self.assertEqual((slot.name, slot.source), ("Ana", "face"))


if __name__ == "__main__":
    unittest.main()
