"""
IGRA multi-player host.

When the IGRA service starts in game mode:
  1) wait for a person (person source and/or hand in camera)
  2) resolve name (face identity if known, else ask by speech)
  3) run one RPS match (first to 1 win)
  4) mark player done, wait for next person
  5) repeat until stopped

Person detection plugs in with set_person_source(fn), where fn() returns
{"person_present": bool, "identity": {"status": "known"|"unknown",
"name": str, "face_id": str}} or {}. Without one, a hand held in front of
the camera counts as a new player.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from approach import approach_player, bearing_from_landmarks
from leaderboard import LeaderboardStore
from robot_arm import RobotArm
from rps_game import RPSGame, Vision
from speech_io import speak

logger = logging.getLogger("igra.host")

PERSON_POLL_S = float(os.getenv("IGRA_PERSON_POLL_S", "0.8"))
BETWEEN_PLAYERS_S = float(os.getenv("IGRA_BETWEEN_PLAYERS_S", "3.0"))
LEAVE_CLEAR_S = float(os.getenv("IGRA_LEAVE_CLEAR_S", "2.5"))
MAX_PLAYERS = int(os.getenv("IGRA_MAX_PLAYERS", "50"))

_person_source: Optional[Callable[[], dict[str, Any]]] = None


def set_person_source(fn: Optional[Callable[[], dict[str, Any]]]) -> None:
    global _person_source
    _person_source = fn


@dataclass
class PlayerSlot:
    name: str
    face_id: Optional[str] = None
    source: str = "unknown"  # face | speech | default
    result: Optional[str] = None
    finished_at: Optional[float] = None


@dataclass
class HostState:
    phase: str = "idle"  # idle|scan|approach|play|between|done|error
    message: str = ""
    current_player: Optional[str] = None
    players_done: list[dict[str, Any]] = field(default_factory=list)
    players_total: int = 0
    round_hint: str = ""
    last_error: Optional[str] = None
    started_at: Optional[float] = None
    stopped: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "message": self.message,
            "current_player": self.current_player,
            "players_done": list(self.players_done),
            "players_total": self.players_total,
            "round_hint": self.round_hint,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "stopped": self.stopped,
            "uptime_seconds": (time.time() - self.started_at) if self.started_at else 0.0,
        }


class GameHost:
    def __init__(
        self,
        *,
        leaderboard: LeaderboardStore,
        dry_run_arm: bool = False,
        show_window: bool = False,
        stop_event: threading.Event | None = None,
        on_update: Callable[[], None] | None = None,
    ) -> None:
        self.leaderboard = leaderboard
        self.dry_run_arm = dry_run_arm
        self.show_window = show_window
        self.stop_event = stop_event or threading.Event()
        self.on_update = on_update
        self.state = HostState()
        self._thread: Optional[threading.Thread] = None
        self._seen_face_ids: set[str] = set()
        self._seen_names: set[str] = set()

    def _publish(self, phase: str, message: str = "", **kwargs: Any) -> None:
        self.state.phase = phase
        self.state.message = message
        for k, v in kwargs.items():
            if hasattr(self.state, k):
                setattr(self.state, k, v)
        if self.on_update:
            try:
                self.on_update()
            except Exception:
                pass
        logger.info("HOST [%s] %s", phase, message)

    def _stopped(self) -> bool:
        return self.stop_event.is_set() or self.state.stopped

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.stop_event.clear()
        self.state = HostState(started_at=time.time())
        self._seen_face_ids.clear()
        self._seen_names.clear()
        self._thread = threading.Thread(target=self._run_loop, name="igra-host", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self.state.stopped = True
        self.stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    def _vision_snapshot(self) -> dict[str, Any]:
        if _person_source is None:
            return {}
        try:
            return _person_source() or {}
        except Exception as exc:
            logger.debug("person source failed: %s", exc)
            return {}

    def _current_person(self) -> Optional[PlayerSlot]:
        """Best-effort person from the person source identity."""
        status = self._vision_snapshot()
        if not status:
            return None
        present = bool(status.get("person_present"))
        identity = status.get("identity") or {}
        if not isinstance(identity, dict):
            identity = {}
        name = identity.get("name") or identity.get("display_name")
        face_id = identity.get("face_id")
        id_status = identity.get("status")  # known|unknown

        if face_id and face_id in self._seen_face_ids:
            return None  # already played
        if name and str(name).strip().lower() in self._seen_names:
            return None

        if id_status == "known" and name:
            return PlayerSlot(name=str(name).strip(), face_id=face_id, source="face")
        if present:
            # unknown person in front — will ask name in RPSGame
            return PlayerSlot(name="", face_id=face_id, source="unknown")
        return None

    def _wait_for_next_player(self, vision: Vision) -> Optional[PlayerSlot]:
        """
        Block until a new person is available or stop.
        Prefer face identity; fall back to 'hand in camera' as presence.
        """
        self._publish("scan", "Tražim sledećeg igrača…")
        clear_since: Optional[float] = None

        while not self._stopped():
            slot = self._current_person()
            if slot is not None:
                # require a short "clear" after previous player so we don't re-grab same person
                if clear_since is not None and time.time() - clear_since < LEAVE_CLEAR_S:
                    time.sleep(0.2)
                    continue
                return slot

            # fallback presence: any hand in frame for ~1s
            try:
                if vision.is_showing_something(duration=1.2, min_frames=4):
                    return PlayerSlot(name="", source="hand")
            except Exception:
                pass

            # track empty scene for leave-clear
            status = self._vision_snapshot()
            present = bool(status.get("person_present"))
            if not present:
                if clear_since is None:
                    clear_since = time.time()
            else:
                clear_since = None

            time.sleep(PERSON_POLL_S)
        return None

    def _estimate_bearing(self, vision: Vision) -> float | None:
        """Use current hand/person x in the IGRA camera as yaw proxy."""
        try:
            xs: list[float] = []
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and len(xs) < 8:
                frame = vision._read_frame()
                if frame is None or vision.detector is None:
                    time.sleep(0.05)
                    continue
                results = vision.detector.find_hands(frame)
                if results.multi_hand_landmarks:
                    b = bearing_from_landmarks(results.multi_hand_landmarks[0])
                    if b is not None:
                        xs.append(b)
                time.sleep(0.05)
            if not xs:
                return None
            return sum(xs) / len(xs)
        except Exception as exc:
            logger.debug("bearing estimate failed: %s", exc)
            return None

    def _approach(self, player: PlayerSlot, vision: Vision | None = None) -> None:
        """Face player (turn) then short step forward."""
        label = player.name or "igrač"
        self._publish("approach", f"Prilazim: {label}", current_player=label or "igrač")

        bearing = self._estimate_bearing(vision) if vision is not None else None
        if bearing is not None:
            self._publish(
                "approach",
                f"Prilazim {label} (yaw {bearing:.2f} rad / {bearing * 57.3:.0f}°)",
                current_player=label,
            )
        else:
            self._publish("approach", f"Prilazim {label} (bez yaw procene — samo napred)", current_player=label)

        result = approach_player(bearing_rad=bearing, distance_m=None)
        self.state.round_hint = f"nav ok={result.get('ok')} dry={not result.get('execute')} err={result.get('error')}"
        logger.info("Approach result: %s", result)
        if not result.get("ok"):
            # Non-fatal: still greet and play in place
            self._publish(
                "approach",
                f"Nav nije uspeo ({result.get('error')}) — igram u mestu",
                current_player=label,
            )

        speak(f"Zdravo{' ' + label if player.name else ''}.")
        time.sleep(0.3)

    def _play_one(self, player: PlayerSlot, vision: Vision, arm: RobotArm) -> PlayerSlot:
        name = player.name.strip() if player.name else ""
        game = RPSGame(
            arm,
            vision,
            player_name=name or "igrač",
            leaderboard=self.leaderboard,
        )
        if player.source == "face" and name:
            # Known face: greet and go straight into play (skip ask/name)
            self._publish("play", f"Poznat igrač {name} — krećemo", current_player=name)
            from rps_game import State

            game.state = State.PLAY
            game.say(f"Zdravo ponovo, {name}. Igramo papir-kamen-makaze!")
        else:
            self._publish(
                "play",
                "Nova partija — pitam ime" if not name else f"Partija: {name}",
                current_player=name or "?",
            )

        result = game.run()
        player.name = game.player_name or player.name or "igrač"
        player.result = None if result is None else result.value
        player.finished_at = time.time()
        return player

    def _run_loop(self) -> None:
        arm = RobotArm(dry_run=self.dry_run_arm)
        vision = Vision(show_window=self.show_window)
        try:
            # Vision camera shared for whole session
            vision.start()
            self._publish("scan", "IGRA session started — čekam igrače")

            while not self._stopped() and self.state.players_total < MAX_PLAYERS:
                player = self._wait_for_next_player(vision)
                if player is None or self._stopped():
                    break

                self._approach(player, vision)
                if self._stopped():
                    break

                try:
                    finished = self._play_with_shared_vision(player, vision, arm)
                except Exception as exc:
                    logger.exception("Match failed")
                    self.state.last_error = str(exc)
                    self._publish("error", f"Greška u partiji: {exc}")
                    time.sleep(2.0)
                    continue

                self.state.players_total += 1
                if finished.face_id:
                    self._seen_face_ids.add(finished.face_id)
                if finished.name:
                    self._seen_names.add(finished.name.strip().lower())
                self.state.players_done.append(
                    {
                        "name": finished.name,
                        "face_id": finished.face_id,
                        "source": finished.source,
                        "result": finished.result,
                        "finished_at": finished.finished_at,
                    }
                )
                self._publish(
                    "between",
                    f"Gotovo sa {finished.name}. Leaderboard ažuriran. Sledeći…",
                    current_player=None,
                )
                speak("Sledeći igrač, molim.")
                time.sleep(BETWEEN_PLAYERS_S)
                self._wait_person_clear()

            self._publish("done", "Sesija završena — svi obrađeni ili stop.")
            speak("To je to. Pogledajte rang listu.")
        except Exception as exc:
            logger.exception("Host crashed")
            self.state.last_error = str(exc)
            self._publish("error", str(exc))
        finally:
            try:
                vision.stop()
            except Exception:
                pass

    def _play_with_shared_vision(self, player: PlayerSlot, vision: Vision, arm: RobotArm) -> PlayerSlot:
        """
        RPSGame normally starts/stops vision. We patch start/stop no-ops so the
        host keeps the camera for the whole multi-player session.
        """
        orig_start = vision.start
        orig_stop = vision.stop
        vision.start = lambda: None  # type: ignore
        vision.stop = lambda: None  # type: ignore
        try:
            return self._play_one(player, vision, arm)
        finally:
            vision.start = orig_start  # type: ignore
            vision.stop = orig_stop  # type: ignore

    def _wait_person_clear(self) -> None:
        """Best-effort wait until the person source says nobody is there (or timeout)."""
        deadline = time.time() + 15.0
        while not self._stopped() and time.time() < deadline:
            status = self._vision_snapshot()
            if not status:
                break
            if not status.get("person_present"):
                time.sleep(LEAVE_CLEAR_S)
                break
            time.sleep(0.5)
