"""
IGRA service — multi-player RPS host + leaderboard API.

Default mode (IGRA_MODE=game):
  On start → detect players one-by-one → ask name → play first-to-1 → next.

Legacy debug (IGRA_MODE=detect):
  hand gesture → announce RADI.

Run:
  python igra_api.py --port 8105
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from announce import announce_sync  # noqa: E402
from camera_source import normalize_camera_id  # noqa: E402
from game_host import GameHost  # noqa: E402
from leaderboard import LeaderboardStore  # noqa: E402

logger = logging.getLogger("igra")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DEFAULT_PORT = int(os.getenv("IGRA_PORT", "8105"))
DEFAULT_CAMERA = os.getenv("IGRA_CAMERA", "0")
DEFAULT_DB = os.getenv("IGRA_GESTURE_DB", str(ROOT / "gestures.json"))
DEFAULT_LEADERBOARD = os.getenv("IGRA_LEADERBOARD_DB", str(ROOT / "data" / "igra_leaderboard.db"))
# game = multi-player RPS host (default); detect = legacy RADI loop
IGRA_MODE = (os.getenv("IGRA_MODE") or "game").strip().lower()
SHOW_WINDOW = os.getenv("IGRA_SHOW_WINDOW", "0").strip() not in {"0", "false", "False", "no"}
ANNOUNCE_TEXT = os.getenv("IGRA_ANNOUNCE_TEXT", "RADI")
ROBOT_ID = os.getenv("IGRA_ROBOT_ID", "agibot-a2-ultra")
DRY_ARM = os.getenv("IGRA_ARM_DRY_RUN", "0").strip() in {"1", "true", "True"}


class IgraState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mode = IGRA_MODE
        self.running = False
        self.started_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self.camera_requested = normalize_camera_id(DEFAULT_CAMERA)
        self.game: dict[str, Any] = {}
        self.leaderboard_path = DEFAULT_LEADERBOARD

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "service": "igra",
                "mode": self.mode,
                "running": self.running,
                "started_at": self.started_at,
                "last_error": self.last_error,
                "camera_requested": self.camera_requested,
                "game": dict(self.game),
                "leaderboard_path": self.leaderboard_path,
                "uptime_seconds": (time.time() - self.started_at) if self.started_at else 0.0,
            }


STATE = IgraState()
LEADERBOARD = LeaderboardStore(DEFAULT_LEADERBOARD)
HOST: Optional[GameHost] = None
_host_stop = threading.Event()


def _on_host_update() -> None:
    if HOST is None:
        return
    with STATE.lock:
        STATE.game = HOST.state.snapshot()
        STATE.running = not HOST.state.stopped and HOST.state.phase not in {"done", "idle"}
        if HOST.state.last_error:
            STATE.last_error = HOST.state.last_error


def start_game_host() -> None:
    global HOST
    stop_game_host()
    _host_stop.clear()
    with STATE.lock:
        STATE.mode = "game"
        STATE.running = True
        STATE.started_at = time.time()
        STATE.last_error = None
        STATE.game = {"phase": "starting", "message": "Pokrećem sesiju…"}
    HOST = GameHost(
        leaderboard=LEADERBOARD,
        dry_run_arm=DRY_ARM,
        show_window=SHOW_WINDOW,
        stop_event=_host_stop,
        on_update=_on_host_update,
    )
    HOST.start_background()
    logger.info("IGRA game host started (dry_arm=%s camera=%s)", DRY_ARM, DEFAULT_CAMERA)


def stop_game_host() -> None:
    global HOST
    _host_stop.set()
    if HOST is not None:
        HOST.stop(join_timeout=8.0)
        HOST = None
    with STATE.lock:
        STATE.running = False
        if STATE.game.get("phase") not in {"done", "error"}:
            STATE.game = {**STATE.game, "phase": "idle", "message": "Stopped", "stopped": True}


# --- legacy RADI detect loop (optional) ---
_detect_stop = threading.Event()
_detect_worker: Optional[threading.Thread] = None


def start_detect_worker() -> None:
    global _detect_worker
    if _detect_worker and _detect_worker.is_alive():
        return
    from gesture_database import GestureDatabase
    from gesture_utils import landmarks_to_vector
    from hand_gesture_detector import HandGestureDetector
    from camera_source import open_camera

    def _loop() -> None:
        import cv2

        db = GestureDatabase().load(DEFAULT_DB)
        try:
            cap, resolved, backend = open_camera(DEFAULT_CAMERA, startup_timeout=12.0)
        except Exception as exc:
            with STATE.lock:
                STATE.last_error = str(exc)
                STATE.running = False
            return
        detector = HandGestureDetector(max_hands=1)
        pending = None
        pending_n = 0
        last_fire = None
        stable_n = int(os.getenv("IGRA_STABLE_FRAMES", "3"))
        with STATE.lock:
            STATE.running = True
            STATE.started_at = time.time()
            STATE.mode = "detect"
        try:
            while not _detect_stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                if backend == "opencv" and str(resolved) in {"0", "1"}:
                    frame = cv2.flip(frame, 1)
                results = detector.find_hands(frame)
                if results.multi_hand_landmarks and db.real_samples():
                    vec = landmarks_to_vector(results.multi_hand_landmarks[0])
                    label, _ = db.match(vec)
                    if label is None:
                        pending, pending_n, last_fire = None, 0, None
                    else:
                        pending_n = pending_n + 1 if label == pending else 1
                        pending = label
                        if pending_n >= stable_n and label != last_fire:
                            last_fire = label
                            announce_sync(ANNOUNCE_TEXT)
                            with STATE.lock:
                                STATE.game = {
                                    "phase": "detect",
                                    "message": f"gesture={label}",
                                    "current_player": None,
                                }
                else:
                    pending, pending_n, last_fire = None, 0, None
                time.sleep(0.01)
        finally:
            detector.close()
            cap.release()
            with STATE.lock:
                STATE.running = False

    _detect_stop.clear()
    _detect_worker = threading.Thread(target=_loop, name="igra-detect", daemon=True)
    _detect_worker.start()


def stop_detect_worker() -> None:
    _detect_stop.set()
    if _detect_worker and _detect_worker.is_alive():
        _detect_worker.join(timeout=3.0)


def start_worker() -> None:
    mode = (os.getenv("IGRA_MODE") or STATE.mode or "game").strip().lower()
    if mode == "detect":
        start_detect_worker()
    else:
        start_game_host()


def stop_worker() -> None:
    stop_game_host()
    stop_detect_worker()


app = FastAPI(title="IGRA", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRecordRequest(BaseModel):
    player_name: str = Field(..., min_length=1)
    winner: str = Field(..., description="human | robot | draw")
    human_choice: Optional[str] = None
    robot_choice: Optional[str] = None
    comm_mode: Optional[str] = None
    posture_mode: Optional[str] = None
    robot_id: Optional[str] = None


class AnnounceRequest(BaseModel):
    text: str = "RADI"


@app.on_event("startup")
def _on_startup() -> None:
    auto = os.getenv("IGRA_AUTO_START_DETECT", "1").strip() not in {"0", "false", "False"}
    # Prefer game auto-start; env IGRA_AUTO_START_GAME overrides
    auto_game = os.getenv("IGRA_AUTO_START_GAME", "1" if auto else "0").strip() not in {
        "0",
        "false",
        "False",
    }
    if auto_game:
        start_worker()


@app.on_event("shutdown")
def _on_shutdown() -> None:
    stop_worker()


@app.get("/health")
def health() -> dict[str, Any]:
    snap = STATE.snapshot()
    ok = snap["running"] or snap["game"].get("phase") in {"done", "idle", "between", "scan", "play", "approach"}
    return {
        "status": "ok" if ok or snap["last_error"] is None else "degraded",
        "running": snap["running"],
        "mode": snap["mode"],
        "phase": snap["game"].get("phase"),
    }


@app.get("/status")
def status() -> dict[str, Any]:
    snap = STATE.snapshot()
    if HOST is not None:
        snap["game"] = HOST.state.snapshot()
    snap["leaderboard"] = LEADERBOARD.snapshot()
    return snap


@app.get("/leaderboard")
def leaderboard() -> dict[str, Any]:
    return LEADERBOARD.snapshot()


@app.post("/leaderboard/match")
def record_match(body: MatchRecordRequest) -> dict[str, Any]:
    winner = body.winner.strip().lower()
    if winner not in {"human", "robot", "draw"}:
        raise HTTPException(status_code=400, detail="winner must be human|robot|draw")
    return LEADERBOARD.record_match(
        player_name=body.player_name,
        robot_id=body.robot_id or ROBOT_ID,
        winner=winner,
        human_choice=body.human_choice,
        robot_choice=body.robot_choice,
        comm_mode=body.comm_mode,
        posture_mode=body.posture_mode,
    )


@app.post("/game/start")
def game_start() -> dict[str, Any]:
    start_game_host()
    return {"status": "starting", **STATE.snapshot()}


@app.post("/game/stop")
def game_stop() -> dict[str, Any]:
    stop_game_host()
    return {"status": "stopped", **STATE.snapshot()}


@app.post("/detect/start")
def detect_start() -> dict[str, Any]:
    stop_game_host()
    start_detect_worker()
    return {"status": "starting", **STATE.snapshot()}


@app.post("/detect/stop")
def detect_stop() -> dict[str, Any]:
    stop_detect_worker()
    return {"status": "stopped", **STATE.snapshot()}


@app.post("/announce")
def announce_now(body: AnnounceRequest) -> dict[str, Any]:
    return announce_sync(body.text or "RADI", cooldown_s=0.0)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "igra",
        "mode": STATE.mode,
        "docs": "/docs",
        "health": "/health",
        "status": "/status",
        "leaderboard": "/leaderboard",
        "game_start": "POST /game/start",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="IGRA multi-player RPS service")
    parser.add_argument("--host", default=os.getenv("IGRA_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    logger.info("Starting IGRA on %s:%s mode=%s", args.host, args.port, IGRA_MODE)
    uvicorn.run(
        "igra_api:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
