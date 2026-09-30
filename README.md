# IGRA — rock-paper-scissors for a humanoid robot

Hand-gesture recognition (MediaPipe landmarks + augmented k-NN), a
rock-paper-scissors game state machine with a speech path and a sign-language
("deaf") path, robot arm/hand poses over HTTP-RPC, and a SQLite leaderboard.

Everything runs on a laptop with a webcam; robot hardware is optional.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python collect_gestures.py        # record samples (c = capture, q = quit)
python detect_gestures.py         # live recognition window

# one game, no robot motion, answers typed in the terminal
IGRA_ARM_DRY_RUN=1 IGRA_SHOW_WINDOW=1 python rps_game.py

# multi-player service + leaderboard API
IGRA_ARM_DRY_RUN=1 python igra_api.py --port 8105

python -m unittest discover tests -v
```

## Files

| File | Role |
|---|---|
| `gesture_utils.py` | 21 landmarks → 63-float vector, wrist-origin + scale normalized |
| `gesture_augment.py` | expand each real sample into ~40 mild rotated/scaled/noisy variants |
| `gesture_database.py` | JSON store, outlier pruning, k-NN vote with margin vs runner-up |
| `hand_gesture_detector.py` | MediaPipe wrapper (legacy `solutions` and Tasks backends) |
| `collect_gestures.py` / `detect_gestures.py` | capture and live-test tools |
| `rps_game.py` | game state machine: invite → name → rounds until one win |
| `robot_arm.py` | arm pumps, R/P/S hand poses, YOU/ROCK invite, thumb verdict |
| `game_host.py` | multi-player loop: find player → approach → play → next |
| `approach.py` | camera x → yaw bearing, turn + short step |
| `leaderboard.py` | SQLite players/matches |
| `igra_api.py` | FastAPI service (`/status`, `/leaderboard`, `/game/start`, …) |
| `speech_io.py` | speak/listen + yes/no and name parsing (Serbian + English) |
| `announce.py`, `camera_source.py`, `head_screen.py` | output/input adapters |

Vision labels in `gestures.json`: `rock` / `paper` / `scissors` / `deaf` / `y` / `o` / `u`
(also accepts kamen, papir, makaze, gluvonem). There is no yes/no sign: the robot
signs the invite itself, and the player assents with `rock` or `y`.

## Plugging in a robot

The game logic talks to the outside world only through small hooks. Defaults
work on a laptop; replace them to run on real hardware.

| Hook | Default | Replace with |
|---|---|---|
| `announce.set_speaker(fn(text) -> bool)` | macOS `say` / espeak | robot speaker, cloud TTS |
| `speech_io.set_listener(fn(timeout) -> str \| None)` | typed terminal line | any STT |
| `camera_source.set_camera_opener(fn(device) -> capture)` | OpenCV index/path | robot camera stream |
| `head_screen.set_display(show(text, s), warm())` | log only | face/head display |
| `approach.set_mover(turn(rad), forward(m), after())` | dry-run log | locomotion API |
| `game_host.set_person_source(fn() -> dict)` | hand-in-frame presence | person/face detector |

`robot_arm.py` drives the arm and hand directly over HTTP-RPC (`IGRA_MC_URL`).

## Env knobs

| Var | Default | Meaning |
|---|---|---|
| `IGRA_CAMERA` | `0` | OpenCV index/path, or the id passed to a custom opener |
| `IGRA_GESTURE_DB` | `gestures.json` | gesture samples |
| `IGRA_SHOW_WINDOW` | `0` | OpenCV preview |
| `IGRA_ARM_DRY_RUN` | `0` | print arm/hand commands instead of sending |
| `IGRA_NAV_EXECUTE` | `0` | allow real approach motion (needs a mover) |
| `IGRA_MAX_DISTANCE` | `3.2` | match threshold |
| `IGRA_STABLE_FRAMES` | `3` | frames before a detect-mode fire |
| `IGRA_ANNOUNCE_TEXT` | `RADI` | phrase in detect mode |
| `IGRA_HEAD_SCREEN` | `1` | `0` disables display calls |
| `IGRA_MODE` | `game` | `game` or `detect` |
