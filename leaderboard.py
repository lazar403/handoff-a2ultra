"""SQLite leaderboard for IGRA (RPS) players and matches."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "igra_leaderboard.db"


class LeaderboardStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    draws INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_name TEXT NOT NULL,
                    robot_id TEXT NOT NULL,
                    human_choice TEXT,
                    robot_choice TEXT,
                    winner TEXT NOT NULL,
                    comm_mode TEXT,
                    posture_mode TEXT,
                    created_at REAL NOT NULL
                );
                """
            )

    def list_players(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT name, wins, losses, draws, created_at, updated_at
                FROM players
                ORDER BY wins DESC, name ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_matches(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, player_name, robot_id, human_choice, robot_choice,
                       winner, comm_mode, posture_mode, created_at
                FROM matches
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def snapshot(self, match_limit: int = 50) -> dict[str, Any]:
        players = self.list_players()
        return {
            "players": players,
            "matches": self.list_matches(limit=match_limit),
            "player_count": len(players),
            "db_path": str(self.db_path),
        }

    def ensure_player(self, name: str) -> None:
        clean = (name or "").strip() or "player"
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO players (name, wins, losses, draws, created_at, updated_at)
                VALUES (?, 0, 0, 0, ?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (clean, now, now),
            )

    def record_match(
        self,
        *,
        player_name: str,
        robot_id: str,
        winner: str,
        human_choice: str | None = None,
        robot_choice: str | None = None,
        comm_mode: str | None = None,
        posture_mode: str | None = None,
    ) -> dict[str, Any]:
        player = (player_name or "").strip() or "player"
        robot = (robot_id or "").strip() or "robot"
        result = (winner or "").strip().lower() or "draw"
        now = time.time()
        self.ensure_player(player)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO matches (
                    player_name, robot_id, human_choice, robot_choice,
                    winner, comm_mode, posture_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    player,
                    robot,
                    human_choice,
                    robot_choice,
                    result,
                    comm_mode,
                    posture_mode,
                    now,
                ),
            )
            if result == "human":
                conn.execute(
                    "UPDATE players SET wins = wins + 1, updated_at = ? WHERE name = ?",
                    (now, player),
                )
            elif result == "robot":
                conn.execute(
                    "UPDATE players SET losses = losses + 1, updated_at = ? WHERE name = ?",
                    (now, player),
                )
            else:
                conn.execute(
                    "UPDATE players SET draws = draws + 1, updated_at = ? WHERE name = ?",
                    (now, player),
                )

        return self.snapshot()
