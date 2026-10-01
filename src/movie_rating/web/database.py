"""SQLite用户观影和反馈持久化；每次操作使用短连接。"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import PROJECT_ROOT, load_app_config, resolve_project_path


VALID_STATUSES = {"want_to_watch", "watched"}
VALID_FEEDBACK = {None, "like", "neutral", "dislike", "not_interested"}


def default_database_path() -> Path:
    return resolve_project_path(load_app_config()["database_path"])


@contextmanager
def _connection(path: Path | str) -> Iterator[sqlite3.Connection]:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=5.0)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.row_factory = sqlite3.Row
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database(path: Path | str | None = None) -> Path:
    """幂等创建反馈表和常用索引。"""
    database_path = Path(path) if path is not None else default_database_path()
    with _connection(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_key TEXT NOT NULL,
                user_id INTEGER,
                movie_id INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('want_to_watch', 'watched')),
                personal_rating REAL CHECK(personal_rating IS NULL OR (personal_rating >= 1 AND personal_rating <= 5)),
                feedback TEXT CHECK(feedback IS NULL OR feedback IN ('like', 'neutral', 'dislike', 'not_interested')),
                note TEXT NOT NULL DEFAULT '' CHECK(length(note) <= 500),
                predicted_rating REAL CHECK(predicted_rating IS NULL OR (predicted_rating >= 1 AND predicted_rating <= 5)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(profile_key, movie_id)
            );
            CREATE INDEX IF NOT EXISTS idx_interactions_profile ON user_interactions(profile_key);
            CREATE INDEX IF NOT EXISTS idx_interactions_movie ON user_interactions(movie_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_status ON user_interactions(status);
            CREATE INDEX IF NOT EXISTS idx_interactions_updated ON user_interactions(updated_at);
            """
        )
    return database_path.resolve()


def _validate_interaction(
    profile_key: str,
    user_id: int | None,
    movie_id: int,
    status: str,
    personal_rating: float | None,
    feedback: str | None,
    note: str,
    predicted_rating: float | None,
) -> None:
    if not str(profile_key).strip() or len(str(profile_key)) > 100:
        raise ValueError("profile_key不能为空且不能超过100字符")
    if user_id is not None and not 1 <= int(user_id) <= 943:
        raise ValueError("user_id必须位于1至943")
    if not 1 <= int(movie_id) <= 1682:
        raise ValueError("movie_id必须位于1至1682")
    if status not in VALID_STATUSES:
        raise ValueError(f"非法观影状态：{status}")
    if feedback not in VALID_FEEDBACK:
        raise ValueError(f"非法反馈：{feedback}")
    if personal_rating is not None and not 1 <= float(personal_rating) <= 5:
        raise ValueError("个人评分必须位于1至5")
    if predicted_rating is not None and not 1 <= float(predicted_rating) <= 5:
        raise ValueError("预测评分必须位于1至5")
    if len(str(note)) > 500:
        raise ValueError("备注不能超过500字符")


def upsert_interaction(
    profile_key: str,
    user_id: int | None,
    movie_id: int,
    status: str,
    personal_rating: float | None = None,
    feedback: str | None = None,
    note: str = "",
    predicted_rating: float | None = None,
    path: Path | str | None = None,
) -> None:
    """新增或更新一条用户—电影记录，全部值通过参数化SQL传入。"""
    _validate_interaction(
        profile_key, user_id, movie_id, status, personal_rating, feedback, note, predicted_rating
    )
    database_path = initialize_database(path)
    now = datetime.now(timezone.utc).isoformat()
    with _connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO user_interactions
              (profile_key, user_id, movie_id, status, personal_rating, feedback, note,
               predicted_rating, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_key, movie_id) DO UPDATE SET
              user_id=excluded.user_id,
              status=excluded.status,
              personal_rating=excluded.personal_rating,
              feedback=excluded.feedback,
              note=excluded.note,
              predicted_rating=excluded.predicted_rating,
              updated_at=excluded.updated_at
            """,
            (
                str(profile_key).strip(), user_id, int(movie_id), status,
                personal_rating, feedback, str(note), predicted_rating, now, now,
            ),
        )


def get_interactions(
    profile_key: str, path: Path | str | None = None
) -> pd.DataFrame:
    database_path = initialize_database(path)
    with _connection(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM user_interactions WHERE profile_key = ? ORDER BY updated_at DESC, id DESC",
            (str(profile_key),),
        ).fetchall()
    columns = [
        "id", "profile_key", "user_id", "movie_id", "status", "personal_rating",
        "feedback", "note", "predicted_rating", "created_at", "updated_at",
    ]
    return pd.DataFrame([dict(row) for row in rows], columns=columns)


def get_interaction(
    profile_key: str, movie_id: int, path: Path | str | None = None
) -> dict | None:
    database_path = initialize_database(path)
    with _connection(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM user_interactions WHERE profile_key = ? AND movie_id = ?",
            (str(profile_key), int(movie_id)),
        ).fetchone()
    return dict(row) if row is not None else None


def delete_interaction(
    profile_key: str, movie_id: int, path: Path | str | None = None
) -> bool:
    database_path = initialize_database(path)
    with _connection(database_path) as connection:
        cursor = connection.execute(
            "DELETE FROM user_interactions WHERE profile_key = ? AND movie_id = ?",
            (str(profile_key), int(movie_id)),
        )
    return cursor.rowcount > 0


def get_feedback_fingerprint(profile_key: str, path: Path | str | None = None) -> str:
    """为需要缓存的用户结果提供反馈版本指纹。"""
    frame = get_interactions(profile_key, path)
    payload = frame.to_json(orient="records", date_format="iso")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
