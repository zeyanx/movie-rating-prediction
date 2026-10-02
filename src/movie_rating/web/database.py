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
    """幂等创建反馈表、中文电影资料表和常用索引。"""
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

            CREATE TABLE IF NOT EXISTS movie_localizations (
                movie_id INTEGER PRIMARY KEY,
                title_zh TEXT NOT NULL,
                aliases_zh TEXT NOT NULL DEFAULT '',
                source_name TEXT NOT NULL,
                source_url TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS chinese_movies (
                catalog_id TEXT PRIMARY KEY,
                title_zh TEXT NOT NULL,
                title_en TEXT NOT NULL DEFAULT '',
                release_year INTEGER NOT NULL,
                origin TEXT NOT NULL,
                genres TEXT NOT NULL,
                overview_zh TEXT NOT NULL DEFAULT '',
                source_name TEXT NOT NULL,
                source_url TEXT NOT NULL DEFAULT '',
                movielens_movie_id INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_chinese_movies_year ON chinese_movies(release_year);
            CREATE INDEX IF NOT EXISTS idx_chinese_movies_origin ON chinese_movies(origin);

            CREATE TABLE IF NOT EXISTS catalog_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_key TEXT NOT NULL,
                catalog_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('want_to_watch', 'watched')),
                personal_rating REAL CHECK(personal_rating IS NULL OR (personal_rating >= 1 AND personal_rating <= 5)),
                note TEXT NOT NULL DEFAULT '' CHECK(length(note) <= 500),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(profile_key, catalog_id),
                FOREIGN KEY(catalog_id) REFERENCES chinese_movies(catalog_id)
            );
            CREATE INDEX IF NOT EXISTS idx_catalog_interactions_profile
              ON catalog_interactions(profile_key);
            """
        )
        _seed_catalog_tables(connection)
    return database_path.resolve()


def _text_or_empty(value: object) -> str:
    """将CSV中的空值稳定转换为空字符串。"""
    return "" if pd.isna(value) else str(value).strip()


def _seed_catalog_tables(connection: sqlite3.Connection) -> None:
    """把版本控制中的中文资料CSV幂等同步到SQLite。"""
    catalog_dir = PROJECT_ROOT / "data" / "catalog"
    localization_path = catalog_dir / "movie_localizations.csv"
    chinese_path = catalog_dir / "chinese_movies.csv"
    if not localization_path.is_file() or not chinese_path.is_file():
        raise FileNotFoundError("缺少data/catalog中文电影资料文件")

    localizations = pd.read_csv(localization_path, encoding="utf-8-sig")
    required_localization = {"movie_id", "title_zh", "aliases_zh", "source_name", "source_url"}
    if set(localizations.columns) != required_localization or localizations["movie_id"].duplicated().any():
        raise ValueError("movie_localizations.csv字段或主键异常")
    connection.executemany(
        """
        INSERT INTO movie_localizations(movie_id, title_zh, aliases_zh, source_name, source_url)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(movie_id) DO UPDATE SET
          title_zh=excluded.title_zh,
          aliases_zh=excluded.aliases_zh,
          source_name=excluded.source_name,
          source_url=excluded.source_url
        """,
        [
            (
                int(row.movie_id), _text_or_empty(row.title_zh),
                _text_or_empty(row.aliases_zh), _text_or_empty(row.source_name),
                _text_or_empty(row.source_url),
            )
            for row in localizations.itertuples(index=False)
        ],
    )

    chinese = pd.read_csv(chinese_path, encoding="utf-8-sig")
    expected = [
        "catalog_id", "title_zh", "title_en", "release_year", "origin", "genres",
        "overview_zh", "source_name", "source_url", "movielens_movie_id",
    ]
    if list(chinese.columns) != expected or chinese["catalog_id"].duplicated().any():
        raise ValueError("chinese_movies.csv字段或主键异常")
    connection.executemany(
        """
        INSERT INTO chinese_movies(
          catalog_id, title_zh, title_en, release_year, origin, genres,
          overview_zh, source_name, source_url, movielens_movie_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(catalog_id) DO UPDATE SET
          title_zh=excluded.title_zh,
          title_en=excluded.title_en,
          release_year=excluded.release_year,
          origin=excluded.origin,
          genres=excluded.genres,
          overview_zh=excluded.overview_zh,
          source_name=excluded.source_name,
          source_url=excluded.source_url,
          movielens_movie_id=excluded.movielens_movie_id
        """,
        [
            (
                _text_or_empty(row.catalog_id), _text_or_empty(row.title_zh),
                _text_or_empty(row.title_en), int(row.release_year),
                _text_or_empty(row.origin), _text_or_empty(row.genres),
                _text_or_empty(row.overview_zh), _text_or_empty(row.source_name),
                _text_or_empty(row.source_url),
                None if pd.isna(row.movielens_movie_id) else int(row.movielens_movie_id),
            )
            for row in chinese.itertuples(index=False)
        ],
    )


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


def get_movie_localizations(path: Path | str | None = None) -> pd.DataFrame:
    """读取MovieLens影片的中文片名与检索别名。"""
    database_path = initialize_database(path)
    with _connection(database_path) as connection:
        rows = connection.execute(
            "SELECT movie_id, title_zh, aliases_zh, source_name, source_url "
            "FROM movie_localizations ORDER BY movie_id"
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def get_chinese_movies(path: Path | str | None = None) -> pd.DataFrame:
    """读取独立中国电影扩展库；其中未映射MovieLens的影片不参与模型推理。"""
    database_path = initialize_database(path)
    with _connection(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM chinese_movies ORDER BY release_year DESC, title_zh"
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def upsert_catalog_interaction(
    profile_key: str,
    catalog_id: str,
    status: str,
    personal_rating: float | None = None,
    note: str = "",
    path: Path | str | None = None,
) -> None:
    """保存扩展中国电影的想看、已看和个人评分，不伪造模型预测。"""
    if not str(profile_key).strip() or len(str(profile_key)) > 100:
        raise ValueError("profile_key不能为空且不能超过100字符")
    if status not in VALID_STATUSES:
        raise ValueError(f"非法观影状态：{status}")
    if personal_rating is not None and not 1 <= float(personal_rating) <= 5:
        raise ValueError("个人评分必须位于1至5")
    if len(str(note)) > 500:
        raise ValueError("备注不能超过500字符")
    database_path = initialize_database(path)
    now = datetime.now(timezone.utc).isoformat()
    with _connection(database_path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM chinese_movies WHERE catalog_id = ?", (str(catalog_id),)
        ).fetchone()
        if exists is None:
            raise ValueError(f"扩展电影ID不存在：{catalog_id}")
        connection.execute(
            """
            INSERT INTO catalog_interactions(
              profile_key, catalog_id, status, personal_rating, note, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_key, catalog_id) DO UPDATE SET
              status=excluded.status,
              personal_rating=excluded.personal_rating,
              note=excluded.note,
              updated_at=excluded.updated_at
            """,
            (
                str(profile_key).strip(), str(catalog_id), status,
                personal_rating, str(note), now, now,
            ),
        )


def get_catalog_interactions(
    profile_key: str, path: Path | str | None = None
) -> pd.DataFrame:
    """读取扩展中国电影的个人记录并附带影片资料。"""
    database_path = initialize_database(path)
    with _connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT i.*, m.title_zh, m.title_en, m.release_year, m.origin, m.genres
            FROM catalog_interactions AS i
            JOIN chinese_movies AS m ON m.catalog_id = i.catalog_id
            WHERE i.profile_key = ?
            ORDER BY i.updated_at DESC, i.id DESC
            """,
            (str(profile_key),),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def delete_catalog_interaction(
    profile_key: str, catalog_id: str, path: Path | str | None = None
) -> bool:
    """删除一条扩展中国电影个人记录。"""
    database_path = initialize_database(path)
    with _connection(database_path) as connection:
        cursor = connection.execute(
            "DELETE FROM catalog_interactions WHERE profile_key = ? AND catalog_id = ?",
            (str(profile_key), str(catalog_id)),
        )
    return cursor.rowcount > 0
