"""MovieLens 100K 数据加载、质量检查和固定划分工具。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
USERS_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]
MOVIES_COLUMNS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "genres",
]
SPLIT_COLUMNS = ["interaction_id", *RATINGS_COLUMNS]


@dataclass(frozen=True)
class RawData:
    """保存三张规范化原始表，便于在各阶段共享统一接口。"""

    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame


def _strip_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """只清理字符串首尾空格，不改变缺失值和业务含义。"""
    cleaned = frame.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return cleaned


def load_raw_data(raw_dir: Path) -> RawData:
    """按固定字段类型读取第1阶段生成的三张CSV。"""
    raw_dir = Path(raw_dir)
    required_paths = {
        "ratings": raw_dir / "ratings.csv",
        "movies": raw_dir / "movies.csv",
        "users": raw_dir / "users.csv",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少第1阶段原始CSV：{missing}")

    ratings = pd.read_csv(
        required_paths["ratings"],
        encoding="utf-8-sig",
        dtype={
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "int64",
            "timestamp": "int64",
        },
    )
    users = pd.read_csv(
        required_paths["users"],
        encoding="utf-8-sig",
        dtype={
            "user_id": "int64",
            "age": "int64",
            "gender": "string",
            "occupation": "string",
            "zip_code": "string",
        },
    )
    movies = pd.read_csv(
        required_paths["movies"],
        encoding="utf-8-sig",
        dtype={
            "movie_id": "int64",
            "title": "string",
            "release_date": "string",
            "video_release_date": "string",
            "imdb_url": "string",
            "genres": "string",
        },
    )
    return RawData(
        ratings=_strip_string_columns(ratings),
        movies=_strip_string_columns(movies),
        users=_strip_string_columns(users),
    )


def validate_raw_data(data: RawData) -> None:
    """验证字段、规模、主键、评分范围和表间引用完整性。"""
    if list(data.ratings.columns) != RATINGS_COLUMNS:
        raise ValueError(f"ratings.csv字段异常：{list(data.ratings.columns)}")
    if list(data.users.columns) != USERS_COLUMNS:
        raise ValueError(f"users.csv字段异常：{list(data.users.columns)}")
    if list(data.movies.columns) != MOVIES_COLUMNS:
        raise ValueError(f"movies.csv字段异常：{list(data.movies.columns)}")

    expected_counts = {"ratings": 100_000, "users": 943, "movies": 1_682}
    actual_counts = {
        "ratings": len(data.ratings),
        "users": len(data.users),
        "movies": len(data.movies),
    }
    if actual_counts != expected_counts:
        raise ValueError(f"MovieLens 100K规模异常：{actual_counts}")
    if not data.ratings["rating"].between(1, 5).all():
        raise ValueError("ratings.csv存在1至5之外的评分")
    if (data.ratings[["user_id", "movie_id", "timestamp"]] <= 0).any().any():
        raise ValueError("ratings.csv存在非正的ID或时间戳")
    if (data.users[["user_id", "age"]] <= 0).any().any():
        raise ValueError("users.csv存在非正的用户ID或年龄")
    if not set(data.users["gender"].dropna().unique()).issubset({"M", "F"}):
        raise ValueError("users.csv存在M/F之外的性别编码")
    if (data.movies["movie_id"] <= 0).any():
        raise ValueError("movies.csv存在非正的movie_id")
    if data.movies["genres"].isna().any():
        raise ValueError("movies.csv存在缺失的电影类型")
    if not data.users["user_id"].is_unique:
        raise ValueError("users.csv存在重复user_id")
    if not data.movies["movie_id"].is_unique:
        raise ValueError("movies.csv存在重复movie_id")
    if data.ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("ratings.csv存在重复的user_id/movie_id组合")

    unknown_users = set(data.ratings["user_id"]) - set(data.users["user_id"])
    unknown_movies = set(data.ratings["movie_id"]) - set(data.movies["movie_id"])
    if unknown_users:
        raise ValueError(f"评分表引用了不存在的用户：{sorted(unknown_users)[:10]}")
    if unknown_movies:
        raise ValueError(f"评分表引用了不存在的电影：{sorted(unknown_movies)[:10]}")

    # validate='many_to_one'会在元数据主键不唯一时直接报错。
    merged = data.ratings.merge(data.users, on="user_id", how="left", validate="many_to_one")
    merged = merged.merge(data.movies, on="movie_id", how="left", validate="many_to_one")
    if len(merged) != len(data.ratings):
        raise ValueError("评分表与元数据合并后记录数发生变化")


def _frame_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """生成单张表可JSON序列化的质量摘要。"""
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def rating_distribution(ratings: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """返回1至5分的数量和占比，键使用字符串以便稳定写入JSON。"""
    counts = ratings["rating"].value_counts().sort_index()
    total = len(ratings)
    return {
        str(score): {
            "count": int(counts.get(score, 0)),
            "proportion": float(counts.get(score, 0) / total),
        }
        for score in range(1, 6)
    }


def build_data_quality_report(data: RawData) -> dict[str, Any]:
    """统计缺失、重复、时间解析、评分分布和表间连接质量。"""
    rating_datetimes = pd.to_datetime(
        data.ratings["timestamp"], unit="s", utc=True, errors="coerce"
    )
    release_dates = pd.to_datetime(
        data.movies["release_date"], format="%d-%b-%Y", errors="coerce"
    )
    release_missing = data.movies["release_date"].isna()
    release_unparseable = release_dates.isna() & ~release_missing

    return {
        "dataset": "MovieLens 100K",
        "tables": {
            "ratings": _frame_quality(data.ratings),
            "users": _frame_quality(data.users),
            "movies": _frame_quality(data.movies),
        },
        "rating_range": {
            "min": int(data.ratings["rating"].min()),
            "max": int(data.ratings["rating"].max()),
        },
        "user_age_range": {
            "min": int(data.users["age"].min()),
            "max": int(data.users["age"].max()),
        },
        "gender_values": sorted(data.users["gender"].dropna().unique().tolist()),
        "rating_distribution": rating_distribution(data.ratings),
        "unique_users_in_ratings": int(data.ratings["user_id"].nunique()),
        "unique_movies_in_ratings": int(data.ratings["movie_id"].nunique()),
        "duplicate_user_movie_pairs": int(
            data.ratings.duplicated(["user_id", "movie_id"]).sum()
        ),
        "invalid_rating_timestamps": int(rating_datetimes.isna().sum()),
        "rating_datetime_utc_min": rating_datetimes.min().isoformat(),
        "rating_datetime_utc_max": rating_datetimes.max().isoformat(),
        "missing_release_dates": int(release_missing.sum()),
        "unparseable_non_missing_release_dates": int(release_unparseable.sum()),
        "join_validation": "many_to_one通过，100000条评分均匹配用户和电影元数据",
        "preprocessing_policy": {
            "string_cleanup": "仅清理字符串首尾空格",
            "missing_values": "不凭空填补；video_release_date本阶段不作为模型输入",
            "timestamp": "仅额外解析为UTC用于质量检查，原始整数时间戳保持不变",
            "feature_engineering": "本阶段未进行ID编码、独热编码、统计特征或标准化",
        },
    }


def split_ratings(
    ratings: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加入稳定交互ID，并按评分值分层划分训练集和测试集。"""
    if list(ratings.columns) != RATINGS_COLUMNS:
        raise ValueError("划分前ratings字段与规范不一致")
    if not 0 < test_size < 1:
        raise ValueError("test_size必须位于0和1之间")

    indexed = ratings.copy()
    indexed.insert(0, "interaction_id", np.arange(len(indexed), dtype=np.int64))
    train, test = train_test_split(
        indexed,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=indexed["rating"],
    )
    train = train.sort_values("interaction_id").reset_index(drop=True)
    test = test.sort_values("interaction_id").reset_index(drop=True)
    return train[SPLIT_COLUMNS], test[SPLIT_COLUMNS]


def save_split(train: pd.DataFrame, test: pd.DataFrame, processed_dir: Path) -> None:
    """将固定划分写入规范路径，供全部后续模型复用。"""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train_ratings.csv", index=False, encoding="utf-8-sig")
    test.to_csv(processed_dir / "test_ratings.csv", index=False, encoding="utf-8-sig")


def calculate_file_sha256(path: Path) -> str:
    """分块计算文件SHA-256，用于验证原始数据和划分未被修改。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cold_start_statistics(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, int]:
    """统计测试集中训练阶段未出现的用户和电影，不移动任何记录。"""
    unseen_users = set(test["user_id"]) - set(train["user_id"])
    unseen_movies = set(test["movie_id"]) - set(train["movie_id"])
    cold_mask = test["user_id"].isin(unseen_users) | test["movie_id"].isin(unseen_movies)
    return {
        "unseen_test_users": len(unseen_users),
        "unseen_test_movies": len(unseen_movies),
        "cold_start_test_rows": int(cold_mask.sum()),
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    """以稳定、可读的UTF-8格式写入JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
