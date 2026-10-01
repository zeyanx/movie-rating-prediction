"""第1阶段独立验收脚本：验证环境、依赖与 MovieLens 100K 数据。"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_COLUMNS = {
    "ratings.csv": ["user_id", "movie_id", "rating", "timestamp"],
    "users.csv": ["user_id", "age", "gender", "occupation", "zip_code"],
    "movies.csv": [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "genres",
    ],
}


def require(condition: bool, message: str) -> None:
    """条件不满足时立即中止，确保验证失败不会被伪装成成功。"""
    if not condition:
        raise AssertionError(message)


def verify_python_and_packages() -> None:
    """检查 Python 版本及所有阶段依赖。"""
    print("=== Python 与依赖检查 ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"解释器: {sys.executable}")
    require(sys.version_info[:2] == (3, 10), "本项目要求 Python 3.10")

    package_names = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("torch", "torch"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
        ("joblib", "joblib"),
    ]
    for display_name, import_name in package_names:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "未知")
        print(f"{display_name}: {version}")
    print(f"sqlite3: {sqlite3.sqlite_version}（Python 标准库）")


def load_and_verify_csv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载三个CSV并检查结构、规模、范围和引用完整性。"""
    print("\n=== 数据文件检查 ===")
    for filename in EXPECTED_COLUMNS:
        require((RAW_DIR / filename).is_file(), f"缺少文件：{RAW_DIR / filename}")
    require((RAW_DIR / "dataset_info.json").is_file(), "缺少 dataset_info.json")

    ratings = pd.read_csv(RAW_DIR / "ratings.csv", encoding="utf-8-sig")
    users = pd.read_csv(RAW_DIR / "users.csv", encoding="utf-8-sig", dtype={"zip_code": "string"})
    movies = pd.read_csv(RAW_DIR / "movies.csv", encoding="utf-8-sig")

    frames = {"ratings.csv": ratings, "users.csv": users, "movies.csv": movies}
    for filename, expected in EXPECTED_COLUMNS.items():
        require(list(frames[filename].columns) == expected, f"{filename} 字段不符合要求")

    require(len(ratings) == 100_000, f"评分数应为100000，实际为{len(ratings)}")
    require(len(users) == 943, f"用户数应为943，实际为{len(users)}")
    require(len(movies) == 1_682, f"电影数应为1682，实际为{len(movies)}")
    require(ratings["rating"].min() == 1, "最低评分应为1")
    require(ratings["rating"].max() == 5, "最高评分应为5")
    require(ratings["rating"].between(1, 5).all(), "存在1至5之外的评分")

    require(users["user_id"].is_unique, "users.csv 存在重复 user_id")
    require(movies["movie_id"].is_unique, "movies.csv 存在重复 movie_id")
    require(
        not ratings.duplicated(["user_id", "movie_id"]).any(),
        "ratings.csv 存在重复的 user_id/movie_id 评分主键",
    )
    require(
        set(ratings["user_id"]).issubset(set(users["user_id"])),
        "评分表包含用户表中不存在的 user_id",
    )
    require(
        set(ratings["movie_id"]).issubset(set(movies["movie_id"])),
        "评分表包含电影表中不存在的 movie_id",
    )
    require(movies["genres"].notna().all(), "电影类型中存在空值")

    info = json.loads((RAW_DIR / "dataset_info.json").read_text(encoding="utf-8"))
    require(info["rating_count"] == len(ratings), "dataset_info 的评分数不一致")
    require(info["user_count"] == len(users), "dataset_info 的用户数不一致")
    require(info["movie_count"] == len(movies), "dataset_info 的电影数不一致")
    require(info["rating_min"] == 1 and info["rating_max"] == 5, "dataset_info 的评分范围不一致")

    print(f"评分数：{len(ratings)}")
    print(f"用户数：{len(users)}")
    print(f"电影数：{len(movies)}")
    print(f"评分范围：{ratings['rating'].min()} - {ratings['rating'].max()}")
    return ratings, users, movies


def show_samples(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    """输出少量样例，便于人工复核编码和字段。"""
    print("\n=== 数据样例 ===")
    print("ratings.csv:")
    print(ratings.head(3).to_string(index=False))
    print("\nusers.csv:")
    print(users.head(3).to_string(index=False))
    print("\nmovies.csv:")
    print(movies.head(3).to_string(index=False))


def main() -> int:
    verify_python_and_packages()
    ratings, users, movies = load_and_verify_csv()
    show_samples(ratings, users, movies)
    print("\n第1阶段验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
