"""网页静态数据、搜索、统计与模型输入构造。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from movie_rating.data import RawData, load_raw_data
from movie_rating.neural_data import enrich_interactions

from .config import PROJECT_ROOT
from .database import get_chinese_movies, get_movie_localizations


@st.cache_data(show_spinner=False)
def load_raw_tables(root: str = str(PROJECT_ROOT)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """缓存三张规范化原始表。返回顺序为评分、电影、用户。"""
    data: RawData = load_raw_data(Path(root) / "data" / "raw")
    return data.ratings, data.movies, data.users


@st.cache_data(show_spinner=False)
def load_train_ratings(root: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """推荐和用户画像只读取固定训练集，避免测试集泄漏。"""
    path = Path(root) / "data" / "processed" / "train_ratings.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少固定训练集：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"interaction_id", "user_id", "movie_id", "rating", "timestamp"}
    if not required.issubset(frame.columns) or len(frame) != 80_000:
        raise ValueError("固定训练集字段或行数异常")
    return frame


@st.cache_data(show_spinner=False)
def load_stage5_reports(root: str = str(PROJECT_ROOT)) -> dict[str, Any]:
    """集中加载模型实验室所需的小型报告，不加载2万行预测表。"""
    base = Path(root) / "reports" / "stage5"
    csv_names = {
        "metrics": "metrics_comparison.csv",
        "efficiency": "efficiency_benchmark.csv",
        "bootstrap": "paired_bootstrap_summary.csv",
        "ensemble_importance": "ensemble_group_importance.csv",
        "mlp_importance": "mlp_permutation_importance.csv",
        "segments": "segment_metrics.csv",
        "residuals": "residual_summary.csv",
        "rating_diagnostics": "actual_rating_diagnostics.csv",
    }
    missing = [name for name in csv_names.values() if not (base / name).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少第5阶段报告：{missing}")
    reports: dict[str, Any] = {
        key: pd.read_csv(base / name, encoding="utf-8-sig")
        for key, name in csv_names.items()
    }
    recommendation_path = base / "model_recommendation.json"
    reports["recommendation"] = json.loads(
        recommendation_path.read_text(encoding="utf-8-sig")
    )
    return reports


@st.cache_data(show_spinner=False)
def movie_statistics(root: str = str(PROJECT_ROOT), prior_count: int = 20) -> pd.DataFrame:
    """只用训练集生成电影热度、均值和贝叶斯平滑评分。"""
    train = load_train_ratings(root)
    _, movies, _ = load_raw_tables(root)
    grouped = train.groupby("movie_id")["rating"].agg(rating_count="count", rating_mean="mean")
    global_mean = float(train["rating"].mean())
    grouped["bayesian_popularity"] = (
        grouped["rating_count"] * grouped["rating_mean"] + prior_count * global_mean
    ) / (grouped["rating_count"] + prior_count)
    result = movies.merge(grouped, on="movie_id", how="left")
    result["rating_count"] = result["rating_count"].fillna(0).astype(int)
    result["rating_mean"] = result["rating_mean"].fillna(global_mean)
    result["bayesian_popularity"] = result["bayesian_popularity"].fillna(global_mean)
    result["release_year"] = pd.to_datetime(
        result["release_date"], format="%d-%b-%Y", errors="coerce"
    ).dt.year.astype("Int64")
    localizations = get_movie_localizations()
    result = result.merge(localizations, on="movie_id", how="left", validate="one_to_one")
    for column in ["title_zh", "aliases_zh", "source_name", "source_url"]:
        result[column] = result[column].fillna("").astype(str)
    result["display_title"] = result.apply(
        lambda row: f"{row['title_zh']} / {row['title']}" if row["title_zh"] else str(row["title"]),
        axis=1,
    )
    return result


def search_movies(
    query: str = "",
    genre: str | None = None,
    release_year: int | None = None,
    limit: int = 100,
    root: str = str(PROJECT_ROOT),
) -> pd.DataFrame:
    """按标题字面子串检索，不把用户输入解释为正则表达式。"""
    if not 1 <= int(limit) <= 500:
        raise ValueError("搜索结果上限必须位于1至500")
    movies = movie_statistics(root).copy()
    clean_query = str(query).strip()
    if clean_query:
        searchable = (
            movies["title"].fillna("") + " "
            + movies["title_zh"].fillna("") + " "
            + movies["aliases_zh"].fillna("")
        )
        movies = movies[searchable.str.contains(clean_query, case=False, regex=False, na=False)]
    if genre and genre != "全部":
        movies = movies[
            movies["genres"].fillna("").str.split("|").map(lambda values: genre in values)
        ]
    if release_year is not None:
        movies = movies[movies["release_year"] == int(release_year)]
    return movies.sort_values(["rating_count", "title"], ascending=[False, True]).head(limit).reset_index(drop=True)


def get_movie(movie_id: int, root: str = str(PROJECT_ROOT)) -> pd.Series:
    movies = movie_statistics(root)
    matches = movies[movies["movie_id"] == int(movie_id)]
    if matches.empty:
        raise ValueError(f"电影ID不存在：{movie_id}")
    return matches.iloc[0]


def get_user_history(user_id: int, root: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    train = load_train_ratings(root)
    movies = movie_statistics(root)
    history = train[train["user_id"] == int(user_id)].merge(
        movies[["movie_id", "title", "title_zh", "display_title", "genres"]],
        on="movie_id", how="left", validate="many_to_one"
    )
    return history.sort_values("timestamp", ascending=False).reset_index(drop=True)


def search_chinese_catalog(
    query: str = "",
    origin: str | None = None,
    genre: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """检索独立中国电影扩展库，全部筛选均为字面匹配。"""
    if not 1 <= int(limit) <= 500:
        raise ValueError("搜索结果上限必须位于1至500")
    movies = get_chinese_movies().copy()
    clean_query = str(query).strip()
    if clean_query:
        searchable = (
            movies["title_zh"].fillna("") + " "
            + movies["title_en"].fillna("") + " "
            + movies["overview_zh"].fillna("")
        )
        movies = movies[searchable.str.contains(clean_query, case=False, regex=False, na=False)]
    if origin and origin != "全部":
        movies = movies[
            movies["origin"].fillna("").str.split("|").map(lambda values: origin in values)
        ]
    if genre and genre != "全部":
        movies = movies[
            movies["genres"].fillna("").str.split("|").map(lambda values: genre in values)
        ]
    if start_year is not None:
        movies = movies[movies["release_year"] >= int(start_year)]
    if end_year is not None:
        movies = movies[movies["release_year"] <= int(end_year)]
    return movies.sort_values(
        ["release_year", "title_zh"], ascending=[False, True]
    ).head(limit).reset_index(drop=True)


def chinese_catalog_facets() -> tuple[list[str], list[str]]:
    """返回扩展库的地区与类型筛选项。"""
    movies = get_chinese_movies()
    origins = sorted({value for text in movies["origin"] for value in str(text).split("|") if value})
    genres = sorted({value for text in movies["genres"] for value in str(text).split("|") if value})
    return origins, genres


@st.cache_data(show_spinner=False)
def load_chinese_rated_catalog(root: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """加载具有真实评分训练依据的中国电影目录。"""
    path = Path(root) / "data" / "catalog" / "chinese_rated_movies.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少中国电影训练目录：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "catalog_id", "cn_movie_index", "title_zh", "title_en", "release_year",
        "origins", "languages", "wikidata_id", "imdb_id", "source_url",
    }
    if not required.issubset(frame.columns) or len(frame) < 100:
        raise ValueError("中国电影训练目录字段或规模异常")
    if frame["catalog_id"].duplicated().any() or frame["cn_movie_index"].duplicated().any():
        raise ValueError("中国电影训练目录主键重复")
    frame["release_year"] = pd.to_numeric(frame["release_year"], errors="coerce").astype("Int64")
    return frame.sort_values(["release_year", "title_zh"], ascending=[False, True]).reset_index(drop=True)


def search_chinese_rated_catalog(
    query: str = "",
    origin: str | None = None,
    language: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    limit: int = 100,
    root: str = str(PROJECT_ROOT),
) -> pd.DataFrame:
    """检索300部具有MovieLens 25M训练评分的中国电影。"""
    if not 1 <= int(limit) <= 500:
        raise ValueError("搜索结果上限必须位于1至500")
    movies = load_chinese_rated_catalog(root).copy()
    clean_query = str(query).strip()
    if clean_query:
        searchable = movies["title_zh"].fillna("") + " " + movies["title_en"].fillna("")
        movies = movies[searchable.str.contains(clean_query, case=False, regex=False, na=False)]
    if origin and origin != "全部":
        movies = movies[movies["origins"].fillna("").str.split("|").map(lambda x: origin in x)]
    if language and language != "全部":
        movies = movies[movies["languages"].fillna("").str.split("|").map(lambda x: language in x)]
    if start_year is not None:
        movies = movies[movies["release_year"] >= int(start_year)]
    if end_year is not None:
        movies = movies[movies["release_year"] <= int(end_year)]
    return movies.head(limit).reset_index(drop=True)


def chinese_rated_facets(root: str = str(PROJECT_ROOT)) -> tuple[list[str], list[str]]:
    movies = load_chinese_rated_catalog(root)
    origins = sorted({v for text in movies["origins"] for v in str(text).split("|") if v})
    languages = sorted({v for text in movies["languages"] for v in str(text).split("|") if v})
    return origins, languages


def get_reference_timestamp(user_id: int, train: pd.DataFrame | None = None) -> int:
    """使用用户最新训练时间；没有历史时回退到训练集最大时间。"""
    frame = load_train_ratings() if train is None else train
    user_rows = frame.loc[frame["user_id"] == int(user_id), "timestamp"]
    return int(user_rows.max() if not user_rows.empty else frame["timestamp"].max())


def build_prediction_frame(
    user_id: int,
    movie_ids: list[int] | pd.Series,
    root: str = str(PROJECT_ROOT),
) -> pd.DataFrame:
    """构造与第3、4阶段模型兼容的用户—电影交互特征。"""
    ratings, movies, users = load_raw_tables(root)
    del ratings
    if int(user_id) not in set(users["user_id"]):
        raise ValueError(f"用户ID不存在：{user_id}")
    ids = [int(value) for value in movie_ids]
    if not ids:
        return pd.DataFrame()
    unknown = sorted(set(ids) - set(movies["movie_id"]))
    if unknown:
        raise ValueError(f"电影ID不存在：{unknown[:10]}")
    interactions = pd.DataFrame({
        "user_id": int(user_id),
        "movie_id": ids,
        "timestamp": get_reference_timestamp(user_id, load_train_ratings(root)),
    })
    return enrich_interactions(interactions, users, movies)


def rating_distribution(movie_id: int, root: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """电影详情分布只呈现训练期已观测评分。"""
    train = load_train_ratings(root)
    counts = train.loc[train["movie_id"] == int(movie_id), "rating"].value_counts()
    return pd.DataFrame({
        "评分": list(range(1, 6)),
        "数量": [int(counts.get(score, 0)) for score in range(1, 6)],
    })


def all_genres(root: str = str(PROJECT_ROOT)) -> list[str]:
    _, movies, _ = load_raw_tables(root)
    values: set[str] = set()
    for text in movies["genres"].dropna().astype(str):
        values.update(value for value in text.split("|") if value)
    return sorted(values)
