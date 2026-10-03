"""中国电影真实评分库的数据检索与模型推理服务。

该模块使用独立文件名，避免 Streamlit Cloud 增量更新时复用旧版
``data_service``/``model_service`` 模块缓存。
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from movie_rating.chinese_models import (
    ChineseRatingMLP,
    build_numeric_features,
    build_tree_features,
    standardize_numeric,
)

from .config import PROJECT_ROOT


@st.cache_data(show_spinner=False)
def load_chinese_rated_catalog(root: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """加载可公开部署的300部中国电影目录。"""
    path = Path(root) / "data" / "catalog" / "chinese_rated_movies.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少中国电影训练目录：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "catalog_id", "cn_movie_index", "title_zh", "title_en", "release_year",
        "origins", "languages", "genres", "wikidata_id", "imdb_id", "source_url",
    }
    if not required.issubset(frame.columns) or len(frame) < 100:
        raise ValueError("中国电影训练目录字段或规模异常")
    if frame["catalog_id"].duplicated().any() or frame["cn_movie_index"].duplicated().any():
        raise ValueError("中国电影训练目录主键重复")
    frame["release_year"] = pd.to_numeric(frame["release_year"], errors="coerce").astype("Int64")
    return frame.sort_values(
        ["release_year", "title_zh"], ascending=[False, True]
    ).reset_index(drop=True)


def search_chinese_rated_catalog(
    query: str = "",
    origin: str | None = None,
    language: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    limit: int = 100,
    root: str = str(PROJECT_ROOT),
) -> pd.DataFrame:
    """按片名、地区、语言和年份检索真实评分训练库。"""
    if not 1 <= int(limit) <= 500:
        raise ValueError("搜索结果上限必须位于1至500")
    movies = load_chinese_rated_catalog(root).copy()
    clean_query = str(query).strip()
    if clean_query:
        searchable = movies["title_zh"].fillna("") + " " + movies["title_en"].fillna("")
        movies = movies[searchable.str.contains(clean_query, case=False, regex=False, na=False)]
    if origin and origin != "全部":
        movies = movies[movies["origins"].fillna("").str.split("|").map(lambda values: origin in values)]
    if language and language != "全部":
        movies = movies[
            movies["languages"].fillna("").str.split("|").map(lambda values: language in values)
        ]
    if start_year is not None:
        movies = movies[movies["release_year"] >= int(start_year)]
    if end_year is not None:
        movies = movies[movies["release_year"] <= int(end_year)]
    return movies.head(limit).reset_index(drop=True)


def chinese_rated_facets(root: str = str(PROJECT_ROOT)) -> tuple[list[str], list[str]]:
    """返回页面筛选器使用的出品地区与原始语言枚举。"""
    movies = load_chinese_rated_catalog(root)
    origins = sorted({value for text in movies["origins"] for value in str(text).split("|") if value})
    languages = sorted(
        {value for text in movies["languages"] for value in str(text).split("|") if value}
    )
    return origins, languages


@st.cache_resource(show_spinner=False)
def load_cached_chinese_models(root: str = str(PROJECT_ROOT)) -> dict[str, object]:
    """按进程缓存中国电影随机森林、XGBoost和MLP。"""
    base = Path(root) / "models" / "chinese"
    metadata_path = base / "model_metadata.json"
    required = [
        metadata_path,
        base / "mlp.pt",
        base / "random_forest.joblib",
        base / "xgboost.joblib",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少中国电影模型产物：{missing}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = ChineseRatingMLP(
        int(metadata["num_users"]),
        int(metadata["num_movies"]),
        int(metadata["numeric_features"]),
        int(metadata["embedding_dim"]),
        float(metadata["dropout"]),
    )
    try:
        state = torch.load(base / "mlp.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(base / "mlp.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return {
        "metadata": metadata,
        "mlp": model,
        "random_forest": joblib.load(base / "random_forest.joblib"),
        "xgboost": joblib.load(base / "xgboost.joblib"),
    }


def predict_chinese_ratings(
    cn_user_id: int,
    movie_indices: list[int] | np.ndarray,
    model_names: tuple[str, ...] = ("xgboost", "random_forest", "mlp"),
    root: str = str(PROJECT_ROOT),
) -> pd.DataFrame:
    """在独立中国电影用户/影片空间中批量预测评分。"""
    loaded = load_cached_chinese_models(root)
    metadata = loaded["metadata"]
    user_id = int(cn_user_id)
    if not 1 <= user_id <= int(metadata["num_users"]):
        raise ValueError(f"中国电影用户编号必须位于1至{metadata['num_users']}")

    movie_ids = np.asarray(movie_indices, dtype=np.int64).reshape(-1)
    if movie_ids.size == 0:
        return pd.DataFrame(columns=list(model_names))
    if movie_ids.min() < 1 or movie_ids.max() > int(metadata["num_movies"]):
        raise ValueError("中国电影模型索引超出范围")

    users = np.full(movie_ids.size, user_id, dtype=np.int64)
    timestamps = np.full(movie_ids.size, int(metadata["reference_timestamp"]), dtype=np.int64)
    preprocessing = metadata["preprocessing"]
    output: dict[str, np.ndarray] = {}
    for name in model_names:
        if name == "mlp":
            numeric = standardize_numeric(
                build_numeric_features(users, movie_ids, timestamps, preprocessing),
                preprocessing,
            )
            with torch.no_grad():
                values = loaded["mlp"](
                    torch.as_tensor(users, dtype=torch.long),
                    torch.as_tensor(movie_ids, dtype=torch.long),
                    torch.as_tensor(numeric, dtype=torch.float32),
                ).numpy()
        elif name in {"xgboost", "random_forest"}:
            features = build_tree_features(users, movie_ids, timestamps, preprocessing)
            values = loaded[name]["model"].predict(features)
        else:
            raise ValueError(f"不支持的中国电影模型：{name}")
        output[name] = np.clip(np.asarray(values, dtype=float), 0.5, 5.0)
    return pd.DataFrame(output)
