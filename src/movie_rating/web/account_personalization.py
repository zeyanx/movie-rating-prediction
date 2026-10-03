"""新账户冷启动画像匹配。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .config import PROJECT_ROOT
from .data_service import load_raw_tables, load_train_ratings
from .unified_service import GENRE_LABELS


@st.cache_data(show_spinner=False)
def _user_genre_affinity(root: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """计算训练用户相对自身均分的类型偏好，避免高分习惯造成偏差。"""
    train = load_train_ratings(root)
    _, movies, _ = load_raw_tables(root)
    history = train.merge(
        movies[["movie_id", "genres"]], on="movie_id", how="left", validate="many_to_one"
    )
    history["user_mean"] = history.groupby("user_id")["rating"].transform("mean")
    history["affinity"] = history["rating"] - history["user_mean"]
    history["genre"] = history["genres"].fillna("").str.split("|")
    exploded = history.explode("genre")
    exploded = exploded[exploded["genre"].ne("")]
    return exploded.groupby(["user_id", "genre"], as_index=False).agg(
        affinity=("affinity", "mean"), support=("rating", "size")
    )


def choose_proxy_user_id(
    preferred_genres_zh: list[str] | tuple[str, ...],
    root: str = str(PROJECT_ROOT),
) -> int:
    """根据注册时选择的题材，确定可复现的MovieLens代理用户。"""
    selected = {str(value).strip() for value in preferred_genres_zh if str(value).strip()}
    if not selected:
        raise ValueError("请至少选择一种偏好类型")
    english = {source for source, label in GENRE_LABELS.items() if label in selected}
    affinity = _user_genre_affinity(root)
    matched = affinity[affinity["genre"].isin(english)].copy()
    if matched.empty:
        return 1
    # 支持数使用对数平滑，兼顾偏好强度和证据量。
    matched["weighted"] = matched["affinity"] * np.log1p(matched["support"])
    scores = matched.groupby("user_id", as_index=False).agg(
        preference_score=("weighted", "mean"),
        genre_coverage=("genre", "nunique"),
        support=("support", "sum"),
    )
    scores = scores.sort_values(
        ["genre_coverage", "preference_score", "support", "user_id"],
        ascending=[False, False, False, True], kind="mergesort",
    )
    return int(scores.iloc[0]["user_id"])
