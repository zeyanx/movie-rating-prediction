"""统一电影库、评分预测和跨数据源推荐服务。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .chinese_service import (
    load_chinese_rated_catalog,
    predict_chinese_ratings,
)
from .config import PROJECT_ROOT, load_app_config
from .data_service import build_prediction_frame, load_raw_tables, load_train_ratings, movie_statistics
from .model_service import predict_with_models
from .ml25m_service import (
    load_ml25m_catalog,
    ml25m_catalog_ready,
    predict_ml25m_item,
    recommend_ml25m_movies,
)
from .recommendation import (
    build_candidate_frame,
    compute_user_genre_profile,
    generate_recommendation_reasons,
    score_candidates,
)


GENRE_LABELS = {
    "unknown": "未知",
    "Action": "动作",
    "Adventure": "冒险",
    "Animation": "动画",
    "Children's": "儿童",
    "Children": "儿童",
    "Comedy": "喜剧",
    "Crime": "犯罪",
    "Documentary": "纪录片",
    "Drama": "剧情",
    "Fantasy": "奇幻",
    "Film-Noir": "黑色电影",
    "Horror": "恐怖",
    "Musical": "歌舞",
    "Mystery": "悬疑",
    "Romance": "爱情",
    "Sci-Fi": "科幻",
    "Thriller": "惊悚",
    "War": "战争",
    "Western": "西部",
    "(no genres listed)": "未分类",
}

MODEL_LABELS_ZH = {
    "xgboost": "梯度提升树",
    "random_forest": "随机森林",
    "mlp": "神经网络",
}


@st.cache_data(show_spinner=False)
def load_chinese_metadata(root: str = str(PROJECT_ROOT)) -> dict[str, Any]:
    """只读取轻量元数据，浏览电影库时不提前加载模型文件。"""
    path = Path(root) / "models" / "chinese" / "model_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少中国电影模型元数据：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def translate_genres(value: object) -> str:
    """把官方英文类型转换为中文显示文本。"""
    labels = [GENRE_LABELS.get(token, token) for token in str(value or "").split("|") if token]
    return "、".join(labels) if labels else "未分类"


def localize_reason(value: object) -> str:
    """把推荐理由中的模型缩写和英文类型名称替换为中文。"""
    result = str(value).replace("MLP", "神经网络模型")
    for source, target in sorted(GENRE_LABELS.items(), key=lambda item: len(item[0]), reverse=True):
        result = result.replace(source, target)
    return result


@st.cache_data(show_spinner=False)
def unified_catalog(root: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """合并具有中文片名且可由现有模型预测的两类电影。"""
    international = movie_statistics(root)
    international = international[international["title_zh"].ne("")].copy()
    international_frame = pd.DataFrame({
        "item_key": "international:" + international["movie_id"].astype(str),
        "model_space": "international",
        "local_id": international["movie_id"].astype(int),
        "title_zh": international["title_zh"],
        "search_title": (
            international["title_zh"].fillna("") + " "
            + international["title"].fillna("") + " "
            + international["aliases_zh"].fillna("")
        ),
        "origin": "其他国家和地区",
        "category": international["genres"].map(translate_genres),
        "release_year": international["release_year"],
        "rating_count": international["rating_count"].astype(int),
        "rating_mean": international["rating_mean"].astype(float),
        "source_url": international["imdb_url"].fillna(""),
    })

    chinese = load_chinese_rated_catalog(root).copy()
    metadata = load_chinese_metadata(root)
    preprocessing = metadata["preprocessing"]
    movie_mean = np.asarray(preprocessing["movie_mean"], dtype=float)
    movie_count = np.asarray(preprocessing["movie_count"], dtype=int)
    indices = chinese["cn_movie_index"].astype(int).to_numpy()
    chinese_frame = pd.DataFrame({
        "item_key": "china:" + chinese["cn_movie_index"].astype(str),
        "model_space": "china",
        "local_id": indices,
        "title_zh": chinese["title_zh"],
        "search_title": chinese["title_zh"].fillna("") + " " + chinese["title_en"].fillna(""),
        "origin": chinese["origins"].fillna("中国"),
        "category": chinese["genres"].map(translate_genres),
        "release_year": chinese["release_year"],
        "rating_count": movie_count[indices],
        "rating_mean": movie_mean[indices],
        "source_url": chinese["source_url"].fillna(""),
    })

    result = pd.concat([chinese_frame, international_frame], ignore_index=True)
    result["release_year"] = pd.to_numeric(result["release_year"], errors="coerce").astype("Int64")
    return result.sort_values(
        ["release_year", "title_zh"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


def search_unified_catalog(
    query: str = "",
    genre: str = "全部",
    region: str = "全部",
    start_year: int | None = None,
    end_year: int | None = None,
    limit: int = 500,
    root: str = str(PROJECT_ROOT),
) -> pd.DataFrame:
    """在合并电影库中进行中文优先的字面检索。"""
    if not 1 <= int(limit) <= 500:
        raise ValueError("结果数量必须位于1至500")
    movies = unified_catalog(root).copy()
    clean_query = str(query).strip()
    if clean_query:
        movies = movies[
            movies["search_title"].str.contains(clean_query, case=False, regex=False, na=False)
        ]
    if genre and genre != "全部":
        movies = movies[
            movies["category"].str.split("、").map(lambda values: genre in values)
        ]
    if region == "中国电影":
        movies = movies[movies["model_space"] == "china"]
    elif region == "其他国家和地区":
        movies = movies[movies["model_space"] == "international"]
    if start_year is not None:
        movies = movies[movies["release_year"] >= int(start_year)]
    if end_year is not None:
        movies = movies[movies["release_year"] <= int(end_year)]
    return movies.head(int(limit)).reset_index(drop=True)


def unified_genres(root: str = str(PROJECT_ROOT)) -> list[str]:
    """返回中国电影和其他国家电影共用的中文类型列表。"""
    catalog = unified_catalog(root)
    values = {
        genre
        for text in catalog["category"].fillna("")
        for genre in str(text).split("、")
        if genre and genre != "未分类"
    }
    return sorted(values)


@st.cache_data(show_spinner=False)
def expanded_catalog(root: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """返回MovieLens 25M完整目录；对齐项保留现有训练模型空间。"""
    catalog = load_ml25m_catalog(root).copy()
    catalog["category"] = catalog["genres"].map(translate_genres)
    return catalog.sort_values(
        ["release_year", "title_zh"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


def search_expanded_catalog(
    query: str = "",
    genre: str = "全部",
    start_year: int | None = None,
    end_year: int | None = None,
    limit: int = 500,
    root: str = str(PROJECT_ROOT),
) -> pd.DataFrame:
    """在62,423部电影中按片名、类型和年份统一检索。"""
    if not 1 <= int(limit) <= 500:
        raise ValueError("结果数量必须位于1至500")
    movies = expanded_catalog(root).copy()
    clean_query = str(query).strip()
    if clean_query:
        movies = movies[movies["search_title"].str.contains(clean_query, case=False, regex=False, na=False)]
    if genre and genre != "全部":
        movies = movies[movies["category"].str.split("、").map(lambda values: genre in values)]
    if start_year is not None:
        movies = movies[movies["release_year"] >= int(start_year)]
    if end_year is not None:
        movies = movies[movies["release_year"] <= int(end_year)]
    return movies.head(int(limit)).reset_index(drop=True)


def expanded_genres(root: str = str(PROJECT_ROOT)) -> list[str]:
    """返回大型目录中的中文类型选项。"""
    values = {
        genre
        for text in expanded_catalog(root)["category"].fillna("")
        for genre in str(text).split("、")
        if genre and genre != "未分类"
    }
    return sorted(values)


def predict_expanded_item(
    user_id: int, item_key: str, root: str = str(PROJECT_ROOT)
) -> dict[str, Any]:
    """能对齐的影片调用训练模型，其余影片调用25M个性化统计模型。"""
    matches = expanded_catalog(root)
    matches = matches[matches["item_key"] == str(item_key)]
    if matches.empty:
        raise ValueError(f"大型电影库中不存在：{item_key}")
    movie = matches.iloc[0]
    if movie["model_space"] == "ml25m":
        return predict_ml25m_item(user_id, item_key, root)
    routed_key = f"{movie['model_space']}:{int(movie['local_id'])}"
    return predict_unified_item(user_id, routed_key, root)


def large_catalog_ready(root: str = str(PROJECT_ROOT)) -> bool:
    """供页面轻量判断大型目录是否已经生成。"""
    return ml25m_catalog_ready(root)


def predict_unified_item(
    user_id: int,
    item_key: str,
    root: str = str(PROJECT_ROOT),
) -> dict[str, Any]:
    """根据影片来源自动选择正确模型并返回中文预测说明。"""
    matches = unified_catalog(root)
    matches = matches[matches["item_key"] == str(item_key)]
    if matches.empty:
        raise ValueError(f"电影不存在：{item_key}")
    movie = matches.iloc[0]
    local_id = int(movie["local_id"])

    if movie["model_space"] == "china":
        values = predict_chinese_ratings(int(user_id), [local_id], root=root).iloc[0]
        scores = {MODEL_LABELS_ZH[name]: float(values[name]) for name in values.index}
        default_model = "梯度提升树"
        score = scores[default_model]
    else:
        frame = build_prediction_frame(int(user_id), [local_id], root)
        values = predict_with_models(frame, ["random_forest", "xgboost", "mlp"], root).iloc[0]
        scores = {MODEL_LABELS_ZH[name]: float(values[name]) for name in values.index}
        default_model = "神经网络"
        score = scores[default_model]

    reasons = [
        f"{default_model}预测当前用户可能给出 {score:.2f} 分",
        f"该影片历史平均评分为 {float(movie['rating_mean']):.2f} 分",
        f"评分依据包含 {int(movie['rating_count'])} 条历史记录",
    ]
    return {
        "movie": movie.to_dict(),
        "score": score,
        "default_model": default_model,
        "scores": scores,
        "reasons": reasons,
    }


def _international_recommendations(
    user_id: int,
    interactions: pd.DataFrame | None,
    count: int,
    genre: str,
    root: str,
    offset: int = 0,
) -> pd.DataFrame:
    config = load_app_config()
    train = load_train_ratings(root)
    _, movies, _ = load_raw_tables(root)
    stats = movie_statistics(root, int(config["popularity_prior_count"]))
    candidates, _ = build_candidate_frame(
        user_id, stats, train, interactions,
        minimum_popularity_count=0, top_n=max(count + offset, 1),
    )
    candidates = candidates[candidates["title_zh"].ne("")].reset_index(drop=True)
    if genre and genre != "全部":
        candidates = candidates[
            candidates["genres"].map(translate_genres).str.split("、").map(
                lambda values: genre in values
            )
        ].reset_index(drop=True)
    if candidates.empty:
        return candidates
    model_input = build_prediction_frame(user_id, candidates["movie_id"].astype(int).tolist(), root)
    predictions = predict_with_models(model_input, ["mlp"], root)["mlp"].to_numpy()
    profile = compute_user_genre_profile(user_id, train, movies, interactions)
    scored = score_candidates(candidates, predictions, profile, config["recommendation_weights"])
    sparse = int((train["user_id"] == int(user_id)).sum()) < 10
    scored["reasons"] = scored.apply(
        lambda row: [localize_reason(reason) for reason in generate_recommendation_reasons(row, sparse)],
        axis=1,
    )
    scored["item_key"] = "international:" + scored["movie_id"].astype(str)
    scored["model_space"] = "international"
    scored["origin"] = "其他国家和地区"
    scored["category"] = scored["genres"].map(translate_genres)
    scored["title_zh"] = scored["title_zh"].astype(str)
    scored["model_label"] = "神经网络"
    ranked = scored.sort_values(
        ["final_score", "mlp_prediction", "rating_count"],
        ascending=[False, False, False], kind="mergesort",
    ).reset_index(drop=True)
    return ranked.iloc[offset:offset + count].reset_index(drop=True)


def _chinese_recommendations(
    user_id: int,
    count: int,
    genre: str,
    root: str,
    catalog_interactions: pd.DataFrame | None = None,
    offset: int = 0,
) -> pd.DataFrame:
    catalog = load_chinese_rated_catalog(root).copy()
    if catalog_interactions is not None and not catalog_interactions.empty:
        blocked = catalog_interactions[catalog_interactions["status"] == "watched"]
        catalog = catalog[
            ~catalog["catalog_id"].isin(blocked["catalog_id"].astype(str))
        ].reset_index(drop=True)
    if genre and genre != "全部":
        catalog = catalog[
            catalog["genres"].map(translate_genres).str.split("、").map(
                lambda values: genre in values
            )
        ].reset_index(drop=True)
    if catalog.empty:
        return catalog
    indices = catalog["cn_movie_index"].astype(int).to_numpy()
    predictions = predict_chinese_ratings(
        int(user_id), indices, model_names=("xgboost",), root=root
    )["xgboost"].to_numpy()
    preprocessing = load_chinese_metadata(root)["preprocessing"]
    movie_mean = np.asarray(preprocessing["movie_mean"], dtype=float)[indices]
    movie_count = np.asarray(preprocessing["movie_count"], dtype=int)[indices]
    result = catalog.copy()
    result["item_key"] = "china:" + result["cn_movie_index"].astype(str)
    result["model_space"] = "china"
    result["origin"] = result["origins"]
    result["category"] = result["genres"].map(translate_genres)
    result["mlp_prediction"] = predictions
    result["rating_count"] = movie_count
    result["rating_mean"] = movie_mean
    result["final_score"] = (0.9 * predictions + 0.1 * movie_mean).clip(0.5, 5.0)
    result["model_label"] = "梯度提升树"
    result["reasons"] = result.apply(
        lambda row: [
            f"梯度提升树预测当前用户可能给出 {float(row['mlp_prediction']):.2f} 分",
            f"该影片历史平均评分为 {float(row['rating_mean']):.2f} 分",
            f"模型参考了 {int(row['rating_count'])} 条该片历史评分",
        ],
        axis=1,
    )
    ranked = result.sort_values(
        ["final_score", "rating_count", "cn_movie_index"],
        ascending=[False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    return ranked.iloc[offset:offset + count].reset_index(drop=True)


def recommend_unified_movies(
    user_id: int,
    interactions: pd.DataFrame | None = None,
    top_n: int = 10,
    genre: str = "全部",
    region: str = "全部",
    catalog_interactions: pd.DataFrame | None = None,
    root: str = str(PROJECT_ROOT),
    refresh_page: int = 0,
) -> pd.DataFrame:
    """混排两类电影；refresh_page用于稳定地切换到下一批候选。"""
    n = int(top_n)
    if not 1 <= n <= 20:
        raise ValueError("推荐数量必须位于1至20")
    page = int(refresh_page)
    if page < 0:
        raise ValueError("刷新页码不能为负数")
    if region == "中国电影":
        return _chinese_recommendations(
            user_id, n, genre, root, catalog_interactions, offset=page * n
        )
    if region == "其他国家和地区":
        return _international_recommendations(
            user_id, interactions, n, genre, root, offset=page * n
        )

    chinese_count = max(1, n // 2)
    international_count = max(1, n - chinese_count)
    chinese = _chinese_recommendations(
        user_id,
        n,
        genre,
        root,
        catalog_interactions,
        offset=page * chinese_count,
    )
    international = _international_recommendations(
        user_id,
        interactions,
        n,
        genre,
        root,
        offset=page * international_count,
    )

    # 优先保持两类来源均衡；某类候选不足时由另一类自动补齐。
    selected_chinese = chinese.head(chinese_count)
    selected_international = international.head(international_count)
    remaining = n - len(selected_chinese) - len(selected_international)
    if remaining > 0:
        chinese_extra = chinese.iloc[len(selected_chinese):]
        international_extra = international.iloc[len(selected_international):]
        extras = pd.concat([chinese_extra, international_extra], ignore_index=True, sort=False)
        if not extras.empty:
            extras = extras.sort_values(
                ["final_score", "rating_count", "title_zh"],
                ascending=[False, False, True], kind="mergesort",
            ).head(remaining)
        selected_chinese = pd.concat([selected_chinese, extras], ignore_index=True, sort=False)
    combined = pd.concat(
        [selected_chinese, selected_international], ignore_index=True, sort=False
    )
    if combined.empty:
        return combined
    return combined.sort_values(
        ["final_score", "rating_count", "title_zh"],
        ascending=[False, False, True], kind="mergesort",
    ).head(n).reset_index(drop=True)
