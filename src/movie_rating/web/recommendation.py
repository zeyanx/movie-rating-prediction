"""个性化候选过滤、画像、排序和可解释推荐理由。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_app_config
from .data_service import build_prediction_frame, load_raw_tables, load_train_ratings, movie_statistics
from .model_service import predict_with_mlp


def _genre_tokens(value: object) -> list[str]:
    return [token for token in str(value or "").split("|") if token and token != "nan"]


def compute_user_genre_profile(
    user_id: int,
    train: pd.DataFrame,
    movies: pd.DataFrame,
    interactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """用相对个人均分的偏差衡量类型偏好，并附带支持数量。"""
    history = train[train["user_id"] == int(user_id)].merge(
        movies[["movie_id", "genres"]], on="movie_id", how="left", validate="many_to_one"
    )
    rows: list[dict[str, Any]] = []
    if not history.empty:
        user_mean = float(history["rating"].mean())
        for row in history.itertuples(index=False):
            for genre in _genre_tokens(row.genres):
                rows.append({"genre": genre, "deviation": float(row.rating) - user_mean})
    if rows:
        profile = pd.DataFrame(rows).groupby("genre")["deviation"].agg(
            affinity="mean", support="count"
        ).reset_index()
    else:
        profile = pd.DataFrame(columns=["genre", "affinity", "support"])

    # 本地喜欢/不喜欢只作轻量修正，单条反馈不会压过历史画像。
    if interactions is not None and not interactions.empty:
        local = interactions.merge(
            movies[["movie_id", "genres"]], on="movie_id", how="left", validate="many_to_one"
        )
        adjustments: dict[str, list[float]] = {}
        for row in local.itertuples(index=False):
            delta = 0.25 if row.feedback == "like" else -0.25 if row.feedback == "dislike" else 0.0
            for genre in _genre_tokens(row.genres):
                adjustments.setdefault(genre, []).append(delta)
        if adjustments:
            indexed = profile.set_index("genre") if not profile.empty else pd.DataFrame(
                columns=["affinity", "support"]
            )
            for genre, values in adjustments.items():
                if genre not in indexed.index:
                    indexed.loc[genre, ["affinity", "support"]] = [0.0, 0]
                indexed.loc[genre, "affinity"] = float(indexed.loc[genre, "affinity"]) + sum(values)
            profile = indexed.reset_index()
    if not profile.empty:
        profile["support"] = profile["support"].fillna(0).astype(int)
        profile["affinity"] = profile["affinity"].astype(float).clip(-2.0, 2.0)
    return profile.sort_values(["affinity", "support"], ascending=[False, False]).reset_index(drop=True)


def compute_bayesian_popularity(
    train: pd.DataFrame, prior_count: int = 20
) -> pd.DataFrame:
    global_mean = float(train["rating"].mean())
    grouped = train.groupby("movie_id")["rating"].agg(rating_count="count", rating_mean="mean")
    grouped["bayesian_popularity"] = (
        grouped["rating_count"] * grouped["rating_mean"] + prior_count * global_mean
    ) / (grouped["rating_count"] + prior_count)
    return grouped.reset_index()


def build_candidate_frame(
    user_id: int,
    movies: pd.DataFrame,
    train: pd.DataFrame,
    interactions: pd.DataFrame | None = None,
    genre: str | None = None,
    minimum_popularity_count: int = 5,
    top_n: int = 10,
) -> tuple[pd.DataFrame, bool]:
    """构造候选集；必要时只放宽热度，不重新加入明确排除项。"""
    seen = set(train.loc[train["user_id"] == int(user_id), "movie_id"].astype(int))
    excluded = set(seen)
    if interactions is not None and not interactions.empty:
        blocked = interactions[
            (interactions["status"] == "watched")
            | (interactions["feedback"].isin(["dislike", "not_interested"]))
        ]
        excluded.update(blocked["movie_id"].astype(int).tolist())
    candidates = movies[~movies["movie_id"].isin(excluded)].copy()
    if genre and genre != "全部":
        candidates = candidates[
            candidates["genres"].fillna("").str.split("|").map(lambda values: genre in values)
        ]
    filtered = candidates[candidates["rating_count"] >= int(minimum_popularity_count)].copy()
    relaxed = len(filtered) < int(top_n) and int(minimum_popularity_count) > 0
    if relaxed:
        filtered = candidates.copy()
    return filtered.reset_index(drop=True), relaxed


def _genre_score(genres: str, affinity: dict[str, float]) -> tuple[float, list[str]]:
    tokens = _genre_tokens(genres)
    values = [affinity[token] for token in tokens if token in affinity]
    positive = [token for token in tokens if affinity.get(token, 0.0) > 0.05]
    score = 3.0 + (float(np.mean(values)) if values else 0.0)
    return float(np.clip(score, 1.0, 5.0)), positive


def score_candidates(
    candidates: pd.DataFrame,
    mlp_predictions: np.ndarray,
    genre_profile: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    if len(candidates) != len(mlp_predictions):
        raise ValueError("候选电影与MLP预测数量不一致")
    result = candidates.copy()
    affinity = dict(zip(genre_profile.get("genre", []), genre_profile.get("affinity", [])))
    scored = result["genres"].map(lambda value: _genre_score(str(value), affinity))
    result["genre_preference_score"] = scored.map(lambda value: value[0])
    result["matched_preferred_genres"] = scored.map(lambda value: value[1])
    result["mlp_prediction"] = np.clip(np.asarray(mlp_predictions, dtype=float), 1.0, 5.0)
    result["final_score"] = (
        float(weights["mlp_prediction"]) * result["mlp_prediction"]
        + float(weights["genre_preference"]) * result["genre_preference_score"]
        + float(weights["bayesian_popularity"]) * result["bayesian_popularity"].clip(1.0, 5.0)
    ).clip(1.0, 5.0)
    return result


def generate_recommendation_reasons(row: pd.Series, sparse_history: bool = False) -> list[str]:
    reasons = [f"MLP预测你可能会给出 {float(row['mlp_prediction']):.2f} 分"]
    preferred = row.get("matched_preferred_genres", [])
    if isinstance(preferred, list) and preferred:
        reasons.append(f"与你偏好的 {'、'.join(preferred[:3])} 类型匹配")
    if int(row.get("rating_count", 0)) >= 50:
        reasons.append(f"训练集中已有 {int(row['rating_count'])} 条评分，参考信息较充分")
    else:
        reasons.append(f"贝叶斯平滑口碑为 {float(row['bayesian_popularity']):.2f} 分")
    if sparse_history:
        reasons.append("你的训练期历史较少，本次更多参考模型预测和电影热度")
    return reasons[:3]


def recommend_movies(
    user_id: int,
    interactions: pd.DataFrame | None = None,
    top_n: int | None = None,
    genre: str | None = None,
    minimum_popularity_count: int | None = None,
    root: str = str(PROJECT_ROOT),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """批量推理并稳定排序，返回推荐结果和过滤说明。"""
    config = load_app_config()
    n = int(top_n or config["default_top_n"])
    if not 1 <= n <= int(config["max_top_n"]):
        raise ValueError("Top N超出配置范围")
    min_count = int(
        config["minimum_popularity_count"]
        if minimum_popularity_count is None else minimum_popularity_count
    )
    train = load_train_ratings(root)
    _, movies, _ = load_raw_tables(root)
    stats = movie_statistics(root, int(config["popularity_prior_count"]))
    candidates, relaxed = build_candidate_frame(
        user_id, stats, train, interactions, genre, min_count, n
    )
    if candidates.empty:
        return candidates, {"relaxed_popularity": relaxed, "candidate_count": 0}
    profile = compute_user_genre_profile(user_id, train, movies, interactions)
    model_input = build_prediction_frame(user_id, candidates["movie_id"].tolist(), root)
    predictions = predict_with_mlp(model_input, root)
    scored = score_candidates(candidates, predictions, profile, config["recommendation_weights"])
    sparse = int((train["user_id"] == int(user_id)).sum()) < 10
    scored["reasons"] = scored.apply(
        lambda row: generate_recommendation_reasons(row, sparse), axis=1
    )
    scored = scored.sort_values(
        ["final_score", "mlp_prediction", "rating_count", "movie_id"],
        ascending=[False, False, False, True], kind="mergesort",
    ).drop_duplicates("movie_id").head(n).reset_index(drop=True)
    metadata = {
        "relaxed_popularity": relaxed,
        "candidate_count": int(len(candidates)),
        "excluded_training_ratings": int((train["user_id"] == int(user_id)).sum()),
        "profile_genres": int(len(profile)),
    }
    return scored, metadata
