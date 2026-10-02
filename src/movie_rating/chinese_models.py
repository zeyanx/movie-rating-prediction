"""中国电影评分子集使用的轻量神经网络与推理辅助函数。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class ChineseRatingMLP(nn.Module):
    """用户/影片Embedding与无泄漏统计特征融合的评分网络。"""

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        numeric_features: int,
        embedding_dim: int = 24,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=0)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2 + numeric_features, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward(
        self, user_ids: torch.Tensor, movie_ids: torch.Tensor, numeric: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat(
            [self.user_embedding(user_ids), self.movie_embedding(movie_ids), numeric], dim=1
        )
        raw = self.network(features).squeeze(1)
        return 0.5 + 4.5 * torch.sigmoid(raw)


def build_numeric_features(
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    timestamps: np.ndarray,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    """使用训练集统计量构造时间、用户和影片数值特征。"""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    movie_ids = np.asarray(movie_ids, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    datetimes = timestamps.astype("datetime64[s]")
    years = datetimes.astype("datetime64[Y]").astype(np.int64) + 1970
    months = (datetimes.astype("datetime64[M]").astype(np.int64) % 12) + 1
    days = datetimes.astype("datetime64[D]").astype(np.int64)
    weekdays = (days + 3) % 7  # 1970-01-01为星期四，周一记为0。

    user_mean = np.asarray(preprocessing["user_mean"], dtype=np.float32)
    user_count = np.asarray(preprocessing["user_count"], dtype=np.float32)
    movie_mean = np.asarray(preprocessing["movie_mean"], dtype=np.float32)
    movie_count = np.asarray(preprocessing["movie_count"], dtype=np.float32)
    global_mean = float(preprocessing["global_mean"])

    safe_users = np.where((user_ids >= 0) & (user_ids < len(user_mean)), user_ids, 0)
    safe_movies = np.where((movie_ids >= 0) & (movie_ids < len(movie_mean)), movie_ids, 0)
    matrix = np.column_stack([
        years,
        np.sin(2 * np.pi * months / 12.0),
        np.cos(2 * np.pi * months / 12.0),
        np.sin(2 * np.pi * weekdays / 7.0),
        np.cos(2 * np.pi * weekdays / 7.0),
        user_mean[safe_users],
        np.log1p(user_count[safe_users]),
        movie_mean[safe_movies],
        np.log1p(movie_count[safe_movies]),
    ]).astype(np.float32)
    matrix[:, 5] = np.where(user_count[safe_users] > 0, matrix[:, 5], global_mean)
    matrix[:, 7] = np.where(movie_count[safe_movies] > 0, matrix[:, 7], global_mean)
    return matrix


def standardize_numeric(
    matrix: np.ndarray, preprocessing: dict[str, Any]
) -> np.ndarray:
    mean = np.asarray(preprocessing["numeric_mean"], dtype=np.float32)
    scale = np.asarray(preprocessing["numeric_scale"], dtype=np.float32)
    return ((matrix - mean) / scale).astype(np.float32)


def build_tree_features(
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    timestamps: np.ndarray,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    """构造与中国电影集成模型训练时完全一致的11维输入。"""
    numeric = build_numeric_features(user_ids, movie_ids, timestamps, preprocessing)
    return np.column_stack([
        np.asarray(user_ids, dtype=np.float32),
        np.asarray(movie_ids, dtype=np.float32),
        numeric,
    ]).astype(np.float32)
