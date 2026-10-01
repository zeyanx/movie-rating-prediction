"""第4阶段神经网络的数据合并、无泄漏预处理与Dataset。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# MovieLens 100K u.item 的19个官方类型，顺序与原始定义保持一致。
GENRE_CATEGORIES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

REQUIRED_COLUMNS = [
    "user_id", "movie_id", "timestamp", "age", "gender", "occupation",
    "release_date", "genres",
]

NUMERIC_FEATURE_NAMES = [
    "age_standardized", "release_year_standardized", "rating_year_standardized",
    "movie_age_at_rating_standardized", "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos", "hour_sin", "hour_cos",
]


def enrich_interactions(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """将评分交互与用户、电影静态信息进行多对一合并。"""
    enriched = interactions.merge(users, on="user_id", how="left", validate="many_to_one")
    enriched = enriched.merge(movies, on="movie_id", how="left", validate="many_to_one")
    if len(enriched) != len(interactions):
        raise ValueError("评分与静态信息合并后行数发生变化")
    return enriched


class NeuralFeaturePreprocessor:
    """只从拟合数据学习类别映射、填补值和连续特征标准化参数。"""

    unknown_index = 0

    def __init__(self) -> None:
        self.fitted_ = False

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("神经网络预处理器要求输入pandas DataFrame")
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"神经网络输入缺少字段：{missing}")

    @staticmethod
    def _build_mapping(series: pd.Series) -> dict[Any, int]:
        values = sorted(series.dropna().unique().tolist(), key=lambda value: str(value))
        return {value: index + 1 for index, value in enumerate(values)}

    @staticmethod
    def _time_and_numeric(
        frame: pd.DataFrame, release_year_fill: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rating_time = pd.to_datetime(frame["timestamp"], unit="s", utc=True, errors="coerce")
        if rating_time.isna().any():
            raise ValueError("存在无法解析的评分时间戳")
        release_year = pd.to_datetime(
            frame["release_date"], format="%d-%b-%Y", errors="coerce"
        ).dt.year.astype(float).fillna(float(release_year_fill))
        rating_year = rating_time.dt.year.astype(float)
        age = pd.to_numeric(frame["age"], errors="coerce").astype(float)
        movie_age = np.maximum(rating_year.to_numpy() - release_year.to_numpy(), 0.0)
        continuous = np.column_stack(
            [age.to_numpy(), release_year.to_numpy(), rating_year.to_numpy(), movie_age]
        ).astype(np.float64)

        month_angle = 2.0 * np.pi * (rating_time.dt.month.to_numpy() - 1.0) / 12.0
        day_angle = 2.0 * np.pi * rating_time.dt.dayofweek.to_numpy() / 7.0
        hour_angle = 2.0 * np.pi * rating_time.dt.hour.to_numpy() / 24.0
        cyclical = np.column_stack(
            [np.sin(month_angle), np.cos(month_angle), np.sin(day_angle),
             np.cos(day_angle), np.sin(hour_angle), np.cos(hour_angle)]
        ).astype(np.float32)
        return continuous, cyclical

    def fit(self, frame: pd.DataFrame) -> "NeuralFeaturePreprocessor":
        """拟合映射和标准化器；评分列即使存在也不会被读取。"""
        self._validate_frame(frame)
        self.user_mapping_ = self._build_mapping(frame["user_id"])
        self.movie_mapping_ = self._build_mapping(frame["movie_id"])
        self.gender_mapping_ = self._build_mapping(frame["gender"].astype("string"))
        self.occupation_mapping_ = self._build_mapping(frame["occupation"].astype("string"))

        parsed_release_year = pd.to_datetime(
            frame["release_date"], format="%d-%b-%Y", errors="coerce"
        ).dt.year.astype(float)
        if parsed_release_year.notna().sum() == 0:
            raise ValueError("拟合数据中没有可解析的电影上映年份")
        self.release_year_fill_ = float(parsed_release_year.median())

        continuous, _ = self._time_and_numeric(frame, self.release_year_fill_)
        self.continuous_fill_values_ = np.nanmedian(continuous, axis=0)
        if not np.isfinite(self.continuous_fill_values_).all():
            raise ValueError("连续特征无法确定有限的训练集填补值")
        filled = np.where(np.isfinite(continuous), continuous, self.continuous_fill_values_)
        self.scaler_ = StandardScaler().fit(filled)
        self.genre_categories_ = list(GENRE_CATEGORIES)
        self.numeric_feature_names_ = list(NUMERIC_FEATURE_NAMES)
        self.input_feature_names_ = [
            "user_embedding_index", "movie_embedding_index",
            "gender_embedding_index", "occupation_embedding_index",
            *self.numeric_feature_names_,
            *[f"genre={genre}" for genre in self.genre_categories_],
        ]
        self.fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("NeuralFeaturePreprocessor尚未拟合")

    @staticmethod
    def _map_with_unknown(series: pd.Series, mapping: dict[Any, int]) -> np.ndarray:
        return series.map(mapping).fillna(0).astype(np.int64).to_numpy()

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """转换为模型张量所需数组；未知类别统一映射为0。"""
        self._check_fitted()
        self._validate_frame(frame)
        continuous, cyclical = self._time_and_numeric(frame, self.release_year_fill_)
        continuous = np.where(
            np.isfinite(continuous), continuous, self.continuous_fill_values_
        )
        continuous_scaled = self.scaler_.transform(continuous).astype(np.float32)
        numeric = np.column_stack([continuous_scaled, cyclical]).astype(np.float32)

        genres = frame["genres"].fillna("").astype(str)
        genre_matrix = np.column_stack(
            [genres.str.split("|").map(lambda tokens, name=genre: name in tokens).to_numpy()
             for genre in self.genre_categories_]
        ).astype(np.float32)

        transformed = {
            "user_index": self._map_with_unknown(frame["user_id"], self.user_mapping_),
            "movie_index": self._map_with_unknown(frame["movie_id"], self.movie_mapping_),
            "gender_index": self._map_with_unknown(
                frame["gender"].astype("string"), self.gender_mapping_
            ),
            "occupation_index": self._map_with_unknown(
                frame["occupation"].astype("string"), self.occupation_mapping_
            ),
            "numeric": numeric,
            "genres": genre_matrix,
        }
        for name, values in transformed.items():
            if not np.isfinite(values).all():
                raise ValueError(f"转换后的{name}包含NaN或无穷值")
        return transformed

    def fit_transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(frame).transform(frame)

    @property
    def vocabulary_sizes(self) -> dict[str, int]:
        self._check_fitted()
        return {
            "user": len(self.user_mapping_) + 1,
            "movie": len(self.movie_mapping_) + 1,
            "gender": len(self.gender_mapping_) + 1,
            "occupation": len(self.occupation_mapping_) + 1,
        }

    @property
    def numeric_feature_count(self) -> int:
        return len(self.numeric_feature_names_)

    @property
    def genre_feature_count(self) -> int:
        return len(self.genre_categories_)

    def save(self, path: Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "NeuralFeaturePreprocessor":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError("预处理器文件类型不正确")
        loaded._check_fitted()
        return loaded


class MovieRatingDataset(Dataset):
    """将预处理后的NumPy数组一次性转换为CPU张量。"""

    def __init__(
        self,
        features: dict[str, np.ndarray],
        labels: np.ndarray | pd.Series | None = None,
    ) -> None:
        lengths = {len(values) for values in features.values()}
        if len(lengths) != 1:
            raise ValueError("各输入特征的样本数不一致")
        self.features = {
            "user_index": torch.as_tensor(features["user_index"], dtype=torch.long),
            "movie_index": torch.as_tensor(features["movie_index"], dtype=torch.long),
            "gender_index": torch.as_tensor(features["gender_index"], dtype=torch.long),
            "occupation_index": torch.as_tensor(features["occupation_index"], dtype=torch.long),
            "numeric": torch.as_tensor(features["numeric"], dtype=torch.float32),
            "genres": torch.as_tensor(features["genres"], dtype=torch.float32),
        }
        self.labels = None
        if labels is not None:
            values = np.asarray(labels, dtype=np.float32).reshape(-1)
            if len(values) != self.__len__():
                raise ValueError("标签数量与输入特征数量不一致")
            if not np.isfinite(values).all():
                raise ValueError("标签包含NaN或无穷值")
            self.labels = torch.as_tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        return int(next(iter(self.features.values())).shape[0])

    def __getitem__(self, index: int):
        item = {name: values[index] for name, values in self.features.items()}
        if self.labels is None:
            return item
        return item, self.labels[index]
