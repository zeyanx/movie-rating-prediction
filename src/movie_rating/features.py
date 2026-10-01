"""第3阶段无泄漏电影评分特征工程。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


REQUIRED_INPUT_COLUMNS = [
    "user_id",
    "movie_id",
    "timestamp",
    "age",
    "gender",
    "occupation",
    "release_date",
    "genres",
]


class MovieFeatureEngineer(BaseEstimator, TransformerMixin):
    """构造静态特征与逐行留一的用户/电影目标统计特征。"""

    def __init__(
        self,
        user_smoothing: float = 10.0,
        movie_smoothing: float = 20.0,
    ) -> None:
        self.user_smoothing = user_smoothing
        self.movie_smoothing = movie_smoothing

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MovieFeatureEngineer要求输入pandas DataFrame")
        missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(f"特征输入缺少字段：{missing}")
        if "rating" in X.columns:
            raise ValueError("目标列rating不得进入特征转换器")
        if self.user_smoothing <= 0 or self.movie_smoothing <= 0:
            raise ValueError("平滑参数必须为正数")

    @staticmethod
    def _as_target(y: Any, expected_length: int) -> np.ndarray:
        if y is None:
            raise ValueError("拟合目标统计特征时必须提供y")
        values = np.asarray(y, dtype=float).reshape(-1)
        if values.size != expected_length:
            raise ValueError("X与y长度不一致")
        if values.size < 2 or not np.isfinite(values).all():
            raise ValueError("y必须包含至少两个有限评分")
        return values

    def fit(self, X: pd.DataFrame, y: Any) -> "MovieFeatureEngineer":
        """只从当前训练部分学习类别、填充值和目标统计映射。"""
        self._validate_input(X)
        target = self._as_target(y, len(X))
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.global_mean_ = float(target.mean())
        self.total_rating_sum_ = float(target.sum())
        self.total_rating_count_ = int(target.size)

        statistics = pd.DataFrame(
            {
                "user_id": X["user_id"].to_numpy(),
                "movie_id": X["movie_id"].to_numpy(),
                "target": target,
            }
        )
        user_stats = statistics.groupby("user_id", sort=False)["target"].agg(["sum", "count"])
        movie_stats = statistics.groupby("movie_id", sort=False)["target"].agg(["sum", "count"])
        self.user_sum_ = user_stats["sum"].to_dict()
        self.user_count_ = user_stats["count"].astype(int).to_dict()
        self.movie_sum_ = movie_stats["sum"].to_dict()
        self.movie_count_ = movie_stats["count"].astype(int).to_dict()

        self.gender_categories_ = sorted(X["gender"].dropna().astype(str).unique().tolist())
        self.occupation_categories_ = sorted(
            X["occupation"].dropna().astype(str).unique().tolist()
        )
        genre_tokens: set[str] = set()
        for value in X["genres"].dropna().astype(str):
            genre_tokens.update(token.strip() for token in value.split("|") if token.strip())
        self.genre_categories_ = sorted(genre_tokens)

        release_year = pd.to_datetime(
            X["release_date"], format="%d-%b-%Y", errors="coerce"
        ).dt.year
        if release_year.notna().sum() == 0:
            raise ValueError("当前训练数据没有可解析的电影上映年份")
        self.release_year_median_ = float(release_year.median())
        self.feature_names_out_ = np.asarray(self._build_feature_names(), dtype=object)
        return self

    def _target_statistics(
        self,
        X: pd.DataFrame,
        y_leave_one_out: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        user_sum = X["user_id"].map(self.user_sum_).to_numpy(dtype=float)
        user_count = X["user_id"].map(self.user_count_).to_numpy(dtype=float)
        movie_sum = X["movie_id"].map(self.movie_sum_).to_numpy(dtype=float)
        movie_count = X["movie_id"].map(self.movie_count_).to_numpy(dtype=float)

        if y_leave_one_out is not None:
            if len(y_leave_one_out) != len(X):
                raise ValueError("留一评分与输入长度不一致")
            loo_global = (
                self.total_rating_sum_ - y_leave_one_out
            ) / (self.total_rating_count_ - 1)
            user_effective_count = user_count - 1.0
            movie_effective_count = movie_count - 1.0
            user_mean = (
                user_sum
                - y_leave_one_out
                + self.user_smoothing * loo_global
            ) / (user_effective_count + self.user_smoothing)
            movie_mean = (
                movie_sum
                - y_leave_one_out
                + self.movie_smoothing * loo_global
            ) / (movie_effective_count + self.movie_smoothing)
            return user_mean, movie_mean, user_effective_count, movie_effective_count

        user_effective_count = np.nan_to_num(user_count, nan=0.0)
        movie_effective_count = np.nan_to_num(movie_count, nan=0.0)
        user_mean = (
            user_sum + self.user_smoothing * self.global_mean_
        ) / (user_count + self.user_smoothing)
        movie_mean = (
            movie_sum + self.movie_smoothing * self.global_mean_
        ) / (movie_count + self.movie_smoothing)
        user_mean = np.nan_to_num(user_mean, nan=self.global_mean_)
        movie_mean = np.nan_to_num(movie_mean, nan=self.global_mean_)
        return user_mean, movie_mean, user_effective_count, movie_effective_count

    def _build_feature_names(self) -> list[str]:
        base = [
            "user_age",
            "release_year",
            "rating_year",
            "rating_month",
            "rating_day_of_week",
            "rating_hour",
            "movie_age_at_rating",
            "user_smoothed_mean",
            "movie_smoothed_mean",
            "user_rating_count_log",
            "movie_rating_count_log",
            "user_movie_mean_difference",
        ]
        categorical = [f"gender={value}" for value in self.gender_categories_]
        categorical += [f"occupation={value}" for value in self.occupation_categories_]
        categorical += [f"genre={value}" for value in self.genre_categories_]
        return [*base, *categorical]

    def _build_features(
        self,
        X: pd.DataFrame,
        y_leave_one_out: np.ndarray | None,
    ) -> np.ndarray:
        self._validate_input(X)
        rating_datetime = pd.to_datetime(X["timestamp"], unit="s", utc=True, errors="coerce")
        if rating_datetime.isna().any():
            raise ValueError("存在无法解析的评分时间戳")
        release_year = pd.to_datetime(
            X["release_date"], format="%d-%b-%Y", errors="coerce"
        ).dt.year.astype(float)
        release_year = release_year.fillna(self.release_year_median_)
        rating_year = rating_datetime.dt.year.to_numpy(dtype=float)
        movie_age = np.maximum(rating_year - release_year.to_numpy(dtype=float), 0.0)

        user_mean, movie_mean, user_count, movie_count = self._target_statistics(
            X, y_leave_one_out
        )
        numeric_columns = [
            X["age"].to_numpy(dtype=float),
            release_year.to_numpy(dtype=float),
            rating_year,
            rating_datetime.dt.month.to_numpy(dtype=float),
            rating_datetime.dt.dayofweek.to_numpy(dtype=float),
            rating_datetime.dt.hour.to_numpy(dtype=float),
            movie_age,
            user_mean,
            movie_mean,
            np.log1p(np.maximum(user_count, 0.0)),
            np.log1p(np.maximum(movie_count, 0.0)),
            user_mean - movie_mean,
        ]

        gender_values = X["gender"].fillna("").astype(str).to_numpy()
        occupation_values = X["occupation"].fillna("").astype(str).to_numpy()
        categorical_columns = [
            (gender_values == category).astype(float) for category in self.gender_categories_
        ]
        categorical_columns += [
            (occupation_values == category).astype(float)
            for category in self.occupation_categories_
        ]
        genre_frame = X["genres"].fillna("").astype(str).str.get_dummies(sep="|")
        genre_frame = genre_frame.reindex(columns=self.genre_categories_, fill_value=0)
        categorical_columns += [
            genre_frame[column].to_numpy(dtype=float) for column in self.genre_categories_
        ]

        matrix = np.column_stack([*numeric_columns, *categorical_columns]).astype(
            np.float32, copy=False
        )
        if matrix.shape[1] != len(self.feature_names_out_):
            raise RuntimeError("特征矩阵列数与特征名称数量不一致")
        if not np.isfinite(matrix).all():
            raise ValueError("特征矩阵包含NaN或无穷大")
        return matrix

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """使用当前训练部分学到的映射转换验证集或测试集。"""
        check_is_fitted(self, "feature_names_out_")
        return self._build_features(X, y_leave_one_out=None)

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params: Any) -> np.ndarray:
        """拟合后用逐行留一统计转换训练数据，避免目标泄漏。"""
        del fit_params
        target = self._as_target(y, len(X))
        self.fit(X, target)
        return self._build_features(X, y_leave_one_out=target)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """返回稳定的输出特征名称。"""
        del input_features
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def get_feature_groups(self) -> dict[str, str]:
        """返回特征名称到论文展示分组的映射。"""
        names = self.get_feature_names_out()
        groups: dict[str, str] = {}
        statistical = {
            "user_smoothed_mean",
            "movie_smoothed_mean",
            "user_rating_count_log",
            "movie_rating_count_log",
            "user_movie_mean_difference",
        }
        time_features = {
            "release_year",
            "rating_year",
            "rating_month",
            "rating_day_of_week",
            "rating_hour",
            "movie_age_at_rating",
        }
        for name in names:
            value = str(name)
            if value in statistical:
                groups[value] = "训练集统计"
            elif value in time_features:
                groups[value] = "时间"
            elif value.startswith("genre="):
                groups[value] = "电影类型"
            elif value.startswith("gender=") or value.startswith("occupation=") or value == "user_age":
                groups[value] = "用户属性"
            else:
                groups[value] = "其他"
        return groups
