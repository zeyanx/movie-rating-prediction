"""第3阶段集成回归模型、交叉验证和重要性工具。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .baseline import evaluate_predictions
from .features import MovieFeatureEngineer


class ClippedRegressor(BaseEstimator, RegressorMixin):
    """包装任意回归器，使交叉验证和最终预测统一裁剪至合法评分范围。"""

    def __init__(
        self,
        estimator: Any,
        rating_min: float = 1.0,
        rating_max: float = 5.0,
    ) -> None:
        self.estimator = estimator
        self.rating_min = rating_min
        self.rating_max = rating_max

    def fit(self, X: np.ndarray, y: Any) -> "ClippedRegressor":
        if self.rating_min >= self.rating_max:
            raise ValueError("rating_min必须小于rating_max")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("ClippedRegressor尚未拟合")
        prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        return np.clip(prediction, self.rating_min, self.rating_max)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("ClippedRegressor尚未拟合")
        if not hasattr(self.estimator_, "feature_importances_"):
            raise AttributeError("基础回归器不提供feature_importances_")
        return np.asarray(self.estimator_.feature_importances_, dtype=float)


def build_model_pipelines(config: dict[str, Any]) -> dict[str, Pipeline]:
    """根据固定配置构造共享同类特征工程的三个独立Pipeline。"""
    feature_config = config["feature_engineering"]
    model_config = config["models"]

    random_forest = RandomForestRegressor(**model_config["random_forest"])
    ada_config = dict(model_config["adaboost"])
    tree_config = ada_config.pop("estimator")
    adaboost = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(**tree_config),
        **ada_config,
    )
    xgboost = XGBRegressor(**model_config["xgboost"], verbosity=0)
    estimators = {
        "random_forest": random_forest,
        "adaboost": adaboost,
        "xgboost": xgboost,
    }

    pipelines: dict[str, Pipeline] = {}
    for name, estimator in estimators.items():
        pipelines[name] = Pipeline(
            steps=[
                ("features", MovieFeatureEngineer(**feature_config)),
                ("model", ClippedRegressor(estimator=estimator)),
            ]
        )
    return pipelines


def evaluate_cv(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv_config: dict[str, Any],
) -> pd.DataFrame:
    """依次执行模型和折，避免嵌套并行并保留每折明细。"""
    splitter = KFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"],
    )
    rows: list[dict[str, Any]] = []
    for model_name, pipeline in pipelines.items():
        print(f"\n开始5折交叉验证：{model_name}")
        for fold, (train_index, validation_index) in enumerate(splitter.split(X), start=1):
            estimator = clone(pipeline)
            fit_start = time.perf_counter()
            estimator.fit(X.iloc[train_index], y.iloc[train_index])
            fit_time = time.perf_counter() - fit_start
            score_start = time.perf_counter()
            prediction = estimator.predict(X.iloc[validation_index])
            score_time = time.perf_counter() - score_start
            metrics = evaluate_predictions(y.iloc[validation_index], prediction)
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_rows": len(train_index),
                    "validation_rows": len(validation_index),
                    **metrics,
                    "fit_time_seconds": fit_time,
                    "score_time_seconds": score_time,
                }
            )
            print(
                f"  第{fold}/5折：RMSE={metrics['rmse']:.6f}，"
                f"MAE={metrics['mae']:.6f}，R²={metrics['r2']:.6f}，"
                f"训练={fit_time:.2f}s"
            )
    return pd.DataFrame(rows)


def summarize_cv(fold_results: pd.DataFrame) -> pd.DataFrame:
    """汇总五折均值、样本标准差和按CV RMSE确定的名次。"""
    summary = (
        fold_results.groupby("model", sort=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            fit_time_mean=("fit_time_seconds", "mean"),
            score_time_mean=("score_time_seconds", "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values(
        ["rmse_mean", "mae_mean", "fit_time_mean"], kind="stable"
    ).reset_index(drop=True)
    summary["rank_by_cv_rmse"] = np.arange(1, len(summary) + 1)
    return summary


def fit_final_models(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, Pipeline], dict[str, float]]:
    """在完整固定训练集上拟合三个最终模型。"""
    fitted: dict[str, Pipeline] = {}
    durations: dict[str, float] = {}
    for model_name, pipeline in pipelines.items():
        print(f"拟合完整训练集：{model_name}")
        start = time.perf_counter()
        model = clone(pipeline).fit(X_train, y_train)
        durations[model_name] = time.perf_counter() - start
        fitted[model_name] = model
    return fitted, durations


def extract_feature_importance(
    model_name: str,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """提取并归一化一个最终Pipeline的原生特征重要性。"""
    transformer = pipeline.named_steps["features"]
    regressor = pipeline.named_steps["model"]
    names = transformer.get_feature_names_out()
    groups = transformer.get_feature_groups()
    importance = regressor.feature_importances_
    if len(names) != len(importance):
        raise ValueError(f"{model_name}特征名称与重要性数量不一致")
    total = float(importance.sum())
    if not np.isfinite(importance).all() or total <= 0:
        raise ValueError(f"{model_name}特征重要性无效")
    frame = pd.DataFrame(
        {
            "model": model_name,
            "feature": names.astype(str),
            "feature_group": [groups[str(name)] for name in names],
            "importance": importance,
            "normalized_importance": importance / total,
        }
    )
    frame = frame.sort_values("normalized_importance", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame
