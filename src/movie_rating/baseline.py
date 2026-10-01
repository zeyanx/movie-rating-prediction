"""只依赖训练集评分的全局均值基线模型。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


class GlobalMeanBaseline:
    """使用训练集全局评分均值预测所有用户—电影组合。"""

    def __init__(
        self,
        rating_min: float = 1.0,
        rating_max: float = 5.0,
        random_state: int = 42,
    ) -> None:
        if rating_min >= rating_max:
            raise ValueError("rating_min必须小于rating_max")
        self.model_name = "global_mean_baseline"
        self.rating_min = float(rating_min)
        self.rating_max = float(rating_max)
        self.random_state = int(random_state)
        self.global_mean_: float | None = None
        self.training_samples_: int | None = None
        self.trained_at_: str | None = None

    def fit(self, y_train: pd.Series | np.ndarray | list[float]) -> "GlobalMeanBaseline":
        """只根据训练集评分拟合全局均值。"""
        values = np.asarray(y_train, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("训练评分不能为空")
        if not np.isfinite(values).all():
            raise ValueError("训练评分包含NaN或无穷大")
        if ((values < self.rating_min) | (values > self.rating_max)).any():
            raise ValueError("训练评分超出合法范围")

        self.global_mean_ = float(values.mean())
        self.training_samples_ = int(values.size)
        self.trained_at_ = datetime.now(timezone.utc).isoformat()
        return self

    def _check_fitted(self) -> None:
        if self.global_mean_ is None or self.training_samples_ is None:
            raise RuntimeError("GlobalMeanBaseline尚未拟合")

    def predict(self, n_samples: int) -> np.ndarray:
        """返回指定长度的一维常数预测，并裁剪至合法评分范围。"""
        self._check_fitted()
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 0:
            raise ValueError("n_samples必须是非负整数")
        prediction = float(np.clip(self.global_mean_, self.rating_min, self.rating_max))
        return np.full(int(n_samples), prediction, dtype=float)

    def predict_dataframe(self, dataframe: pd.DataFrame) -> np.ndarray:
        """根据数据表行数生成预测，便于后续统一模型接口。"""
        return self.predict(len(dataframe))

    def metadata(self) -> dict[str, Any]:
        """返回可用于展示和审计的训练元数据。"""
        self._check_fitted()
        return {
            "model_name": self.model_name,
            "global_mean": self.global_mean_,
            "rating_min": self.rating_min,
            "rating_max": self.rating_max,
            "training_samples": self.training_samples_,
            "random_state": self.random_state,
            "trained_at": self.trained_at_,
        }

    def save(self, path: Path) -> None:
        """使用joblib保存完整模型对象。"""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "GlobalMeanBaseline":
        """从joblib文件加载模型并检查对象类型和拟合状态。"""
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError(f"模型文件类型错误：{type(model)!r}")
        model._check_fitted()
        return model


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """使用scikit-learn计算RMSE、MAE和R²。"""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("真实值与预测值长度不一致")
    if actual.size == 0:
        raise ValueError("评估数据不能为空")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("评估数据包含NaN或无穷大")
    return {
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
