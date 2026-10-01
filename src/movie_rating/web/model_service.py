"""统一模型加载与只读推理服务。"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from movie_rating.neural import load_mlp_artifacts, predict_ratings

from .config import PROJECT_ROOT


@st.cache_resource(show_spinner=False)
def load_cached_mlp(root: str = str(PROJECT_ROOT)):
    """全局只读缓存最终MLP、预处理器和元数据。"""
    base = Path(root) / "models" / "stage4"
    required = [base / "mlp_final.pt", base / "final_preprocessor.joblib", base / "model_metadata.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少第4阶段MLP产物：{missing}")
    return load_mlp_artifacts(*required, device="cpu")


@st.cache_resource(show_spinner=False)
def load_cached_ensemble(model_name: str, root: str = str(PROJECT_ROOT)):
    """按需加载集成模型，首页和普通详情不会预加载全部模型。"""
    if model_name not in {"random_forest", "adaboost", "xgboost"}:
        raise ValueError(f"不支持的集成模型：{model_name}")
    path = Path(root) / "models" / "stage3" / f"{model_name}.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"缺少第3阶段模型：{path}")
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_cached_baseline(root: str = str(PROJECT_ROOT)):
    path = Path(root) / "models" / "global_mean_baseline.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"缺少全局均值模型：{path}")
    return joblib.load(path)


def _validate_predictions(values: np.ndarray, expected_rows: int) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if len(result) != expected_rows:
        raise RuntimeError("模型输出行数与输入不一致")
    if not np.isfinite(result).all():
        raise RuntimeError("模型预测包含NaN或无穷值")
    return np.clip(result, 1.0, 5.0)


def predict_with_mlp(frame: pd.DataFrame, root: str = str(PROJECT_ROOT)) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    model, preprocessor, _ = load_cached_mlp(root)
    return _validate_predictions(
        predict_ratings(model, preprocessor, frame, device="cpu", batch_size=1024), len(frame)
    )


def predict_with_ensemble(
    frame: pd.DataFrame, model_name: str, root: str = str(PROJECT_ROOT)
) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    model = load_cached_ensemble(model_name, root)
    return _validate_predictions(model.predict(frame), len(frame))


def predict_with_global_mean(frame: pd.DataFrame, root: str = str(PROJECT_ROOT)) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    model = load_cached_baseline(root)
    return _validate_predictions(model.predict_dataframe(frame), len(frame))


def predict_with_models(
    frame: pd.DataFrame,
    model_names: list[str] | tuple[str, ...],
    root: str = str(PROJECT_ROOT),
) -> pd.DataFrame:
    """以稳定输入顺序返回多模型预测。"""
    result = pd.DataFrame(index=frame.index)
    for name in model_names:
        if name == "mlp":
            prediction = predict_with_mlp(frame, root)
        elif name == "global_mean":
            prediction = predict_with_global_mean(frame, root)
        else:
            prediction = predict_with_ensemble(frame, name, root)
        result[name] = prediction
    return result.reset_index(drop=True)
