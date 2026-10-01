"""第5阶段模型对比、Bootstrap、分组误差、重要性和效率工具。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from .neural_data import MovieRatingDataset


MODEL_NAMES = ["global_mean", "random_forest", "adaboost", "xgboost", "mlp"]
PREDICTION_COLUMNS = {name: f"{name}_prediction" for name in MODEL_NAMES}


def _validate_prediction_arrays(actual: np.ndarray, predicted: np.ndarray) -> None:
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("真实评分与预测评分形状不一致或为空")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("真实评分或预测评分包含NaN/无穷值")


def calculate_regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    """统一计算回归指标和基本预测统计。"""
    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)
    _validate_prediction_arrays(y_true, y_pred)
    error = y_pred - y_true
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_error": float(error.mean()),
        "error_std": float(error.std(ddof=0)),
        "prediction_min": float(y_pred.min()),
        "prediction_max": float(y_pred.max()),
    }


def _entity_group(counts: pd.Series, low_threshold: float, high_threshold: float) -> pd.Series:
    return pd.Series(
        np.select(
            [counts <= low_threshold, counts <= high_threshold],
            ["low", "medium"],
            default="high",
        ),
        index=counts.index,
        dtype="string",
    )


def build_unified_predictions(
    test: pd.DataFrame,
    train: pd.DataFrame,
    stage2_predictions: pd.DataFrame,
    stage3_predictions: pd.DataFrame,
    stage4_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按interaction_id一对一合并三阶段预测，并添加训练集来源的分组。"""
    required_test = {"interaction_id", "user_id", "movie_id", "rating"}
    if not required_test.issubset(test.columns):
        raise ValueError(f"固定测试集缺少字段：{required_test - set(test.columns)}")
    if not test["interaction_id"].is_unique:
        raise ValueError("固定测试集interaction_id不唯一")

    base = test[["interaction_id", "user_id", "movie_id", "rating"]].copy()
    base = base.rename(columns={"rating": "actual_rating"})
    sources = [
        (stage2_predictions, "rating", "predicted_rating", "global_mean_prediction"),
        (stage3_predictions, "rating", None, None),
        (stage4_predictions, "actual_rating", "predicted_rating", "mlp_prediction"),
    ]
    for frame, actual_column, prediction_column, output_column in sources:
        if not frame["interaction_id"].is_unique:
            raise ValueError("阶段预测文件存在重复interaction_id")
        check = frame[["interaction_id", actual_column]].rename(columns={actual_column: "source_actual"})
        check = base[["interaction_id", "actual_rating"]].merge(
            check, on="interaction_id", how="left", validate="one_to_one"
        )
        if check["source_actual"].isna().any() or not np.allclose(
            check["actual_rating"], check["source_actual"], rtol=0, atol=1e-12
        ):
            raise ValueError("不同阶段预测文件的真实评分不一致")
        if prediction_column:
            piece = frame[["interaction_id", prediction_column]].rename(
                columns={prediction_column: output_column}
            )
            base = base.merge(piece, on="interaction_id", how="left", validate="one_to_one")

    stage3_columns = [
        "interaction_id", "random_forest_prediction", "adaboost_prediction", "xgboost_prediction"
    ]
    if not set(stage3_columns).issubset(stage3_predictions.columns):
        raise ValueError("第3阶段预测列不完整")
    base = base.merge(
        stage3_predictions[stage3_columns], on="interaction_id", how="left", validate="one_to_one"
    )
    if len(base) != len(test) or base.isna().any().any():
        raise ValueError("统一预测合并后行数变化或出现缺失值")

    prediction_columns = list(PREDICTION_COLUMNS.values())
    values = base[prediction_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 1.0) & (values <= 5.0)).all():
        raise ValueError("统一预测包含非法值或超出1至5")

    user_counts = train.groupby("user_id").size().astype(int)
    movie_counts = train.groupby("movie_id").size().astype(int)
    user_q1, user_q2 = user_counts.quantile([1 / 3, 2 / 3]).tolist()
    movie_q1, movie_q2 = movie_counts.quantile([1 / 3, 2 / 3]).tolist()
    base["user_train_rating_count"] = base["user_id"].map(user_counts).fillna(0).astype(int)
    base["movie_train_rating_count"] = base["movie_id"].map(movie_counts).fillna(0).astype(int)
    base["user_seen_in_train"] = base["user_train_rating_count"] > 0
    base["movie_seen_in_train"] = base["movie_train_rating_count"] > 0
    base["cold_start_group"] = np.select(
        [
            base["user_seen_in_train"] & base["movie_seen_in_train"],
            ~base["user_seen_in_train"] & base["movie_seen_in_train"],
            base["user_seen_in_train"] & ~base["movie_seen_in_train"],
        ],
        ["both_seen", "unseen_user_only", "unseen_movie_only"],
        default="both_unseen",
    )
    base["user_activity_group"] = _entity_group(
        base["user_train_rating_count"], user_q1, user_q2
    )
    movie_groups = _entity_group(base["movie_train_rating_count"], movie_q1, movie_q2)
    movie_groups.loc[base["movie_train_rating_count"] == 0] = "unseen"
    base["movie_popularity_group"] = movie_groups

    metadata = {
        "source": "仅使用第2阶段80000条训练数据的实体评分次数",
        "user_activity_thresholds": {"low_max": float(user_q1), "medium_max": float(user_q2)},
        "movie_popularity_thresholds": {"low_max": float(movie_q1), "medium_max": float(movie_q2)},
        "unseen_test_user_count": int((~base["user_seen_in_train"]).sum()),
        "unseen_test_movie_count": int(base.loc[~base["movie_seen_in_train"], "movie_id"].nunique()),
        "unseen_movie_test_rows": int((~base["movie_seen_in_train"]).sum()),
    }
    return base, metadata


def build_metrics_comparison(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual = unified["actual_rating"].to_numpy(dtype=float)
    for model in MODEL_NAMES:
        rows.append({"model": model, **calculate_regression_metrics(actual, unified[PREDICTION_COLUMNS[model]])})
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"] == "global_mean"].iloc[0]
    result["rmse_improvement_vs_baseline_percent"] = (
        (baseline["rmse"] - result["rmse"]) / baseline["rmse"] * 100.0
    )
    result["mae_improvement_vs_baseline_percent"] = (
        (baseline["mae"] - result["mae"]) / baseline["mae"] * 100.0
    )
    result["rmse_rank"] = result["rmse"].rank(method="min", ascending=True).astype(int)
    result["mae_rank"] = result["mae"].rank(method="min", ascending=True).astype(int)
    result["r2_rank"] = result["r2"].rank(method="min", ascending=False).astype(int)
    return result.sort_values("rmse_rank").reset_index(drop=True)


def paired_bootstrap_compare(
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    ensemble_models: list[str],
    resamples: int,
    confidence_level: float,
    random_seed: int,
) -> pd.DataFrame:
    """对相同测试记录进行成对有放回重采样，差异定义为MLP减集成模型。"""
    y = np.asarray(actual, dtype=float)
    squared_errors = {name: (np.asarray(predictions[name]) - y) ** 2 for name in predictions}
    absolute_errors = {name: np.abs(np.asarray(predictions[name]) - y) for name in predictions}
    rng = np.random.default_rng(random_seed)
    delta_store = {(model, metric): np.empty(resamples) for model in ensemble_models for metric in ["rmse", "mae"]}
    for repeat in range(resamples):
        indices = rng.integers(0, len(y), size=len(y))
        mlp_rmse = float(np.sqrt(squared_errors["mlp"][indices].mean()))
        mlp_mae = float(absolute_errors["mlp"][indices].mean())
        for model in ensemble_models:
            delta_store[(model, "rmse")][repeat] = mlp_rmse - float(
                np.sqrt(squared_errors[model][indices].mean())
            )
            delta_store[(model, "mae")][repeat] = mlp_mae - float(
                absolute_errors[model][indices].mean()
            )
    alpha = (1.0 - confidence_level) / 2.0
    rows = []
    for model in ensemble_models:
        point = {
            "rmse": float(np.sqrt(squared_errors["mlp"].mean()) - np.sqrt(squared_errors[model].mean())),
            "mae": float(absolute_errors["mlp"].mean() - absolute_errors[model].mean()),
        }
        for metric in ["rmse", "mae"]:
            values = delta_store[(model, metric)]
            rows.append({
                "comparison": f"mlp_vs_{model}",
                "ensemble_model": model,
                "metric": metric,
                "delta_definition": "mlp_metric_minus_ensemble_metric",
                "point_estimate_delta": point[metric],
                "bootstrap_mean_delta": float(values.mean()),
                "confidence_lower": float(np.quantile(values, alpha)),
                "confidence_upper": float(np.quantile(values, 1.0 - alpha)),
                "mlp_win_probability": float((values < 0).mean()),
                "resamples": int(resamples),
                "confidence_level": float(confidence_level),
                "random_seed": int(random_seed),
            })
    return pd.DataFrame(rows)


def _safe_group_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    result = calculate_regression_metrics(actual, predicted)
    if len(actual) < 2 or np.var(actual) == 0:
        result["r2"] = np.nan
    return result


def calculate_segment_metrics(unified: pd.DataFrame) -> pd.DataFrame:
    segment_columns = {
        "actual_rating": "actual_rating",
        "cold_start": "cold_start_group",
        "user_activity": "user_activity_group",
        "movie_popularity": "movie_popularity_group",
    }
    rows = []
    for segment_type, column in segment_columns.items():
        for segment_value, group in unified.groupby(column, observed=True, sort=True):
            actual = group["actual_rating"].to_numpy(dtype=float)
            for model in MODEL_NAMES:
                result = _safe_group_metrics(actual, group[PREDICTION_COLUMNS[model]].to_numpy(dtype=float))
                rows.append({
                    "segment_type": segment_type,
                    "segment_value": str(segment_value),
                    "model": model,
                    "sample_count": len(group),
                    "rmse": result["rmse"],
                    "mae": result["mae"],
                    "r2": result["r2"],
                    "mean_error": result["mean_error"],
                    "r2_note": "样本不足或真实评分无方差" if pd.isna(result["r2"]) else "",
                })
    return pd.DataFrame(rows)


def calculate_residual_summary(unified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = unified["actual_rating"].to_numpy(dtype=float)
    summary_rows = []
    rating_rows = []
    for model in MODEL_NAMES:
        predicted = unified[PREDICTION_COLUMNS[model]].to_numpy(dtype=float)
        error = predicted - actual
        absolute = np.abs(error)
        summary_rows.append({
            "model": model,
            "mean_error": float(error.mean()),
            "error_std": float(error.std(ddof=0)),
            "median_absolute_error": float(np.median(absolute)),
            "p90_absolute_error": float(np.quantile(absolute, 0.90)),
            "p95_absolute_error": float(np.quantile(absolute, 0.95)),
            "over_prediction_rate": float((error > 0).mean()),
            "under_prediction_rate": float((error < 0).mean()),
            "exact_or_near_rate": float((absolute <= 0.5).mean()),
            "prediction_min": float(predicted.min()),
            "prediction_max": float(predicted.max()),
        })
        for rating in range(1, 6):
            mask = actual == rating
            group_pred = predicted[mask]
            group_actual = actual[mask]
            group_error = group_pred - group_actual
            rating_rows.append({
                "actual_rating": rating,
                "model": model,
                "sample_count": int(mask.sum()),
                "mean_prediction": float(group_pred.mean()),
                "mean_error": float(group_error.mean()),
                "rmse": float(np.sqrt(np.mean(group_error ** 2))),
                "mae": float(np.mean(np.abs(group_error))),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(rating_rows)


def aggregate_ensemble_importance(
    importance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"model", "feature", "feature_group", "importance"}
    if not required.issubset(importance.columns):
        raise ValueError(f"集成重要性缺少字段：{required - set(importance.columns)}")
    detailed = importance[["model", "feature", "feature_group", "importance"]].copy()
    totals = detailed.groupby("model")["importance"].transform("sum")
    if (totals <= 0).any():
        raise ValueError("集成模型重要性总和不是正数")
    detailed["normalized_importance"] = detailed["importance"] / totals
    detailed["rank"] = detailed.groupby("model")["importance"].rank(
        method="first", ascending=False
    ).astype(int)
    detailed = detailed.sort_values(["model", "rank"]).reset_index(drop=True)
    grouped = detailed.groupby(["model", "feature_group"], as_index=False)["normalized_importance"].sum()
    grouped = grouped.rename(columns={"normalized_importance": "importance"})
    grouped["rank"] = grouped.groupby("model")["importance"].rank(
        method="first", ascending=False
    ).astype(int)
    return detailed, grouped.sort_values(["model", "rank"]).reset_index(drop=True)


def _predict_transformed(
    model: torch.nn.Module,
    features: dict[str, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(MovieRatingDataset(features), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for batch in loader:
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)


def calculate_mlp_permutation_importance(
    model: torch.nn.Module,
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    repeats: int,
    random_seed: int,
    batch_size: int = 1024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在验证集张量上逐组排列，模型仅推理，不进行梯度或参数更新。"""
    actual = np.asarray(labels, dtype=float)
    baseline_prediction = _predict_transformed(model, features, batch_size)
    baseline_rmse = float(np.sqrt(mean_squared_error(actual, baseline_prediction)))
    group_specs: dict[str, tuple[str, list[int] | None]] = {
        "user_embedding": ("user_index", None),
        "movie_embedding": ("movie_index", None),
        "gender_embedding": ("gender_index", None),
        "occupation_embedding": ("occupation_index", None),
        "age": ("numeric", [0]),
        "release_year": ("numeric", [1]),
        "rating_year": ("numeric", [2]),
        "movie_age_at_rating": ("numeric", [3]),
        "rating_month_cycle": ("numeric", [4, 5]),
        "rating_day_of_week_cycle": ("numeric", [6, 7]),
        "rating_hour_cycle": ("numeric", [8, 9]),
        "genres": ("genres", list(range(features["genres"].shape[1]))),
    }
    rng = np.random.default_rng(random_seed)
    rows = []
    for group_name, (array_name, columns) in group_specs.items():
        for repeat in range(1, repeats + 1):
            permutation = rng.permutation(len(actual))
            permuted = {name: values for name, values in features.items()}
            changed = features[array_name].copy()
            if columns is None:
                changed = changed[permutation]
            else:
                changed[:, columns] = changed[permutation][:, columns]
            permuted[array_name] = changed
            prediction = _predict_transformed(model, permuted, batch_size)
            permuted_rmse = float(np.sqrt(mean_squared_error(actual, prediction)))
            rows.append({
                "feature_group": group_name,
                "repeat": repeat,
                "permuted_rmse": permuted_rmse,
                "rmse_increase": permuted_rmse - baseline_rmse,
                "random_seed": random_seed,
                "evaluation_rows": len(actual),
            })
    repeats_frame = pd.DataFrame(rows)
    summary = repeats_frame.groupby("feature_group", as_index=False)["rmse_increase"].agg(
        rmse_increase_mean="mean",
        rmse_increase_std=lambda values: values.std(ddof=1),
        rmse_increase_min="min",
        rmse_increase_max="max",
    )
    summary.insert(1, "baseline_rmse", baseline_rmse)
    positive = summary["rmse_increase_mean"].clip(lower=0)
    positive_sum = float(positive.sum())
    summary["positive_normalized_importance"] = positive / positive_sum if positive_sum > 0 else 0.0
    summary["rank"] = summary["rmse_increase_mean"].rank(method="first", ascending=False).astype(int)
    return repeats_frame, summary.sort_values("rank").reset_index(drop=True)


def benchmark_model_inference(
    model_name: str,
    artifact_paths: list[Path],
    load_callable: Callable[[], Any],
    predict_callable: Callable[[Any], np.ndarray],
    row_count: int,
    warmup_runs: int,
    repeats: int,
) -> dict[str, Any]:
    """分别测量加载和端到端推理，返回同一机器上的相对效率指标。"""
    load_times = []
    loaded = None
    for _ in range(repeats):
        start = time.perf_counter()
        loaded = load_callable()
        load_times.append(time.perf_counter() - start)
    if loaded is None:
        raise RuntimeError(f"{model_name}加载失败")
    for _ in range(warmup_runs):
        warm_prediction = np.asarray(predict_callable(loaded))
        if len(warm_prediction) != row_count:
            raise ValueError(f"{model_name}预热预测数量错误")
    inference_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        prediction = np.asarray(predict_callable(loaded))
        inference_times.append(time.perf_counter() - start)
        if len(prediction) != row_count or not np.isfinite(prediction).all():
            raise ValueError(f"{model_name}推理输出异常")
    inference_mean = float(np.mean(inference_times))
    size_bytes = int(sum(Path(path).stat().st_size for path in artifact_paths))
    return {
        "model": model_name,
        "artifact_size_bytes": size_bytes,
        "artifact_size_mb": size_bytes / (1024 ** 2),
        "load_time_mean_seconds": float(np.mean(load_times)),
        "load_time_std_seconds": float(np.std(load_times, ddof=1)) if repeats > 1 else 0.0,
        "inference_time_mean_seconds": inference_mean,
        "inference_time_std_seconds": float(np.std(inference_times, ddof=1)) if repeats > 1 else 0.0,
        "milliseconds_per_1000_rows": inference_mean * 1000.0 / row_count * 1000.0,
        "rows_per_second": row_count / inference_mean,
        "warmup_runs": warmup_runs,
        "benchmark_repeats": repeats,
        "device": "cpu",
    }
