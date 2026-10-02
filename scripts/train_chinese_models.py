"""训练中国电影评分子集上的随机森林、XGBoost和PyTorch MLP。"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.chinese_models import (  # noqa: E402
    ChineseRatingMLP,
    build_tree_features,
    build_numeric_features,
    standardize_numeric,
)


DATA_DIR = ROOT / "data" / "chinese_training"
MODEL_DIR = ROOT / "models" / "chinese"
REPORT_DIR = ROOT / "reports" / "chinese_training"
SEED = 42
FEATURE_COLUMNS = [
    "cn_user_id", "cn_movie_index", "rating_year", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos", "user_mean", "user_log_count", "movie_mean",
    "movie_log_count",
]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def temporal_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """每位用户最后一次评分作测试、倒数第二次作验证，避免随机划分泄漏未来。"""
    ordered = ratings.sort_values(["cn_user_id", "timestamp", "cn_movie_index"]).copy()
    reverse_rank = ordered.groupby("cn_user_id").cumcount(ascending=False)
    test = ordered[reverse_rank == 0].copy()
    validation = ordered[reverse_rank == 1].copy()
    train = ordered[reverse_rank >= 2].copy()
    if train["cn_user_id"].nunique() != ratings["cn_user_id"].nunique():
        raise ValueError("时间划分后训练集缺少用户；请提高最小用户评分数。")
    return train, validation, test


def fit_preprocessing(frame: pd.DataFrame, num_users: int, num_movies: int) -> dict[str, Any]:
    global_mean = float(frame["rating"].mean())
    user_stats = frame.groupby("cn_user_id")["rating"].agg(["mean", "count"])
    movie_stats = frame.groupby("cn_movie_index")["rating"].agg(["mean", "count"])
    user_mean = np.full(num_users + 1, global_mean, dtype=np.float32)
    user_count = np.zeros(num_users + 1, dtype=np.float32)
    movie_mean = np.full(num_movies + 1, global_mean, dtype=np.float32)
    movie_count = np.zeros(num_movies + 1, dtype=np.float32)
    user_indices = user_stats.index.to_numpy(dtype=int)
    movie_indices = movie_stats.index.to_numpy(dtype=int)
    user_mean[user_indices] = user_stats["mean"].to_numpy(dtype=np.float32)
    user_count[user_indices] = user_stats["count"].to_numpy(dtype=np.float32)
    movie_mean[movie_indices] = movie_stats["mean"].to_numpy(dtype=np.float32)
    movie_count[movie_indices] = movie_stats["count"].to_numpy(dtype=np.float32)
    preliminary = {
        "global_mean": global_mean,
        "user_mean": user_mean,
        "user_count": user_count,
        "movie_mean": movie_mean,
        "movie_count": movie_count,
    }
    numeric = build_numeric_features(
        frame["cn_user_id"].to_numpy(), frame["cn_movie_index"].to_numpy(),
        frame["timestamp"].to_numpy(), preliminary,
    )
    scale = numeric.std(axis=0)
    scale[scale < 1e-6] = 1.0
    preliminary["numeric_mean"] = numeric.mean(axis=0).astype(np.float32)
    preliminary["numeric_scale"] = scale.astype(np.float32)
    return preliminary


def tree_features(frame: pd.DataFrame, preprocessing: dict[str, Any]) -> np.ndarray:
    return build_tree_features(
        frame["cn_user_id"].to_numpy(), frame["cn_movie_index"].to_numpy(),
        frame["timestamp"].to_numpy(), preprocessing,
    )


def tensor_loader(
    frame: pd.DataFrame,
    preprocessing: dict[str, Any],
    shuffle: bool,
    batch_size: int = 1024,
) -> DataLoader:
    numeric = build_numeric_features(
        frame["cn_user_id"].to_numpy(), frame["cn_movie_index"].to_numpy(),
        frame["timestamp"].to_numpy(), preprocessing,
    )
    numeric = standardize_numeric(numeric, preprocessing)
    dataset = TensorDataset(
        torch.as_tensor(frame["cn_user_id"].to_numpy(), dtype=torch.long),
        torch.as_tensor(frame["cn_movie_index"].to_numpy(), dtype=torch.long),
        torch.as_tensor(numeric, dtype=torch.float32),
        torch.as_tensor(frame["rating"].to_numpy(), dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def predict_mlp(model: ChineseRatingMLP, loader: DataLoader) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for users, movies, numeric, _ in loader:
            output.append(model(users, movies, numeric).cpu().numpy())
    return np.concatenate(output)


def select_epoch(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    preprocessing: dict[str, Any],
    num_users: int,
    num_movies: int,
) -> tuple[int, list[dict[str, float]]]:
    set_seed()
    model = ChineseRatingMLP(num_users, num_movies, numeric_features=9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    train_loader = tensor_loader(train, preprocessing, shuffle=True)
    validation_loader = tensor_loader(validation, preprocessing, shuffle=False)
    best_rmse = math.inf
    best_epoch = 1
    patience = 0
    history = []
    for epoch in range(1, 26):
        model.train()
        losses = []
        for users, movies, numeric, target in train_loader:
            optimizer.zero_grad()
            prediction = model(users, movies, numeric)
            loss = loss_fn(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        prediction = predict_mlp(model, validation_loader)
        score = metrics(validation["rating"].to_numpy(), prediction)
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), **score})
        print(f"MLP选择阶段 epoch={epoch:02d} val_RMSE={score['rmse']:.5f}")
        if score["rmse"] < best_rmse - 1e-5:
            best_rmse = score["rmse"]
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
    return best_epoch, history


def train_final_mlp(
    frame: pd.DataFrame,
    preprocessing: dict[str, Any],
    num_users: int,
    num_movies: int,
    epochs: int,
) -> ChineseRatingMLP:
    set_seed()
    model = ChineseRatingMLP(num_users, num_movies, numeric_features=9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    loader = tensor_loader(frame, preprocessing, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for users, movies, numeric, target in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(users, movies, numeric), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"MLP最终训练 epoch={epoch:02d}/{epochs:02d} MSE={np.mean(losses):.5f}")
    return model


def serializable_preprocessing(preprocessing: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in preprocessing.items()
    }


def main() -> int:
    set_seed()
    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"
    if not ratings_path.is_file() or not movies_path.is_file():
        print("请先运行 scripts/build_chinese_ratings_dataset.py", file=sys.stderr)
        return 1
    ratings = pd.read_csv(ratings_path, encoding="utf-8-sig")
    movies = pd.read_csv(movies_path, encoding="utf-8-sig")
    num_users = int(ratings["cn_user_id"].max())
    num_movies = int(ratings["cn_movie_index"].max())
    train, validation, test = temporal_split(ratings)
    train_validation = pd.concat([train, validation], ignore_index=True)

    inner_preprocessing = fit_preprocessing(train, num_users, num_movies)
    best_epoch, history = select_epoch(
        train, validation, inner_preprocessing, num_users, num_movies
    )
    final_preprocessing = fit_preprocessing(train_validation, num_users, num_movies)

    x_train = tree_features(train_validation, final_preprocessing)
    y_train = train_validation["rating"].to_numpy(dtype=np.float32)
    x_test = tree_features(test, final_preprocessing)
    y_test = test["rating"].to_numpy(dtype=np.float32)

    start = time.perf_counter()
    random_forest = RandomForestRegressor(
        n_estimators=100, max_depth=14, min_samples_leaf=5, max_features=0.8,
        n_jobs=-1, random_state=SEED,
    ).fit(x_train, y_train)
    rf_seconds = time.perf_counter() - start
    rf_prediction = np.clip(random_forest.predict(x_test), 0.5, 5.0)

    start = time.perf_counter()
    xgboost = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, objective="reg:squarederror",
        n_jobs=-1, random_state=SEED,
    ).fit(x_train, y_train)
    xgb_seconds = time.perf_counter() - start
    xgb_prediction = np.clip(xgboost.predict(x_test), 0.5, 5.0)

    start = time.perf_counter()
    mlp = train_final_mlp(
        train_validation, final_preprocessing, num_users, num_movies, best_epoch
    )
    mlp_seconds = time.perf_counter() - start
    mlp_prediction = predict_mlp(mlp, tensor_loader(test, final_preprocessing, False))
    baseline_prediction = np.full(len(test), final_preprocessing["global_mean"])

    results = []
    for model_name, prediction, seconds in [
        ("global_mean", baseline_prediction, 0.0),
        ("random_forest", rf_prediction, rf_seconds),
        ("xgboost", xgb_prediction, xgb_seconds),
        ("mlp", mlp_prediction, mlp_seconds),
    ]:
        results.append({"model": model_name, **metrics(y_test, prediction),
                        "training_seconds": seconds})
    metrics_frame = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    metrics_frame.insert(1, "rank", np.arange(1, len(metrics_frame) + 1))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_common = {
        "feature_columns": FEATURE_COLUMNS,
        "preprocessing": serializable_preprocessing(final_preprocessing),
    }
    joblib.dump({"model": random_forest, **bundle_common}, MODEL_DIR / "random_forest.joblib")
    joblib.dump({"model": xgboost, **bundle_common}, MODEL_DIR / "xgboost.joblib")
    torch.save(mlp.state_dict(), MODEL_DIR / "mlp.pt")
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "split": "per-user temporal: last=test, second-last=validation",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "final_train_rows": len(train_validation),
        "test_rows": len(test),
        "num_users": num_users,
        "num_movies": num_movies,
        "selected_epochs": best_epoch,
        "embedding_dim": 24,
        "numeric_features": 9,
        "dropout": 0.12,
        "reference_timestamp": int(train_validation["timestamp"].max()),
        "preprocessing": serializable_preprocessing(final_preprocessing),
        "metrics": metrics_frame.to_dict(orient="records"),
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_frame.to_csv(REPORT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(REPORT_DIR / "mlp_training_history.csv", index=False, encoding="utf-8-sig")
    predictions = test[["cn_user_id", "cn_movie_index", "rating", "timestamp"]].copy()
    predictions["random_forest"] = rf_prediction
    predictions["xgboost"] = xgb_prediction
    predictions["mlp"] = mlp_prediction
    predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    summary = f"""# 中国电影评分扩展实验

- 数据规模：{len(ratings)}条评分、{num_users}名用户、{num_movies}部中国电影。
- 划分：每名用户最后一次评分为测试集、倒数第二次为验证集，其余为训练集。
- 固定测试集：{len(test)}条；MLP通过验证集选择{best_epoch}个epoch，再在训练+验证集重训。
- 最佳测试模型：{metrics_frame.iloc[0]['model']}，RMSE={metrics_frame.iloc[0]['rmse']:.6f}，MAE={metrics_frame.iloc[0]['mae']:.6f}。

该结果是独立的中国电影扩展实验，不替换MovieLens 100K主实验，也不能与主实验指标直接混为同一测试集比较。
"""
    (REPORT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(metrics_frame.to_string(index=False))
    print("中国电影模型训练完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
