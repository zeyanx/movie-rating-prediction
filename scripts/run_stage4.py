"""训练第4阶段PyTorch MLP，保存模型、评估结果和论文素材。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from movie_rating.data import calculate_file_sha256, load_raw_data, write_json
from movie_rating.neural import (
    MovieRatingMLP,
    load_mlp_artifacts,
    predict_ratings,
    save_model_state,
)
from movie_rating.neural_data import (
    MovieRatingDataset,
    NeuralFeaturePreprocessor,
    enrich_interactions,
)


CONFIG_PATH = ROOT / "configs" / "stage4_mlp.json"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "stage4"
REPORT_DIR = ROOT / "reports" / "stage4"
TRAIN_PATH = PROCESSED_DIR / "train_ratings.csv"
TEST_PATH = PROCESSED_DIR / "test_ratings.csv"
STAGE2_MANIFEST_PATH = PROCESSED_DIR / "split_manifest.json"
STAGE4_MANIFEST_PATH = PROCESSED_DIR / "stage4_split_manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_prerequisite_verifications() -> None:
    """训练前复验固定划分和第3阶段成果，但绝不重新训练集成模型。"""
    for script_name in ["verify_stage2.py", "verify_stage3.py"]:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script_name)],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"前置验证失败：{script_name}")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def interaction_id_sha256(values: pd.Series | np.ndarray) -> str:
    ordered = np.sort(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def make_loader(
    features: dict[str, np.ndarray],
    labels: pd.Series | np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        MovieRatingDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def train_one_epoch(
    model: MovieRatingMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    sample_count = 0
    for batch, labels in loader:
        batch = move_batch(batch, device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch)
        loss = loss_function(predictions, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        errors = predictions.detach() - labels
        squared_error_sum += float(torch.sum(errors.square()).cpu())
        absolute_error_sum += float(torch.sum(errors.abs()).cpu())
        sample_count += int(labels.numel())
    mse = squared_error_sum / sample_count
    return {"loss": mse, "rmse": float(np.sqrt(mse)), "mae": absolute_error_sum / sample_count}


def evaluate_loader(
    model: MovieRatingMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.inference_mode():
        for batch, labels in loader:
            output = model(move_batch(batch, device)).cpu().numpy()
            predictions.append(output)
            labels_all.append(labels.numpy())
    predicted = np.concatenate(predictions)
    actual = np.concatenate(labels_all)
    result = metrics(actual, predicted)
    result["loss"] = float(mean_squared_error(actual, predicted))
    return result, predicted


def build_model(
    config: dict[str, Any],
    preprocessor: NeuralFeaturePreprocessor,
    rating_global_mean: float,
) -> MovieRatingMLP:
    return MovieRatingMLP(
        vocabulary_sizes=preprocessor.vocabulary_sizes,
        numeric_feature_count=preprocessor.numeric_feature_count,
        genre_feature_count=preprocessor.genre_feature_count,
        user_embedding_dim=config["user_embedding_dim"],
        movie_embedding_dim=config["movie_embedding_dim"],
        occupation_embedding_dim=config["occupation_embedding_dim"],
        gender_embedding_dim=config["gender_embedding_dim"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        rating_global_mean=rating_global_mean,
    )


def train_validation_model(
    config: dict[str, Any],
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    device: torch.device,
) -> tuple[MovieRatingMLP, NeuralFeaturePreprocessor, pd.DataFrame, dict[str, Any]]:
    seed = int(config["random_seed"])
    preprocessor = NeuralFeaturePreprocessor()
    training_features = preprocessor.fit_transform(training_frame)
    validation_features = preprocessor.transform(validation_frame)
    preprocessor.save(MODEL_DIR / "validation_preprocessor.joblib")
    train_loader = make_loader(
        training_features, training_frame["rating"], config["batch_size"], True, seed
    )
    validation_loader = make_loader(
        validation_features, validation_frame["rating"], config["batch_size"], False, seed
    )
    model = build_model(config, preprocessor, float(training_frame["rating"].mean())).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    loss_function = nn.MSELoss()
    best_rmse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_result = train_one_epoch(model, train_loader, optimizer, loss_function, device)
        validation_result, _ = evaluate_loader(model, validation_loader, device)
        improved = validation_result["rmse"] < best_rmse - config["early_stopping_min_delta"]
        if improved:
            best_rmse = validation_result["rmse"]
            best_epoch = epoch
            stale_epochs = 0
            save_model_state(model, MODEL_DIR / "mlp_validation_best.pt")
        else:
            stale_epochs += 1
        scheduler.step(validation_result["rmse"])
        history.append({
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_rmse": train_result["rmse"],
            "train_mae": train_result["mae"],
            "validation_loss": validation_result["loss"],
            "validation_rmse": validation_result["rmse"],
            "validation_mae": validation_result["mae"],
            "validation_r2": validation_result["r2"],
            "learning_rate": learning_rate,
            "epoch_time_seconds": time.perf_counter() - epoch_start,
            "is_best_epoch": improved,
        })
        print(
            f"Epoch {epoch:02d}/{config['max_epochs']} | "
            f"训练RMSE={train_result['rmse']:.6f} | "
            f"验证RMSE={validation_result['rmse']:.6f} | "
            f"最佳={best_rmse:.6f} | 早停计数={stale_epochs}"
        )
        if stale_epochs >= int(config["early_stopping_patience"]):
            break

    state = torch.load(MODEL_DIR / "mlp_validation_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    best_metrics, _ = evaluate_loader(model, validation_loader, device)
    training_info = {
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "early_stopping_triggered": len(history) < int(config["max_epochs"]),
        "best_validation_metrics": best_metrics,
    }
    return model, preprocessor, pd.DataFrame(history), training_info


def train_final_model(
    config: dict[str, Any],
    full_training_frame: pd.DataFrame,
    best_epoch: int,
    device: torch.device,
) -> tuple[MovieRatingMLP, NeuralFeaturePreprocessor, float]:
    seed = int(config["random_seed"])
    set_reproducible_seed(seed)
    preprocessor = NeuralFeaturePreprocessor()
    features = preprocessor.fit_transform(full_training_frame)
    preprocessor.save(MODEL_DIR / "final_preprocessor.joblib")
    loader = make_loader(features, full_training_frame["rating"], config["batch_size"], True, seed)
    model = build_model(config, preprocessor, float(full_training_frame["rating"].mean())).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    loss_function = nn.MSELoss()
    start = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        result = train_one_epoch(model, loader, optimizer, loss_function, device)
        print(f"最终模型 Epoch {epoch:02d}/{best_epoch} | 训练RMSE={result['rmse']:.6f}")
    elapsed = time.perf_counter() - start
    save_model_state(model, MODEL_DIR / "mlp_final.pt")
    return model, preprocessor, elapsed


def save_training_chart(history: pd.DataFrame, best_epoch: int) -> None:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=(
        "MSE Loss", "RMSE", "MAE", "Learning Rate"
    ))
    for column, row, label in [
        ("train_loss", 1, "训练Loss"), ("validation_loss", 1, "验证Loss"),
        ("train_rmse", 2, "训练RMSE"), ("validation_rmse", 2, "验证RMSE"),
        ("train_mae", 3, "训练MAE"), ("validation_mae", 3, "验证MAE"),
        ("learning_rate", 4, "学习率"),
    ]:
        fig.add_trace(go.Scatter(x=history["epoch"], y=history[column], name=label), row=row, col=1)
    for row in range(1, 5):
        fig.add_vline(x=best_epoch, line_dash="dash", line_color="red", row=row, col=1)
    fig.update_layout(height=1000, title="第4阶段 MLP 训练曲线", template="plotly_white")
    fig.update_xaxes(title_text="Epoch", row=4, col=1)
    fig.write_html(REPORT_DIR / "training_curves.html", include_plotlyjs=True, full_html=True)


def cold_start_table(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    seen_users: set[int],
    seen_movies: set[int],
) -> pd.DataFrame:
    user_seen = test_frame["user_id"].isin(seen_users).to_numpy()
    movie_seen = test_frame["movie_id"].isin(seen_movies).to_numpy()
    groups = {
        "all": np.ones(len(test_frame), dtype=bool),
        "both_seen": user_seen & movie_seen,
        "unseen_user_only": ~user_seen & movie_seen,
        "unseen_movie_only": user_seen & ~movie_seen,
        "both_unseen": ~user_seen & ~movie_seen,
    }
    actual = test_frame["rating"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for group_name, mask in groups.items():
        count = int(mask.sum())
        row: dict[str, Any] = {"group": group_name, "sample_count": count}
        if count:
            row.update({"rmse": float(np.sqrt(mean_squared_error(actual[mask], predictions[mask]))),
                        "mae": float(mean_absolute_error(actual[mask], predictions[mask])),
                        "r2": float(r2_score(actual[mask], predictions[mask])) if count >= 2 else np.nan})
        else:
            row.update({"rmse": np.nan, "mae": np.nan, "r2": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(
    config: dict[str, Any],
    training_info: dict[str, Any],
    test_result: dict[str, float],
    cold_start: pd.DataFrame,
    elapsed: dict[str, float],
    metadata: dict[str, Any],
) -> str:
    # 不依赖pandas可选的tabulate包，直接生成稳定的Markdown表格。
    cold_lines = ["| 分组 | 样本数 | RMSE | MAE | R² |", "|---|---:|---:|---:|---:|"]
    for row in cold_start.to_dict("records"):
        def display(value: Any) -> str:
            return "—" if pd.isna(value) else f"{float(value):.6f}"
        cold_lines.append(
            f"| {row['group']} | {int(row['sample_count'])} | "
            f"{display(row['rmse'])} | {display(row['mae'])} | {display(row['r2'])} |"
        )
    cold_text = "\n".join(cold_lines)
    return f"""# 第4阶段：PyTorch MLP电影评分预测

## 1. 阶段目标与数据

本阶段基于第2阶段固定划分完成PyTorch MLP。固定训练集80000条、固定测试集20000条；80000条训练数据按评分分层为72000条内部训练集和8000条验证集。固定测试集不参与结构、学习率或最佳epoch选择，只在最终模型确定并保存后评估一次。

## 2. 防止数据泄漏

模型未使用用户/电影评分均值、次数、目标编码等标签统计。`rating`仅作为监督标签，`interaction_id`仅用于样本追踪；标题、IMDb链接、邮编均未进入模型。类别映射、缺失值填补和标准化参数只从对应训练数据拟合。

## 3. 特征与网络

- Embedding：用户{config['user_embedding_dim']}维、电影{config['movie_embedding_dim']}维、职业{config['occupation_embedding_dim']}维、性别{config['gender_embedding_dim']}维，索引0统一表示未知类别。
- 静态特征：年龄、职业、性别、19个官方电影类型、上映年。
- 时间特征：评分年、影片年龄，以及月/星期/小时的正余弦周期编码。
- MLP：{' → '.join(str(value) for value in config['hidden_dims'])}，各隐藏层使用Linear、LayerNorm、ReLU和Dropout({config['dropout']})。
- 输出：`1 + 4 × sigmoid(raw)`，天然位于1至5。
- 训练：MSELoss、AdamW，学习率{config['learning_rate']}，权重衰减{config['weight_decay']}，批大小{config['batch_size']}。

## 4. 验证集选择

- 最佳epoch：{training_info['best_epoch']}
- 实际训练epoch：{training_info['epochs_trained']}
- 是否触发早停：{training_info['early_stopping_triggered']}
- 验证RMSE：{training_info['best_validation_metrics']['rmse']:.6f}
- 验证MAE：{training_info['best_validation_metrics']['mae']:.6f}
- 验证R²：{training_info['best_validation_metrics']['r2']:.6f}

最佳epoch确定后，从头使用完整80000条训练数据训练同结构最终模型，未使用测试集调整任何参数。

## 5. 固定测试集结果

| RMSE | MAE | R² |
|---:|---:|---:|
| {test_result['rmse']:.6f} | {test_result['mae']:.6f} | {test_result['r2']:.6f} |

## 6. 冷启动结果

{cold_text}

未见用户或电影映射到Embedding索引0；固定测试集中的未见电影均可正常预测且结果位于1至5。

## 7. 模型产物与复现

- 最终模型：`models/stage4/mlp_final.pt`
- 最终预处理器：`models/stage4/final_preprocessor.joblib`
- 加载元数据：`models/stage4/model_metadata.json`
- 验证阶段最佳权重：`models/stage4/mlp_validation_best.pt`
- 设备：{metadata['device']}
- 验证阶段耗时：{elapsed['validation_training_seconds']:.2f}秒
- 最终模型训练耗时：{elapsed['final_training_seconds']:.2f}秒
- 总耗时：{elapsed['total_seconds']:.2f}秒

通过 `load_mlp_artifacts` 重建模型，再由 `predict_ratings` 执行单条或批量CPU推理。随机种子为{config['random_seed']}，Windows DataLoader使用0个工作进程。

## 8. 局限与论文表述

该模型以用户/电影Embedding学习潜在偏好，并融合人口属性、内容类型和时间信息；与第3阶段依赖平滑评分统计的树模型形成不同建模路线。未知实体只能使用索引0和静态属性回退，冷启动能力仍受限。NCF为可选扩展，本阶段已完成必做MLP。

第5阶段正式的集成学习与神经网络对比尚未开始。
"""


def main() -> int:
    total_start = time.perf_counter()
    run_prerequisite_verifications()
    config = load_json(CONFIG_PATH)
    seed = int(config["random_seed"])
    set_reproducible_seed(seed)
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"要求Python 3.10，当前为{sys.version.split()[0]}")
    device = torch.device("cpu")
    print(f"PyTorch {torch.__version__}；训练设备：{device}")

    stage2_manifest = load_json(STAGE2_MANIFEST_PATH)
    original_train_hash = calculate_file_sha256(TRAIN_PATH)
    original_test_hash = calculate_file_sha256(TEST_PATH)
    if original_train_hash != stage2_manifest["train_sha256"] or original_test_hash != stage2_manifest["test_sha256"]:
        raise RuntimeError("第2阶段固定划分哈希异常，停止训练")

    full_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    # 此处只读取测试交互ID验证集合隔离；测试评分延迟到最终模型保存之后读取。
    test_ids = pd.read_csv(TEST_PATH, usecols=["interaction_id"], encoding="utf-8-sig")
    if len(full_train) != 80_000 or len(test_ids) != 20_000:
        raise RuntimeError("固定训练集或测试集规模异常")
    internal_train, validation = train_test_split(
        full_train, test_size=config["validation_size"], random_state=seed,
        stratify=full_train["rating"], shuffle=True,
    )
    internal_train = internal_train.sort_values("interaction_id").reset_index(drop=True)
    validation = validation.sort_values("interaction_id").reset_index(drop=True)
    if len(internal_train) != 72_000 or len(validation) != 8_000:
        raise RuntimeError("内部训练/验证划分规模异常")
    if set(internal_train["interaction_id"]) & set(validation["interaction_id"]):
        raise RuntimeError("内部训练集与验证集发生重叠")
    if (set(full_train["interaction_id"]) & set(test_ids["interaction_id"])):
        raise RuntimeError("固定训练集与测试集发生重叠")

    write_json({
        "source_train_file": "data/processed/train_ratings.csv",
        "source_test_file": "data/processed/test_ratings.csv",
        "train_sha256": original_train_hash,
        "test_sha256": original_test_hash,
        "random_seed": seed,
        "validation_size": config["validation_size"],
        "stratify_column": "rating",
        "internal_train_rows": len(internal_train),
        "validation_rows": len(validation),
        "fixed_test_rows": len(test_ids),
        "internal_train_interaction_id_sha256": interaction_id_sha256(internal_train["interaction_id"]),
        "validation_interaction_id_sha256": interaction_id_sha256(validation["interaction_id"]),
        "fixed_test_interaction_id_sha256": interaction_id_sha256(test_ids["interaction_id"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, STAGE4_MANIFEST_PATH)

    raw = load_raw_data(RAW_DIR)
    full_train_enriched = enrich_interactions(full_train, raw.users, raw.movies)
    internal_train_enriched = enrich_interactions(internal_train, raw.users, raw.movies)
    validation_enriched = enrich_interactions(validation, raw.users, raw.movies)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validation_start = time.perf_counter()
    _, validation_preprocessor, history, training_info = train_validation_model(
        config, internal_train_enriched, validation_enriched, device
    )
    validation_elapsed = time.perf_counter() - validation_start
    history.to_csv(REPORT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    validation_payload = {
        "best_epoch": training_info["best_epoch"],
        "epochs_trained": training_info["epochs_trained"],
        "early_stopping_triggered": training_info["early_stopping_triggered"],
        **training_info["best_validation_metrics"],
    }
    write_json(validation_payload, REPORT_DIR / "validation_metrics.json")
    save_training_chart(history, training_info["best_epoch"])

    final_model, final_preprocessor, final_elapsed = train_final_model(
        config, full_train_enriched, training_info["best_epoch"], device
    )
    metadata = {
        "model_name": "pytorch_embedding_mlp",
        "model_config": final_model.model_config,
        "network_structure": [*config["hidden_dims"], 1],
        "activation": "ReLU; bounded output uses 1 + 4 * sigmoid(raw)",
        "dropout": config["dropout"],
        "input_feature_names": final_preprocessor.input_feature_names_,
        "numeric_feature_names": final_preprocessor.numeric_feature_names_,
        "genre_categories": final_preprocessor.genre_categories_,
        "vocabulary_sizes": final_preprocessor.vocabulary_sizes,
        "unknown_index": 0,
        "random_seed": seed,
        "best_epoch": training_info["best_epoch"],
        "internal_train_rows": len(internal_train),
        "validation_rows": len(validation),
        "final_train_rows": len(full_train),
        "test_rows": len(test_ids),
        "train_sha256": original_train_hash,
        "test_sha256": original_test_hash,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device": str(device),
        "num_workers": config["num_workers"],
        "deterministic_algorithms": True,
        "model_path": "models/stage4/mlp_final.pt",
        "preprocessor_path": "models/stage4/final_preprocessor.joblib",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(metadata, MODEL_DIR / "model_metadata.json")

    # 最终模型与元数据均固定后，才读取测试标签并执行唯一一次正式测试评估。
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    test_enriched = enrich_interactions(test, raw.users, raw.movies)
    loaded_model, loaded_preprocessor, _ = load_mlp_artifacts(
        MODEL_DIR / "mlp_final.pt", MODEL_DIR / "final_preprocessor.joblib",
        MODEL_DIR / "model_metadata.json", device="cpu",
    )
    predictions = predict_ratings(loaded_model, loaded_preprocessor, test_enriched, device="cpu")
    test_result = metrics(test["rating"].to_numpy(), predictions)
    write_json({**test_result, "sample_count": len(test), "evaluation_count": 1}, REPORT_DIR / "test_metrics.json")
    pd.DataFrame([{**test_result, "sample_count": len(test)}]).to_csv(
        REPORT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig"
    )

    prediction_table = test[["interaction_id", "user_id", "movie_id", "rating"]].copy()
    prediction_table = prediction_table.rename(columns={"rating": "actual_rating"})
    prediction_table["predicted_rating"] = predictions
    prediction_table["error"] = prediction_table["predicted_rating"] - prediction_table["actual_rating"]
    prediction_table["absolute_error"] = prediction_table["error"].abs()
    prediction_table["squared_error"] = prediction_table["error"].pow(2)
    seen_users = set(full_train["user_id"].astype(int))
    seen_movies = set(full_train["movie_id"].astype(int))
    prediction_table["user_seen_in_train"] = prediction_table["user_id"].isin(seen_users)
    prediction_table["movie_seen_in_train"] = prediction_table["movie_id"].isin(seen_movies)
    prediction_table.to_csv(REPORT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    cold_start = cold_start_table(test, predictions, seen_users, seen_movies)
    cold_start.to_csv(REPORT_DIR / "cold_start_metrics.csv", index=False, encoding="utf-8-sig")
    elapsed = {
        "validation_training_seconds": validation_elapsed,
        "final_training_seconds": final_elapsed,
        "total_seconds": time.perf_counter() - total_start,
    }
    training_log = {
        **elapsed,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "random_seed": seed,
        "num_workers": 0,
        "deterministic_algorithms": True,
        "test_evaluation_count": 1,
        "prerequisite_verifications": ["stage2_passed", "stage3_passed"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(training_log, REPORT_DIR / "training_log.json")
    summary = build_summary(config, training_info, test_result, cold_start, elapsed, metadata)
    (REPORT_DIR / "stage4_summary.md").write_text(summary, encoding="utf-8")

    if calculate_file_sha256(TRAIN_PATH) != original_train_hash or calculate_file_sha256(TEST_PATH) != original_test_hash:
        raise RuntimeError("训练过程中固定数据文件被修改")
    print("\n=== 第4阶段结果摘要 ===")
    print(f"最佳epoch：{training_info['best_epoch']}；验证RMSE：{validation_payload['rmse']:.6f}")
    print(f"测试RMSE={test_result['rmse']:.6f}，MAE={test_result['mae']:.6f}，R²={test_result['r2']:.6f}")
    print(f"总耗时：{elapsed['total_seconds']:.2f}秒")
    print("第4阶段主程序执行完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
