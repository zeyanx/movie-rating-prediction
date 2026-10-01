"""独立验收第2阶段的划分、基线模型、指标和可重复性。"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from movie_rating.baseline import GlobalMeanBaseline, evaluate_predictions  # noqa: E402
from movie_rating.data import (  # noqa: E402
    RATINGS_COLUMNS,
    SPLIT_COLUMNS,
    calculate_file_sha256,
    load_raw_data,
    split_ratings,
    validate_raw_data,
)


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "global_mean_baseline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports" / "stage2"
TOLERANCE = 1e-12


def require(condition: bool, message: str) -> None:
    """条件不满足时立即中止，失败状态不会被伪装成成功。"""
    if not condition:
        raise AssertionError(message)


def verify_stage1_subprocess() -> None:
    """用当前解释器复跑第1阶段验收，确认基础数据仍然有效。"""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_stage1.py")],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "第1阶段复验失败：\n" + result.stdout + "\n" + result.stderr
        )
    require("第1阶段验证通过" in result.stdout, "第1阶段脚本没有输出通过标记")
    print("第1阶段复验：通过")


def load_json(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size > 0, f"缺少或空文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_split(
    raw_ratings: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    manifest: dict,
) -> None:
    """验证划分无交叉、无丢失、可重复且分层比例合理。"""
    require(list(train.columns) == SPLIT_COLUMNS, "训练集字段不符合要求")
    require(list(test.columns) == SPLIT_COLUMNS, "测试集字段不符合要求")
    require(len(raw_ratings) == 100_000, "原始评分数不是100000")
    require(len(train) == 80_000, "训练集记录数不是80000")
    require(len(test) == 20_000, "测试集记录数不是20000")
    require(train["interaction_id"].is_unique, "训练集interaction_id不唯一")
    require(test["interaction_id"].is_unique, "测试集interaction_id不唯一")

    train_ids = set(train["interaction_id"])
    test_ids = set(test["interaction_id"])
    require(train_ids.isdisjoint(test_ids), "训练集和测试集存在交叉记录")
    require(train_ids | test_ids == set(range(100_000)), "划分后存在记录丢失或新增")
    require(
        not train.duplicated(["user_id", "movie_id"]).any(),
        "训练集存在重复用户—电影组合",
    )
    require(
        not test.duplicated(["user_id", "movie_id"]).any(),
        "测试集存在重复用户—电影组合",
    )

    reconstructed = (
        pd.concat([train, test], ignore_index=True)
        .sort_values("interaction_id")
        .reset_index(drop=True)
    )
    require(
        reconstructed["interaction_id"].tolist() == list(range(100_000)),
        "interaction_id序列不完整",
    )
    pd.testing.assert_frame_equal(
        reconstructed[RATINGS_COLUMNS],
        raw_ratings.reset_index(drop=True),
        check_dtype=True,
    )

    full_proportions = raw_ratings["rating"].value_counts(normalize=True)
    for name, frame in (("训练集", train), ("测试集", test)):
        proportions = frame["rating"].value_counts(normalize=True)
        drift = (proportions - full_proportions).abs().max()
        require(float(drift) <= 0.0001, f"{name}评分分层比例偏差过大：{drift}")

    require(manifest["random_state"] == 42, "随机种子不是42")
    require(math.isclose(manifest["test_size"], 0.20), "测试比例不是0.20")
    require(manifest["shuffle"] is True, "shuffle未启用")
    require(manifest["stratify_column"] == "rating", "未按rating分层")
    require(manifest["train_rows"] == 80_000, "清单中的训练集数量错误")
    require(manifest["test_rows"] == 20_000, "清单中的测试集数量错误")
    require(
        manifest["source_sha256"] == calculate_file_sha256(RAW_DIR / "ratings.csv"),
        "原始评分文件SHA-256与清单不一致",
    )
    require(
        manifest["train_sha256"]
        == calculate_file_sha256(PROCESSED_DIR / "train_ratings.csv"),
        "训练集SHA-256与清单不一致",
    )
    require(
        manifest["test_sha256"]
        == calculate_file_sha256(PROCESSED_DIR / "test_ratings.csv"),
        "测试集SHA-256与清单不一致",
    )

    # 在内存中使用同一配置再次划分，直接证明固定随机种子的可重复性。
    repeat_train, repeat_test = split_ratings(raw_ratings, test_size=0.20, random_state=42)
    pd.testing.assert_frame_equal(repeat_train, train, check_dtype=True)
    pd.testing.assert_frame_equal(repeat_test, test, check_dtype=True)


def verify_model_and_metrics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict,
) -> None:
    """重新加载模型、计算预测和指标，并与报告逐项核对。"""
    unfitted_model = GlobalMeanBaseline()
    try:
        unfitted_model.predict(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("未拟合模型调用predict时没有报错")

    require(MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0, "模型文件不存在或为空")
    model = GlobalMeanBaseline.load(MODEL_PATH)
    expected_mean = float(train["rating"].mean())
    require(
        math.isclose(model.global_mean_, expected_mean, rel_tol=0.0, abs_tol=TOLERANCE),
        "模型全局均值不是训练集均值",
    )
    require(model.training_samples_ == 80_000, "模型记录的训练样本数错误")
    require(model.random_state == 42, "模型随机种子错误")

    train_pred = model.predict_dataframe(train)
    test_pred = model.predict_dataframe(test)
    require(train_pred.shape == (80_000,), "训练预测数量错误")
    require(test_pred.shape == (20_000,), "测试预测数量错误")
    require(np.isfinite(test_pred).all(), "测试预测包含NaN或无穷大")
    require(np.unique(test_pred).size == 1, "全局均值基线预测不全相同")
    require(((test_pred >= 1) & (test_pred <= 5)).all(), "预测超出1至5分")

    recomputed = {
        "train_metrics": evaluate_predictions(train["rating"], train_pred),
        "test_metrics": evaluate_predictions(test["rating"], test_pred),
    }
    require(metrics["model_name"] == "global_mean_baseline", "指标模型名称错误")
    require(
        math.isclose(metrics["global_mean"], expected_mean, rel_tol=0.0, abs_tol=TOLERANCE),
        "指标文件中的全局均值错误",
    )
    for split_name in ("train_metrics", "test_metrics"):
        for metric_name in ("rmse", "mae", "r2"):
            value = metrics[split_name][metric_name]
            require(math.isfinite(value), f"{split_name}.{metric_name}不是有限值")
            require(
                math.isclose(
                    value,
                    recomputed[split_name][metric_name],
                    rel_tol=0.0,
                    abs_tol=TOLERANCE,
                ),
                f"{split_name}.{metric_name}与重新计算结果不一致",
            )

    prediction_path = REPORT_DIR / "test_predictions.csv"
    require(prediction_path.is_file() and prediction_path.stat().st_size > 0, "缺少测试预测文件")
    predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
    expected_columns = [
        "interaction_id",
        "user_id",
        "movie_id",
        "rating",
        "predicted_rating",
        "residual",
        "absolute_error",
        "squared_error",
    ]
    require(list(predictions.columns) == expected_columns, "测试预测文件字段错误")
    require(len(predictions) == 20_000, "测试预测文件行数错误")
    for column in ("interaction_id", "user_id", "movie_id", "rating"):
        require(
            np.array_equal(predictions[column].to_numpy(), test[column].to_numpy()),
            f"测试预测文件的{column}与测试集不一致",
        )
    require(
        np.allclose(predictions["predicted_rating"], test_pred, rtol=0.0, atol=TOLERANCE),
        "保存的预测值与模型输出不一致",
    )
    expected_residual = test["rating"].to_numpy(dtype=float) - test_pred
    require(
        np.allclose(predictions["residual"], expected_residual, rtol=0.0, atol=TOLERANCE),
        "残差计算错误",
    )
    require(
        np.allclose(
            predictions["absolute_error"], np.abs(expected_residual), rtol=0.0, atol=TOLERANCE
        ),
        "绝对误差计算错误",
    )
    require(
        np.allclose(
            predictions["squared_error"], expected_residual**2, rtol=0.0, atol=TOLERANCE
        ),
        "平方误差计算错误",
    )


def main() -> int:
    print("=== 第2阶段独立验收 ===")
    verify_stage1_subprocess()
    data = load_raw_data(RAW_DIR)
    validate_raw_data(data)

    required_files = [
        PROCESSED_DIR / "train_ratings.csv",
        PROCESSED_DIR / "test_ratings.csv",
        PROCESSED_DIR / "split_manifest.json",
        REPORT_DIR / "data_quality.json",
        REPORT_DIR / "baseline_metrics.json",
        REPORT_DIR / "test_predictions.csv",
        REPORT_DIR / "stage2_summary.md",
        MODEL_PATH,
    ]
    for path in required_files:
        require(path.is_file() and path.stat().st_size > 0, f"缺少或空文件：{path}")

    train = pd.read_csv(PROCESSED_DIR / "train_ratings.csv", encoding="utf-8-sig")
    test = pd.read_csv(PROCESSED_DIR / "test_ratings.csv", encoding="utf-8-sig")
    manifest = load_json(PROCESSED_DIR / "split_manifest.json")
    metrics = load_json(REPORT_DIR / "baseline_metrics.json")
    quality = load_json(REPORT_DIR / "data_quality.json")
    require(quality["dataset"] == "MovieLens 100K", "数据质量报告的数据集名称错误")

    verify_split(data.ratings, train, test, manifest)
    verify_model_and_metrics(train, test, metrics)

    print(f"训练集：{len(train)}条；测试集：{len(test)}条")
    print(f"训练集全局平均评分：{metrics['global_mean']:.8f}")
    print(
        "测试集指标："
        f"RMSE={metrics['test_metrics']['rmse']:.6f}, "
        f"MAE={metrics['test_metrics']['mae']:.6f}, "
        f"R²={metrics['test_metrics']['r2']:.6e}"
    )
    print("划分完整性、文件哈希、模型加载、指标复算和可重复性：全部通过")
    print("第2阶段验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
