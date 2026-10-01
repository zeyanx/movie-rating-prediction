"""第4阶段独立验收：不重新训练，只验证数据、模型、报告和推理闭环。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from movie_rating.data import calculate_file_sha256, load_raw_data
from movie_rating.neural import load_mlp_artifacts, predict_ratings
from movie_rating.neural_data import NeuralFeaturePreprocessor, enrich_interactions


PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "stage4"
REPORT_DIR = ROOT / "reports" / "stage4"
TRAIN_PATH = PROCESSED_DIR / "train_ratings.csv"
TEST_PATH = PROCESSED_DIR / "test_ratings.csv"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict:
    require(path.is_file(), f"缺少文件：{path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_prerequisites() -> None:
    for script_name in ["verify_stage2.py", "verify_stage3.py"]:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script_name)], cwd=ROOT,
            text=True, encoding="utf-8", errors="replace", capture_output=True,
            check=False,
        )
        require(result.returncode == 0, f"{script_name}复验失败：\n{result.stdout}\n{result.stderr}")


def id_hash(values: pd.Series | np.ndarray) -> str:
    ordered = np.sort(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def recompute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def main() -> int:
    print("=== 第4阶段独立验收 ===")
    require(sys.version_info[:2] == (3, 10), f"要求Python 3.10，当前为{sys.version.split()[0]}")
    print(f"Python：{sys.version.split()[0]}；PyTorch：{torch.__version__}")
    verify_prerequisites()
    print("第2、3阶段复验：通过")

    config = load_json(ROOT / "configs" / "stage4_mlp.json")
    required_config = {
        "random_seed", "validation_size", "user_embedding_dim", "movie_embedding_dim",
        "occupation_embedding_dim", "gender_embedding_dim", "hidden_dims", "dropout",
        "batch_size", "learning_rate", "weight_decay", "max_epochs",
        "early_stopping_patience", "early_stopping_min_delta", "num_workers", "device",
    }
    require(required_config.issubset(config), "stage4_mlp.json配置不完整")
    require(config["random_seed"] == 42, "随机种子不是42")
    require(config["num_workers"] == 0, "Windows环境num_workers必须为0")

    stage2_manifest = load_json(PROCESSED_DIR / "split_manifest.json")
    stage4_manifest = load_json(PROCESSED_DIR / "stage4_split_manifest.json")
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    require(len(train) == 80_000, "固定训练集不是80000条")
    require(len(test) == 20_000, "固定测试集不是20000条")
    require(calculate_file_sha256(TRAIN_PATH) == stage2_manifest["train_sha256"], "固定训练集哈希变化")
    require(calculate_file_sha256(TEST_PATH) == stage2_manifest["test_sha256"], "固定测试集哈希变化")
    require(stage4_manifest["train_sha256"] == stage2_manifest["train_sha256"], "第4阶段训练哈希记录错误")
    require(stage4_manifest["test_sha256"] == stage2_manifest["test_sha256"], "第4阶段测试哈希记录错误")

    internal_train, validation = train_test_split(
        train, test_size=0.10, random_state=42, stratify=train["rating"], shuffle=True
    )
    internal_train = internal_train.sort_values("interaction_id").reset_index(drop=True)
    validation = validation.sort_values("interaction_id").reset_index(drop=True)
    require(len(internal_train) == 72_000, "内部训练集不是72000条")
    require(len(validation) == 8_000, "内部验证集不是8000条")
    train_ids = set(internal_train["interaction_id"])
    validation_ids = set(validation["interaction_id"])
    test_ids = set(test["interaction_id"])
    require(not (train_ids & validation_ids), "内部训练集与验证集重叠")
    require(not ((train_ids | validation_ids) & test_ids), "训练/验证数据与固定测试集重叠")
    require(id_hash(internal_train["interaction_id"]) == stage4_manifest["internal_train_interaction_id_sha256"], "内部训练划分不可复现")
    require(id_hash(validation["interaction_id"]) == stage4_manifest["validation_interaction_id_sha256"], "验证划分不可复现")
    require(id_hash(test["interaction_id"]) == stage4_manifest["fixed_test_interaction_id_sha256"], "固定测试ID哈希错误")

    required_artifacts = [
        MODEL_DIR / "mlp_validation_best.pt",
        MODEL_DIR / "mlp_final.pt",
        MODEL_DIR / "validation_preprocessor.joblib",
        MODEL_DIR / "final_preprocessor.joblib",
        MODEL_DIR / "model_metadata.json",
        REPORT_DIR / "training_history.csv",
        REPORT_DIR / "validation_metrics.json",
        REPORT_DIR / "test_metrics.json",
        REPORT_DIR / "test_metrics.csv",
        REPORT_DIR / "test_predictions.csv",
        REPORT_DIR / "cold_start_metrics.csv",
        REPORT_DIR / "training_curves.html",
        REPORT_DIR / "training_log.json",
        REPORT_DIR / "stage4_summary.md",
    ]
    for path in required_artifacts:
        require(path.is_file() and path.stat().st_size > 0, f"产物缺失或为空：{path.relative_to(ROOT)}")

    metadata = load_json(MODEL_DIR / "model_metadata.json")
    require(metadata["best_epoch"] >= 1, "最佳epoch无效")
    require(metadata["final_train_rows"] == 80_000, "最终模型训练样本数记录错误")
    require(metadata["test_rows"] == 20_000, "测试样本数记录错误")
    require(metadata["unknown_index"] == 0, "未知类别索引不是0")
    require(metadata["device"] == "cpu", "最终模型未记录为CPU设备")

    raw = load_raw_data(ROOT / "data" / "raw")
    train_enriched = enrich_interactions(train, raw.users, raw.movies)
    test_enriched = enrich_interactions(test, raw.users, raw.movies)
    internal_enriched = enrich_interactions(internal_train, raw.users, raw.movies)
    final_preprocessor = NeuralFeaturePreprocessor.load(MODEL_DIR / "final_preprocessor.joblib")
    validation_preprocessor = NeuralFeaturePreprocessor.load(MODEL_DIR / "validation_preprocessor.joblib")
    require(set(final_preprocessor.user_mapping_) == set(train["user_id"]), "最终用户映射并非只来自完整训练集")
    require(set(final_preprocessor.movie_mapping_) == set(train["movie_id"]), "最终电影映射并非只来自完整训练集")
    require(set(validation_preprocessor.user_mapping_) == set(internal_train["user_id"]), "验证版用户映射不来自内部训练集")
    require(set(validation_preprocessor.movie_mapping_) == set(internal_train["movie_id"]), "验证版电影映射不来自内部训练集")

    forbidden = {"rating", "interaction_id", "title", "imdb_url", "zip_code", "user_id", "movie_id"}
    feature_names = set(final_preprocessor.input_feature_names_)
    require(not (feature_names & forbidden), f"特征名中出现禁止字段：{feature_names & forbidden}")
    target_terms = ["smoothed_mean", "rating_count", "target_encoding", "global_mean"]
    require(not any(term in name for name in feature_names for term in target_terms), "神经网络输入含评分统计特征")
    require(len(final_preprocessor.genre_categories_) == 19, "电影类型特征不是19个")

    transformed = final_preprocessor.transform(test_enriched)
    for name, values in transformed.items():
        require(np.isfinite(values).all(), f"{name}含NaN或无穷值")
    vocab = final_preprocessor.vocabulary_sizes
    for name, vocab_name in [
        ("user_index", "user"), ("movie_index", "movie"),
        ("gender_index", "gender"), ("occupation_index", "occupation"),
    ]:
        require(int(transformed[name].min()) >= 0, f"{name}存在负索引")
        require(int(transformed[name].max()) < vocab[vocab_name], f"{name}超出Embedding范围")

    known = final_preprocessor.transform(train_enriched.iloc[[0]])
    require(known["user_index"][0] > 0 and known["movie_index"][0] > 0, "已知用户或电影错误映射到0")
    unknown_user_row = train_enriched.iloc[[0]].copy()
    unknown_user_row["user_id"] = 999_999
    require(final_preprocessor.transform(unknown_user_row)["user_index"][0] == 0, "未见用户没有映射到0")
    unseen_movies = set(test["movie_id"]) - set(train["movie_id"])
    require(len(unseen_movies) == 30, "固定测试集未见电影数不是30")
    unseen_row = test_enriched[test_enriched["movie_id"].isin(unseen_movies)].iloc[[0]]
    require(final_preprocessor.transform(unseen_row)["movie_index"][0] == 0, "未见电影没有映射到0")

    model, loaded_preprocessor, _ = load_mlp_artifacts(
        MODEL_DIR / "mlp_final.pt", MODEL_DIR / "final_preprocessor.joblib",
        MODEL_DIR / "model_metadata.json", device="cpu",
    )
    require(not model.training, "加载后的模型没有切换到eval模式")
    parameters = list(model.parameters())
    require(parameters and any(torch.count_nonzero(parameter).item() > 0 for parameter in parameters), "模型参数异常：全部为零")

    single_prediction = predict_ratings(model, loaded_preprocessor, test_enriched.iloc[[0]], device="cpu")
    require(single_prediction.shape == (1,), "单条预测输出形状错误")
    sample_a = predict_ratings(model, loaded_preprocessor, test_enriched.iloc[:32], device="cpu")
    sample_b = predict_ratings(model, loaded_preprocessor, test_enriched.iloc[:32], device="cpu")
    require(np.array_equal(sample_a, sample_b), "相同输入的两次CPU推理结果不一致")
    require(np.all((sample_a >= 1.0) & (sample_a <= 5.0)), "样例预测超出1至5")

    predictions_file = pd.read_csv(REPORT_DIR / "test_predictions.csv", encoding="utf-8-sig")
    require(len(predictions_file) == 20_000, "保存的测试预测不是20000行")
    require(predictions_file["interaction_id"].tolist() == test["interaction_id"].tolist(), "预测文件interaction_id与测试集不一致")
    reloaded_predictions = predict_ratings(model, loaded_preprocessor, test_enriched, device="cpu")
    require(np.all((reloaded_predictions >= 1.0) & (reloaded_predictions <= 5.0)), "完整测试预测超出1至5")
    require(np.allclose(reloaded_predictions, predictions_file["predicted_rating"], rtol=0, atol=5e-7), "重载模型预测与保存结果不一致")

    report_metrics = load_json(REPORT_DIR / "test_metrics.json")
    recalculated = recompute_metrics(
        predictions_file["actual_rating"].to_numpy(),
        predictions_file["predicted_rating"].to_numpy(),
    )
    for name, value in recalculated.items():
        require(np.isfinite(report_metrics[name]), f"测试{name}不是有限数值")
        # CSV十进制序列化会带来约1e-9量级的浮点舍入误差。
        require(np.isclose(value, report_metrics[name], rtol=0, atol=2e-9), f"测试{name}复算不一致")
    require(report_metrics["evaluation_count"] == 1, "测试集评估次数记录不是1")

    cold = pd.read_csv(REPORT_DIR / "cold_start_metrics.csv", encoding="utf-8-sig")
    require(set(cold["group"]) == {"all", "both_seen", "unseen_user_only", "unseen_movie_only", "both_unseen"}, "冷启动分组不完整")
    require(int(cold.loc[cold["group"] == "all", "sample_count"].iloc[0]) == 20_000, "冷启动all分组数量错误")
    exclusive_count = int(cold.loc[cold["group"] != "all", "sample_count"].sum())
    require(exclusive_count == 20_000, "互斥冷启动分组数量之和错误")
    require(int(cold.loc[cold["group"] == "unseen_movie_only", "sample_count"].iloc[0]) == 33, "未见电影测试记录数不是33")

    history = pd.read_csv(REPORT_DIR / "training_history.csv", encoding="utf-8-sig")
    validation_metrics = load_json(REPORT_DIR / "validation_metrics.json")
    best_epoch = int(metadata["best_epoch"])
    require(best_epoch in set(history["epoch"]), "训练历史缺少最佳epoch")
    require(best_epoch == int(validation_metrics["best_epoch"]), "最佳epoch与验证报告不一致")
    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]
    require(np.isclose(best_row["validation_rmse"], validation_metrics["rmse"], atol=1e-10), "最佳验证RMSE与训练历史不一致")

    log = load_json(REPORT_DIR / "training_log.json")
    require(log["test_evaluation_count"] == 1, "训练日志中的测试评估次数错误")
    require((REPORT_DIR / "training_curves.html").stat().st_size > 100_000, "训练曲线HTML过小或不完整")
    require("第5阶段" in (REPORT_DIR / "stage4_summary.md").read_text(encoding="utf-8"), "阶段报告未说明第5阶段状态")
    # 后续阶段完成后仍应能够复验第4阶段；因此验证历史报告中的阶段边界，
    # 不把合法存在的reports/stage5目录误判为第4阶段失败。

    print(f"内部训练/验证/固定测试：{len(internal_train)}/{len(validation)}/{len(test)}")
    print(f"最佳epoch：{best_epoch}；测试RMSE={report_metrics['rmse']:.6f}，MAE={report_metrics['mae']:.6f}，R²={report_metrics['r2']:.6f}")
    print("数据隔离、无泄漏预处理、Embedding边界、模型重载、指标复算和冷启动：全部通过")
    print("第4阶段验证通过")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print(f"第4阶段验证失败：{error}", file=sys.stderr)
        sys.exit(1)
