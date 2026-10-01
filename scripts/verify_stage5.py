"""第5阶段独立验收：不重新训练、不重跑Bootstrap或效率基准。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from movie_rating.comparison import MODEL_NAMES, PREDICTION_COLUMNS, calculate_regression_metrics
from movie_rating.data import calculate_file_sha256


REPORT_DIR = ROOT / "reports" / "stage5"
TRAIN_PATH = ROOT / "data" / "processed" / "train_ratings.csv"
TEST_PATH = ROOT / "data" / "processed" / "test_ratings.csv"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict:
    require(path.is_file(), f"缺少文件：{path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_prerequisites() -> None:
    for script_name in ["verify_stage2.py", "verify_stage3.py", "verify_stage4.py"]:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script_name)], cwd=ROOT,
            text=True, encoding="utf-8", errors="replace", capture_output=True, check=False,
        )
        require(result.returncode == 0, f"{script_name}复验失败：\n{result.stdout}\n{result.stderr}")


def main() -> int:
    print("=== 第5阶段独立验收 ===")
    require(sys.version_info[:2] == (3, 10), f"要求Python 3.10，当前为{sys.version.split()[0]}")
    verify_prerequisites()
    print("第2、3、4阶段复验：通过")

    config = load_json(ROOT / "configs" / "stage5_comparison.json")
    required_config = {
        "random_seed", "bootstrap_resamples", "confidence_level", "permutation_repeats",
        "benchmark_warmup_runs", "benchmark_repeats", "benchmark_device", "model_order",
        "primary_metric", "prediction_range",
    }
    require(required_config.issubset(config), "第5阶段配置字段不完整")
    require(config["random_seed"] == 42, "第5阶段随机种子不是42")
    require(config["bootstrap_resamples"] == 2000, "Bootstrap次数不是2000")
    require(config["confidence_level"] == 0.95, "置信水平不是95%")
    require(config["permutation_repeats"] == 10, "排列重复次数不是10")
    require(config["benchmark_repeats"] >= 5, "效率正式计时少于5次")
    require(config["model_order"] == MODEL_NAMES, "模型顺序配置异常")

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    require(len(train) == 80_000, "固定训练集不是80000条")
    require(len(test) == 20_000, "固定测试集不是20000条")
    stage2_manifest = load_json(ROOT / "data" / "processed" / "split_manifest.json")
    require(calculate_file_sha256(TRAIN_PATH) == stage2_manifest["train_sha256"], "固定训练集哈希变化")
    require(calculate_file_sha256(TEST_PATH) == stage2_manifest["test_sha256"], "固定测试集哈希变化")

    hashes = load_json(REPORT_DIR / "source_artifact_hashes.json")
    require(hashes["all_unchanged"] is True, "主程序未确认源产物保持不变")
    for item in hashes["artifacts"]:
        path = ROOT / item["path"]
        require(path.is_file(), f"源产物缺失：{item['path']}")
        current = calculate_file_sha256(path)
        require(item["unchanged"] is True, f"源产物执行前后变化：{item['path']}")
        require(current == item["before_sha256"] == item["after_sha256"], f"源产物当前哈希变化：{item['path']}")

    required_files = [
        "unified_test_predictions.csv", "metrics_comparison.csv", "metrics_comparison.json",
        "paired_bootstrap_summary.csv", "paired_bootstrap_summary.json", "segment_metrics.csv",
        "segmentation_metadata.json", "residual_summary.csv", "actual_rating_diagnostics.csv",
        "ensemble_feature_importance.csv", "ensemble_group_importance.csv",
        "mlp_permutation_importance_repeats.csv", "mlp_permutation_importance.csv",
        "feature_group_comparison.csv", "efficiency_benchmark.csv", "efficiency_environment.json",
        "model_recommendation.json", "metrics_comparison.html", "residual_analysis.html",
        "segment_comparison.html", "feature_importance_comparison.html", "efficiency_comparison.html",
        "paper_results_material.md", "stage5_summary.md",
    ]
    for name in required_files:
        path = REPORT_DIR / name
        require(path.is_file() and path.stat().st_size > 0, f"产物缺失或为空：reports/stage5/{name}")

    unified = pd.read_csv(REPORT_DIR / "unified_test_predictions.csv", encoding="utf-8-sig")
    require(len(unified) == 20_000, "统一预测表不是20000行")
    require(unified["interaction_id"].is_unique, "统一预测表interaction_id不唯一")
    for column in ["interaction_id", "user_id", "movie_id"]:
        require(unified[column].tolist() == test[column].tolist(), f"统一预测表{column}与固定测试集不一致")
    require(np.array_equal(unified["actual_rating"].to_numpy(), test["rating"].to_numpy()), "统一预测表真实评分不一致")
    for model, column in PREDICTION_COLUMNS.items():
        require(column in unified.columns, f"统一预测表缺少{model}预测")
        values = unified[column].to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"{model}预测含非有限值")
        require(((values >= 1.0) & (values <= 5.0)).all(), f"{model}预测超出1至5")

    metrics = pd.read_csv(REPORT_DIR / "metrics_comparison.csv", encoding="utf-8-sig")
    require(set(metrics["model"]) == set(MODEL_NAMES), "统一指标模型不完整")
    indexed = metrics.set_index("model")
    actual = unified["actual_rating"].to_numpy()
    for model in MODEL_NAMES:
        calculated = calculate_regression_metrics(actual, unified[PREDICTION_COLUMNS[model]])
        for metric in ["rmse", "mae", "r2"]:
            require(np.isclose(calculated[metric], indexed.loc[model, metric], rtol=0, atol=2e-9), f"{model}的{metric}复算不一致")
    require(indexed.sort_values("rmse")["rmse_rank"].tolist() == list(range(1, 6)), "RMSE排名错误")
    require(indexed.sort_values("mae")["mae_rank"].tolist() == list(range(1, 6)), "MAE排名错误")
    require(indexed.sort_values("r2", ascending=False)["r2_rank"].tolist() == list(range(1, 6)), "R²排名错误")
    baseline_rmse = indexed.loc["global_mean", "rmse"]
    baseline_mae = indexed.loc["global_mean", "mae"]
    for model in MODEL_NAMES:
        expected_rmse = (baseline_rmse - indexed.loc[model, "rmse"]) / baseline_rmse * 100
        expected_mae = (baseline_mae - indexed.loc[model, "mae"]) / baseline_mae * 100
        require(np.isclose(expected_rmse, indexed.loc[model, "rmse_improvement_vs_baseline_percent"], atol=1e-10), f"{model} RMSE改善率错误")
        require(np.isclose(expected_mae, indexed.loc[model, "mae_improvement_vs_baseline_percent"], atol=1e-10), f"{model} MAE改善率错误")

    prior = {
        "global_mean": load_json(ROOT / "reports" / "stage2" / "baseline_metrics.json")["test_metrics"],
        **load_json(ROOT / "reports" / "stage3" / "test_metrics.json"),
        "mlp": load_json(ROOT / "reports" / "stage4" / "test_metrics.json"),
    }
    for model, values in prior.items():
        for metric in ["rmse", "mae", "r2"]:
            require(np.isclose(indexed.loc[model, metric], values[metric], atol=2e-8), f"{model}统一指标与前阶段不一致")

    bootstrap = pd.read_csv(REPORT_DIR / "paired_bootstrap_summary.csv", encoding="utf-8-sig")
    require(len(bootstrap) == 6, "Bootstrap汇总应有6行")
    require(set(bootstrap["comparison"]) == {"mlp_vs_random_forest", "mlp_vs_adaboost", "mlp_vs_xgboost"}, "Bootstrap比较对象不完整")
    require(set(bootstrap["metric"]) == {"rmse", "mae"}, "Bootstrap指标不完整")
    require((bootstrap["resamples"] == 2000).all() and (bootstrap["random_seed"] == 42).all(), "Bootstrap配置记录错误")
    require((bootstrap["confidence_level"] == 0.95).all(), "Bootstrap置信水平记录错误")
    require((bootstrap["delta_definition"] == "mlp_metric_minus_ensemble_metric").all(), "Bootstrap差异方向定义错误")
    require((bootstrap["confidence_lower"] <= bootstrap["confidence_upper"]).all(), "Bootstrap置信区间上下界错误")
    require(bootstrap["mlp_win_probability"].between(0, 1).all(), "MLP胜率超出0至1")

    segmentation = load_json(REPORT_DIR / "segmentation_metadata.json")
    require(segmentation["unseen_test_user_count"] == 0, "未见用户记录数异常")
    require(segmentation["unseen_test_movie_count"] == 30, "未见电影数不是30")
    require(segmentation["unseen_movie_test_rows"] == 33, "未见电影测试记录数不是33")
    user_counts = train.groupby("user_id").size()
    movie_counts = train.groupby("movie_id").size()
    expected_user_q = user_counts.quantile([1 / 3, 2 / 3]).tolist()
    expected_movie_q = movie_counts.quantile([1 / 3, 2 / 3]).tolist()
    require(np.allclose(list(segmentation["user_activity_thresholds"].values()), expected_user_q), "用户活跃度阈值并非来自训练集")
    require(np.allclose(list(segmentation["movie_popularity_thresholds"].values()), expected_movie_q), "电影流行度阈值并非来自训练集")
    segment = pd.read_csv(REPORT_DIR / "segment_metrics.csv", encoding="utf-8-sig")
    for segment_type in ["actual_rating", "cold_start", "user_activity", "movie_popularity"]:
        for model in MODEL_NAMES:
            total = int(segment[(segment["segment_type"] == segment_type) & (segment["model"] == model)]["sample_count"].sum())
            require(total == 20_000, f"{segment_type}/{model}分组数量之和不是20000")
    require(not ((segment["sample_count"] == 0) & segment[["rmse", "mae"]].notna().any(axis=1)).any(), "空分组伪造了指标")

    detailed = pd.read_csv(REPORT_DIR / "ensemble_feature_importance.csv", encoding="utf-8-sig")
    grouped = pd.read_csv(REPORT_DIR / "ensemble_group_importance.csv", encoding="utf-8-sig")
    require(set(detailed["model"]) == {"random_forest", "adaboost", "xgboost"}, "集成重要性模型不完整")
    for model in ["random_forest", "adaboost", "xgboost"]:
        require(np.isclose(detailed.loc[detailed["model"] == model, "normalized_importance"].sum(), 1.0, atol=1e-8), f"{model}单特征重要性和不为1")
        require(np.isclose(grouped.loc[grouped["model"] == model, "importance"].sum(), 1.0, atol=1e-8), f"{model}分组重要性和不为1")

    mlp_repeats = pd.read_csv(REPORT_DIR / "mlp_permutation_importance_repeats.csv", encoding="utf-8-sig")
    mlp_summary = pd.read_csv(REPORT_DIR / "mlp_permutation_importance.csv", encoding="utf-8-sig")
    expected_groups = {
        "user_embedding", "movie_embedding", "gender_embedding", "occupation_embedding",
        "age", "release_year", "rating_year", "movie_age_at_rating",
        "rating_month_cycle", "rating_day_of_week_cycle", "rating_hour_cycle", "genres",
    }
    require(set(mlp_summary["feature_group"]) == expected_groups, "MLP排列重要性分组不完整")
    require(len(mlp_repeats) == 120, "MLP排列重复结果不是120行")
    require((mlp_repeats.groupby("feature_group").size() == 10).all(), "MLP每组排列次数不是10")
    require((mlp_repeats["evaluation_rows"] == 8000).all(), "MLP排列重要性不是基于8000条验证记录")
    require(np.isfinite(mlp_repeats[["permuted_rmse", "rmse_increase"]].to_numpy()).all(), "MLP排列结果含非有限值")
    expected_validation = load_json(ROOT / "reports" / "stage4" / "validation_metrics.json")["rmse"]
    require(np.allclose(mlp_summary["baseline_rmse"], expected_validation, atol=2e-8), "MLP基础验证RMSE不一致")
    normalized_sum = float(mlp_summary["positive_normalized_importance"].sum())
    require(np.isclose(normalized_sum, 1.0, atol=1e-8) or np.isclose(normalized_sum, 0.0, atol=1e-8), "MLP正归一化重要性之和异常")
    combined = pd.read_csv(REPORT_DIR / "feature_group_comparison.csv", encoding="utf-8-sig")
    require({"native_tree_split_importance", "grouped_permutation_rmse_increase"}.issubset(set(combined["importance_method"])), "联合重要性未区分方法")

    efficiency = pd.read_csv(REPORT_DIR / "efficiency_benchmark.csv", encoding="utf-8-sig")
    require(set(efficiency["model"]) == set(MODEL_NAMES), "效率基准模型不完整")
    require((efficiency["benchmark_repeats"] >= 5).all(), "效率正式计时少于5次")
    require((efficiency["warmup_runs"] >= 1).all(), "效率基准没有预热")
    numeric_efficiency = efficiency[["artifact_size_bytes", "load_time_mean_seconds", "inference_time_mean_seconds", "rows_per_second"]].to_numpy()
    require(np.isfinite(numeric_efficiency).all(), "效率结果含非有限值")
    require((efficiency["artifact_size_bytes"] > 0).all(), "模型产物大小不是正数")
    require((efficiency[["load_time_mean_seconds", "inference_time_mean_seconds"]].to_numpy() >= 0).all(), "加载或推理时间为负")

    recommendation = load_json(REPORT_DIR / "model_recommendation.json")
    required_recommendation = {
        "recommended_default_model", "recommended_artifact", "primary_reason",
        "secondary_reasons", "best_ensemble_model", "best_neural_model", "fallback_model",
        "streamlit_model_lab_models", "limitations", "selection_time",
    }
    require(required_recommendation.issubset(recommendation), "模型推荐字段不完整")
    require(recommendation["recommended_default_model"] in MODEL_NAMES, "推荐模型名称无效")
    require((ROOT / recommendation["recommended_artifact"]).is_file(), "推荐模型产物不存在")

    for name in ["metrics_comparison.html", "residual_analysis.html", "segment_comparison.html", "feature_importance_comparison.html", "efficiency_comparison.html"]:
        require((REPORT_DIR / name).stat().st_size > 100_000, f"Plotly图表过小或不完整：{name}")
    paper = (REPORT_DIR / "paper_results_material.md").read_text(encoding="utf-8")
    summary = (REPORT_DIR / "stage5_summary.md").read_text(encoding="utf-8")
    for token in ["RMSE =", "MAE =", "R² =", "随机划分", "Bootstrap", "局限"]:
        require(token in paper, f"论文素材缺少内容：{token}")
    require("第6阶段尚未开始" in summary, "阶段报告未说明第6阶段状态")

    source = (ROOT / "scripts" / "run_stage5.py").read_text(encoding="utf-8")
    for forbidden in [".fit(", "optimizer.step", ".backward("]:
        require(forbidden not in source, f"run_stage5.py疑似包含训练逻辑：{forbidden}")
    # 这里验证的是第5阶段历史边界：其训练/比较脚本不包含第6阶段逻辑。
    # 后续阶段出现 app.py 不应让已经完成的第5阶段复验失败。
    require(not (ROOT / "pages").exists(), "不得使用会与st.navigation冲突的根pages目录")

    best = metrics.sort_values("rmse").iloc[0]
    print(f"统一测试集：{len(unified)}条；模型数：{len(metrics)}")
    print(f"最低RMSE模型：{best['model']}，RMSE={best['rmse']:.6f}")
    print("统一指标、Bootstrap、分组误差、两类重要性、效率、推荐与源文件哈希：全部通过")
    print("第5阶段验证通过")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print(f"第5阶段验证失败：{error}", file=sys.stderr)
        sys.exit(1)
