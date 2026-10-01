"""执行第5阶段：复用既有模型与预测，完成统一对比和解释分析。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import torch
import xgboost
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from movie_rating.baseline import GlobalMeanBaseline
from movie_rating.comparison import (
    MODEL_NAMES,
    PREDICTION_COLUMNS,
    aggregate_ensemble_importance,
    benchmark_model_inference,
    build_metrics_comparison,
    build_unified_predictions,
    calculate_mlp_permutation_importance,
    calculate_residual_summary,
    calculate_segment_metrics,
    paired_bootstrap_compare,
)
from movie_rating.data import calculate_file_sha256, load_raw_data, write_json
from movie_rating.neural import MovieRatingMLP, load_mlp_artifacts, predict_ratings
from movie_rating.neural_data import NeuralFeaturePreprocessor, enrich_interactions


CONFIG_PATH = ROOT / "configs" / "stage5_comparison.json"
TRAIN_PATH = ROOT / "data" / "processed" / "train_ratings.csv"
TEST_PATH = ROOT / "data" / "processed" / "test_ratings.csv"
REPORT_DIR = ROOT / "reports" / "stage5"

SOURCE_ARTIFACTS = [
    TRAIN_PATH,
    TEST_PATH,
    ROOT / "models" / "global_mean_baseline.joblib",
    ROOT / "models" / "stage3" / "random_forest.joblib",
    ROOT / "models" / "stage3" / "adaboost.joblib",
    ROOT / "models" / "stage3" / "xgboost.joblib",
    ROOT / "models" / "stage3" / "best_ensemble.joblib",
    ROOT / "models" / "stage4" / "mlp_validation_best.pt",
    ROOT / "models" / "stage4" / "mlp_final.pt",
    ROOT / "models" / "stage4" / "validation_preprocessor.joblib",
    ROOT / "models" / "stage4" / "final_preprocessor.joblib",
    ROOT / "models" / "stage4" / "model_metadata.json",
    ROOT / "reports" / "stage2" / "test_predictions.csv",
    ROOT / "reports" / "stage3" / "test_predictions.csv",
    ROOT / "reports" / "stage3" / "feature_importance.csv",
    ROOT / "reports" / "stage4" / "test_predictions.csv",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """通过pandas JSON编码把NumPy标量转换为标准JSON类型。"""
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def run_prerequisites() -> None:
    for script_name in ["verify_stage2.py", "verify_stage3.py", "verify_stage4.py"]:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script_name)], cwd=ROOT,
            text=True, encoding="utf-8", errors="replace", capture_output=True, check=False,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"前置验证失败：{script_name}")


def source_hashes() -> dict[str, str]:
    result = {}
    for path in SOURCE_ARTIFACTS:
        if not path.is_file():
            raise FileNotFoundError(f"缺少源产物：{path.relative_to(ROOT)}")
        result[path.relative_to(ROOT).as_posix()] = calculate_file_sha256(path)
    return result


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    labels = [str(column) for column in columns]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
    for row in frame[columns].to_dict("records"):
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("—")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def validate_against_prior_reports(metrics: pd.DataFrame) -> None:
    indexed = metrics.set_index("model")
    stage2 = load_json(ROOT / "reports" / "stage2" / "baseline_metrics.json")["test_metrics"]
    stage3 = load_json(ROOT / "reports" / "stage3" / "test_metrics.json")
    stage4 = load_json(ROOT / "reports" / "stage4" / "test_metrics.json")
    expected = {"global_mean": stage2, **stage3, "mlp": stage4}
    for model, report in expected.items():
        for metric in ["rmse", "mae", "r2"]:
            if not np.isclose(indexed.loc[model, metric], report[metric], rtol=0, atol=2e-8):
                raise ValueError(f"{model}的{metric}与前阶段报告不一致")


def build_feature_group_comparison(
    ensemble_group: pd.DataFrame,
    mlp_importance: pd.DataFrame,
) -> pd.DataFrame:
    ensemble = ensemble_group.rename(columns={"importance": "raw_importance"}).copy()
    ensemble["model_family"] = "ensemble"
    ensemble["importance_method"] = "native_tree_split_importance"
    ensemble["normalized_importance"] = ensemble["raw_importance"]
    ensemble["evaluation_data"] = "完整80000条训练集拟合后的模型"
    mlp = mlp_importance.rename(
        columns={
            "rmse_increase_mean": "raw_importance",
            "positive_normalized_importance": "normalized_importance",
        }
    ).copy()
    mlp["model"] = "mlp"
    mlp["model_family"] = "neural_network"
    mlp["importance_method"] = "grouped_permutation_rmse_increase"
    mlp["evaluation_data"] = "固定内部验证集8000条"
    columns = [
        "model", "model_family", "importance_method", "feature_group",
        "raw_importance", "normalized_importance", "rank", "evaluation_data",
    ]
    return pd.concat([ensemble[columns], mlp[columns]], ignore_index=True)


def save_metric_chart(metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("rmse")
    fig = make_subplots(rows=2, cols=2, subplot_titles=("RMSE（越低越好）", "MAE（越低越好）", "R²（越高越好）", "相对基线RMSE改善率"))
    for column, row, col, color in [
        ("rmse", 1, 1, "#2563eb"), ("mae", 1, 2, "#0891b2"),
        ("r2", 2, 1, "#16a34a"), ("rmse_improvement_vs_baseline_percent", 2, 2, "#9333ea"),
    ]:
        fig.add_trace(go.Bar(x=ordered["model"], y=ordered[column], marker_color=color, name=column, text=ordered[column].round(4), textposition="outside"), row=row, col=col)
    fig.update_layout(height=850, title="固定测试集五模型统一对比", showlegend=False, template="plotly_white")
    fig.write_html(REPORT_DIR / "metrics_comparison.html", include_plotlyjs=True, full_html=True)


def save_residual_chart(unified: pd.DataFrame, rating_diagnostics: pd.DataFrame) -> None:
    fig = make_subplots(rows=2, cols=2, subplot_titles=("误差分布", "绝对误差箱线图", "真实评分与平均预测", "真实评分与平均误差"))
    actual = unified["actual_rating"].to_numpy()
    for model in MODEL_NAMES:
        error = unified[PREDICTION_COLUMNS[model]].to_numpy() - actual
        fig.add_trace(go.Histogram(x=error, name=model, opacity=0.5, nbinsx=70, histnorm="probability density"), row=1, col=1)
        fig.add_trace(go.Box(y=np.abs(error), name=model, boxpoints=False, showlegend=False), row=1, col=2)
        diagnostic = rating_diagnostics[rating_diagnostics["model"] == model]
        fig.add_trace(go.Scatter(x=diagnostic["actual_rating"], y=diagnostic["mean_prediction"], mode="lines+markers", name=model, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=diagnostic["actual_rating"], y=diagnostic["mean_error"], mode="lines+markers", name=model, showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[1, 5], y=[1, 5], mode="lines", line={"dash": "dash", "color": "black"}, name="理想预测", showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=2)
    fig.update_layout(height=900, title="残差与评分等级诊断", barmode="overlay", template="plotly_white")
    fig.write_html(REPORT_DIR / "residual_analysis.html", include_plotlyjs=True, full_html=True)


def save_segment_chart(segment: pd.DataFrame) -> None:
    types = ["cold_start", "user_activity", "movie_popularity", "actual_rating"]
    titles = ["冷启动RMSE", "用户活跃度RMSE", "电影流行度RMSE", "真实评分分组MAE"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)
    for index, segment_type in enumerate(types):
        row, col = index // 2 + 1, index % 2 + 1
        part = segment[segment["segment_type"] == segment_type]
        value_column = "mae" if segment_type == "actual_rating" else "rmse"
        for model in MODEL_NAMES:
            group = part[part["model"] == model]
            fig.add_trace(go.Bar(x=group["segment_value"], y=group[value_column], name=model, legendgroup=model, showlegend=index == 0), row=row, col=col)
    fig.update_layout(height=900, title="分组误差对比", barmode="group", template="plotly_white")
    fig.write_html(REPORT_DIR / "segment_comparison.html", include_plotlyjs=True, full_html=True)


def save_importance_chart(
    detailed: pd.DataFrame,
    grouped: pd.DataFrame,
    mlp_importance: pd.DataFrame,
) -> None:
    fig = make_subplots(rows=2, cols=2, subplot_titles=("随机森林前20特征", "XGBoost前20特征", "集成模型分组原生重要性", "MLP验证集分组排列重要性"))
    for col, model in [(1, "random_forest"), (2, "xgboost")]:
        top = detailed[detailed["model"] == model].nsmallest(20, "rank").sort_values("normalized_importance")
        fig.add_trace(go.Bar(x=top["normalized_importance"], y=top["feature"], orientation="h", name=model, showlegend=False), row=1, col=col)
    for model in ["random_forest", "adaboost", "xgboost"]:
        group = grouped[grouped["model"] == model]
        fig.add_trace(go.Bar(x=group["feature_group"], y=group["importance"], name=model), row=2, col=1)
    mlp_sorted = mlp_importance.sort_values("rmse_increase_mean")
    fig.add_trace(go.Bar(x=mlp_sorted["rmse_increase_mean"], y=mlp_sorted["feature_group"], orientation="h", error_x={"type": "data", "array": mlp_sorted["rmse_increase_std"]}, name="MLP排列重要性", showlegend=False), row=2, col=2)
    fig.update_layout(height=1100, title="特征重要性对比（两类方法数值不可直接等比例比较）", barmode="group", template="plotly_white")
    fig.write_html(REPORT_DIR / "feature_importance_comparison.html", include_plotlyjs=True, full_html=True)


def save_efficiency_chart(efficiency: pd.DataFrame, metrics: pd.DataFrame) -> None:
    merged = efficiency.merge(metrics[["model", "rmse"]], on="model", validate="one_to_one")
    fig = make_subplots(rows=2, cols=2, subplot_titles=("模型产物大小（MB）", "加载时间（秒）", "20000条端到端推理时间（秒）", "RMSE—推理速度权衡"))
    for column, row, col in [
        ("artifact_size_mb", 1, 1), ("load_time_mean_seconds", 1, 2),
        ("inference_time_mean_seconds", 2, 1),
    ]:
        fig.add_trace(go.Bar(x=merged["model"], y=merged[column], text=merged[column].round(4), textposition="outside", showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=merged["inference_time_mean_seconds"], y=merged["rmse"], text=merged["model"], mode="markers+text", textposition="top center", marker={"size": 14, "color": merged["artifact_size_mb"], "colorscale": "Viridis", "showscale": True}), row=2, col=2)
    fig.update_layout(height=850, title="同一CPU环境下的效率与精度", template="plotly_white")
    fig.write_html(REPORT_DIR / "efficiency_comparison.html", include_plotlyjs=True, full_html=True)


def build_reports(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    segment: pd.DataFrame,
    residual: pd.DataFrame,
    ensemble_importance: pd.DataFrame,
    mlp_importance: pd.DataFrame,
    efficiency: pd.DataFrame,
    recommendation: dict[str, Any],
    segmentation: dict[str, Any],
) -> tuple[str, str]:
    metric_table = markdown_table(metrics, ["model", "rmse", "mae", "r2", "rmse_improvement_vs_baseline_percent", "rmse_rank"])
    bootstrap_table = markdown_table(bootstrap, ["comparison", "metric", "point_estimate_delta", "confidence_lower", "confidence_upper", "mlp_win_probability"])
    efficiency_table = markdown_table(efficiency, ["model", "artifact_size_mb", "load_time_mean_seconds", "inference_time_mean_seconds", "rows_per_second"])
    top_tree = ensemble_importance.sort_values(["model", "rank"]).groupby("model", as_index=False).first()
    top_mlp = mlp_importance.sort_values("rank").head(5)
    cold = segment[(segment["segment_type"] == "cold_start") & (segment["model"] == "mlp")]
    diagnostics = residual.set_index("model")
    best = metrics.sort_values("rmse").iloc[0]
    rf = metrics.set_index("model").loc["random_forest"]
    paper = f"""# 第5阶段论文结果素材

## 1. 实验目的与环境

在MovieLens 100K上比较全局均值、随机森林、AdaBoost、XGBoost和PyTorch MLP。环境为Windows、Python {sys.version.split()[0]}、scikit-learn {sklearn.__version__}、XGBoost {xgboost.__version__}、PyTorch {torch.__version__}，全部效率测试使用CPU。

## 2. 数据划分与公平性

100000条评分按评分值分层随机划分为80000条训练记录和20000条固定测试记录，随机种子42。三种集成模型的选择来自训练集内部5折交叉验证；MLP的最佳epoch来自8000条内部验证集。固定测试集只用于最终指标复算、误差分析和本阶段工程比较，未用于训练或超参数调整。

本实验采用随机划分而非时间顺序划分。树模型使用包含训练历史平滑统计的特征，MLP使用Embedding、静态属性、电影类型和时间特征，因此结论是两套完整建模路线的比较，而不是仅隔离算法差异的控制实验。

## 3. 评价指标

RMSE = sqrt((1/n) × Σ(y_i - ŷ_i)²)

MAE = (1/n) × Σ|y_i - ŷ_i|

R² = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²

其中，y_i为真实评分，ŷ_i为预测评分，ȳ为测试集真实评分均值。RMSE和MAE越低越好，R²越高越好。

## 4. 统一测试结果

{metric_table}

测试RMSE最低的是{best['model']}（{best['rmse']:.6f}）。MLP相对随机森林的RMSE降低{(rf['rmse'] - metrics.set_index('model').loc['mlp', 'rmse']) / rf['rmse'] * 100:.2f}%。

## 5. 成对Bootstrap

对相同的20000条测试记录进行2000次有放回成对重采样。每次使用同一组抽样索引同时计算MLP与集成模型指标，从而保留同一评分记录上的误差相关性。差异定义为MLP减集成模型，负值表示MLP更优。

{bootstrap_table}

置信区间描述本测试集上的差异稳定性，不代表因果关系，也不使用“完全证明”等过度结论。

## 6. 分组、冷启动与残差

用户活跃度三分位阈值为{segmentation['user_activity_thresholds']}；电影流行度阈值为{segmentation['movie_popularity_thresholds']}，均只由80000条训练记录计算。固定测试集含{segmentation['unseen_test_movie_count']}部未见电影、{segmentation['unseen_movie_test_rows']}条未见电影记录。

MLP冷启动分组结果：

{markdown_table(cold, ['segment_value', 'sample_count', 'rmse', 'mae', 'mean_error'])}

全局均值模型表现出明显向均值收缩；其他模型也通常高估低评分、低估高评分。MLP整体平均误差为{diagnostics.loc['mlp', 'mean_error']:.6f}，90%绝对误差分位数为{diagnostics.loc['mlp', 'p90_absolute_error']:.6f}。

## 7. 特征重要性

集成模型的最高重要性特征如下：

{markdown_table(top_tree, ['model', 'feature', 'feature_group', 'normalized_importance'])}

MLP验证集分组排列重要性前五项：

{markdown_table(top_mlp, ['feature_group', 'rmse_increase_mean', 'rmse_increase_std', 'positive_normalized_importance'])}

树模型数值来自分裂重要性，MLP数值来自验证集分组排列后的RMSE增量。两种方法可以比较各模型内部排名和信息类型，但绝对数值不可直接等比例比较，也不代表因果影响。

## 8. 模型大小与推理效率

{efficiency_table}

效率仅代表本机同一CPU环境下的相对结果，端到端推理包含必要特征转换但不含CSV读取和指标计算。

## 9. 推荐与局限

第6阶段建议默认使用 **{recommendation['recommended_default_model']}**，最佳集成对照保留 **{recommendation['best_ensemble_model']}**，全局均值作为加载失败时的兜底。该工程建议综合使用固定测试集表现、文件大小和本机推理效率；同一测试集不能再用于后续调参或再次声称无偏评估。

MovieLens 100K规模较小，采用随机划分，且模型特征路线不同。未见用户样本在本测试集中为0，无法充分评价新用户冷启动。更严格研究可使用时间顺序划分、嵌套验证或外部数据集。

## 10. 可直接用于论文的结果分析段落

在相同的20000条固定测试记录上，{best['model']}取得最低RMSE。成对Bootstrap进一步量化了MLP与三种集成模型在同一批样本上的误差差异。分组分析表明，低流行度和未见电影仍是主要困难场景；残差分析显示各模型均存在不同程度的均值收缩。树模型主要依赖训练历史统计，而MLP通过用户、电影Embedding及辅助属性学习非线性表示。综合精度、模型体积和本机推理效率，本研究选择{recommendation['recommended_default_model']}作为网页默认模型，同时保留随机森林和XGBoost用于模型实验室对照。
"""
    summary = f"""# 第5阶段：集成学习与神经网络对比实验

## 阶段结论

前置第2、3、4阶段验证全部通过。本阶段未重新训练任何模型，在相同20000条固定测试记录上统一复算指标。

{metric_table}

测试集排名第一为 **{best['model']}**。第3阶段按训练集内部5折CV选择的最佳集成模型仍为 **random_forest**；第4阶段神经网络为 **mlp**。

## 成对Bootstrap

{bootstrap_table}

差异定义为MLP减集成模型，负值表示MLP更好。置信区间不跨0时，只表述为本测试集上的差异较稳定。

## 分组与残差

- 未见电影：{segmentation['unseen_test_movie_count']}部、{segmentation['unseen_movie_test_rows']}条测试记录。
- MLP整体平均误差：{diagnostics.loc['mlp', 'mean_error']:.6f}。
- MLP近似命中率（绝对误差≤0.5）：{diagnostics.loc['mlp', 'exact_or_near_rate']:.2%}。
- 用户活跃度、电影流行度阈值只由训练集计算，未使用测试评分次数。

## 重要性说明

集成模型使用原生树分裂重要性；MLP使用8000条内部验证集的分组排列重要性。两类数值不可直接等比例比较，也不代表因果关系。

MLP排列重要性前五项：

{markdown_table(top_mlp, ['feature_group', 'rmse_increase_mean', 'rmse_increase_std', 'rank'])}

## 效率和第6阶段建议

{efficiency_table}

建议第6阶段默认模型：**{recommendation['recommended_default_model']}**；最佳集成对照：**{recommendation['best_ensemble_model']}**；故障兜底：**{recommendation['fallback_model']}**。该建议使用测试结果做工程选择，后续不得继续使用同一测试集调参。

## 局限

数据规模较小、使用随机划分而非时间划分、树模型与MLP特征路线不同，且没有未见用户测试样本。结果适用于本课程实验范围。

## 主要产物

- `metrics_comparison.*`：统一指标；
- `paired_bootstrap_summary.*`：成对Bootstrap；
- `segment_metrics.csv`、`residual_summary.csv`：误差分析；
- `ensemble_*importance.csv`、`mlp_permutation_importance*.csv`：解释性；
- `efficiency_benchmark.csv`：本机CPU效率；
- 五个Plotly HTML：论文和答辩图表；
- `paper_results_material.md`：论文结果素材。

第6阶段尚未开始。
"""
    return paper, summary


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/10] 运行第2、3、4阶段前置验证")
    run_prerequisites()
    print("[1/10] 前置验证通过")
    config = load_json(CONFIG_PATH)
    before_hashes = source_hashes()

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    if len(train) != 80_000 or len(test) != 20_000:
        raise ValueError("固定训练/测试规模异常")
    stage2_predictions = pd.read_csv(ROOT / "reports" / "stage2" / "test_predictions.csv", encoding="utf-8-sig")
    stage3_predictions = pd.read_csv(ROOT / "reports" / "stage3" / "test_predictions.csv", encoding="utf-8-sig")
    stage4_predictions = pd.read_csv(ROOT / "reports" / "stage4" / "test_predictions.csv", encoding="utf-8-sig")
    unified, segmentation = build_unified_predictions(
        test, train, stage2_predictions, stage3_predictions, stage4_predictions
    )
    unified.to_csv(REPORT_DIR / "unified_test_predictions.csv", index=False, encoding="utf-8-sig")
    write_json(segmentation, REPORT_DIR / "segmentation_metadata.json")
    print(f"[2/10] 统一预测表：{len(unified)}条")

    metrics = build_metrics_comparison(unified)
    validate_against_prior_reports(metrics)
    metrics.to_csv(REPORT_DIR / "metrics_comparison.csv", index=False, encoding="utf-8-sig")
    write_json({"evaluation_data": "固定测试集20000条", "models": frame_records(metrics)}, REPORT_DIR / "metrics_comparison.json")
    print("[3/10] 五模型测试指标复算完成")

    prediction_arrays = {name: unified[PREDICTION_COLUMNS[name]].to_numpy() for name in MODEL_NAMES}
    bootstrap = paired_bootstrap_compare(
        unified["actual_rating"].to_numpy(), prediction_arrays,
        ["random_forest", "adaboost", "xgboost"],
        config["bootstrap_resamples"], config["confidence_level"], config["random_seed"],
    )
    bootstrap.to_csv(REPORT_DIR / "paired_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    write_json({"delta_definition": "MLP指标减集成模型指标，负值表示MLP更好", "results": frame_records(bootstrap)}, REPORT_DIR / "paired_bootstrap_summary.json")
    print(f"[4/10] 成对Bootstrap完成：{config['bootstrap_resamples']}次")

    segment = calculate_segment_metrics(unified)
    residual, rating_diagnostics = calculate_residual_summary(unified)
    segment.to_csv(REPORT_DIR / "segment_metrics.csv", index=False, encoding="utf-8-sig")
    residual.to_csv(REPORT_DIR / "residual_summary.csv", index=False, encoding="utf-8-sig")
    rating_diagnostics.to_csv(REPORT_DIR / "actual_rating_diagnostics.csv", index=False, encoding="utf-8-sig")
    print("[5/10] 分组误差和残差分析完成")

    raw_importance = pd.read_csv(ROOT / "reports" / "stage3" / "feature_importance.csv", encoding="utf-8-sig")
    detailed_importance, grouped_importance = aggregate_ensemble_importance(raw_importance)
    detailed_importance.to_csv(REPORT_DIR / "ensemble_feature_importance.csv", index=False, encoding="utf-8-sig")
    grouped_importance.to_csv(REPORT_DIR / "ensemble_group_importance.csv", index=False, encoding="utf-8-sig")

    stage4_config = load_json(ROOT / "configs" / "stage4_mlp.json")
    internal_train, validation = train_test_split(
        train, test_size=0.10, random_state=42, stratify=train["rating"], shuffle=True
    )
    internal_train = internal_train.sort_values("interaction_id").reset_index(drop=True)
    validation = validation.sort_values("interaction_id").reset_index(drop=True)
    raw = load_raw_data(ROOT / "data" / "raw")
    validation_enriched = enrich_interactions(validation, raw.users, raw.movies)
    validation_preprocessor = NeuralFeaturePreprocessor.load(
        ROOT / "models" / "stage4" / "validation_preprocessor.joblib"
    )
    validation_features = validation_preprocessor.transform(validation_enriched)
    validation_model = MovieRatingMLP(
        vocabulary_sizes=validation_preprocessor.vocabulary_sizes,
        numeric_feature_count=validation_preprocessor.numeric_feature_count,
        genre_feature_count=validation_preprocessor.genre_feature_count,
        user_embedding_dim=stage4_config["user_embedding_dim"],
        movie_embedding_dim=stage4_config["movie_embedding_dim"],
        occupation_embedding_dim=stage4_config["occupation_embedding_dim"],
        gender_embedding_dim=stage4_config["gender_embedding_dim"],
        hidden_dims=stage4_config["hidden_dims"],
        dropout=stage4_config["dropout"],
        rating_global_mean=float(internal_train["rating"].mean()),
    )
    validation_state = torch.load(
        ROOT / "models" / "stage4" / "mlp_validation_best.pt",
        map_location="cpu", weights_only=True,
    )
    validation_model.load_state_dict(validation_state)
    validation_model.eval()
    permutation_repeats, mlp_importance = calculate_mlp_permutation_importance(
        validation_model, validation_features, validation["rating"].to_numpy(),
        config["permutation_repeats"], config["random_seed"], stage4_config["batch_size"],
    )
    expected_validation_rmse = load_json(ROOT / "reports" / "stage4" / "validation_metrics.json")["rmse"]
    actual_validation_rmse = float(mlp_importance["baseline_rmse"].iloc[0])
    if not np.isclose(actual_validation_rmse, expected_validation_rmse, rtol=0, atol=2e-8):
        raise ValueError("MLP排列重要性的基础验证RMSE与第4阶段不一致")
    permutation_repeats.to_csv(REPORT_DIR / "mlp_permutation_importance_repeats.csv", index=False, encoding="utf-8-sig")
    mlp_importance.to_csv(REPORT_DIR / "mlp_permutation_importance.csv", index=False, encoding="utf-8-sig")
    feature_comparison = build_feature_group_comparison(grouped_importance, mlp_importance)
    feature_comparison.to_csv(REPORT_DIR / "feature_group_comparison.csv", index=False, encoding="utf-8-sig")
    print("[6/10] 集成模型原生重要性和MLP验证集排列重要性完成")

    test_enriched = enrich_interactions(test, raw.users, raw.movies)
    ensemble_test = test_enriched.drop(columns=["rating"])
    warmups = config["benchmark_warmup_runs"]
    repeats = config["benchmark_repeats"]
    rows = []
    baseline_path = ROOT / "models" / "global_mean_baseline.joblib"
    rows.append(benchmark_model_inference(
        "global_mean", [baseline_path], lambda: GlobalMeanBaseline.load(baseline_path),
        lambda model: model.predict(len(ensemble_test)), len(test), warmups, repeats,
    ))
    for model_name in ["random_forest", "adaboost", "xgboost"]:
        model_path = ROOT / "models" / "stage3" / f"{model_name}.joblib"
        rows.append(benchmark_model_inference(
            model_name, [model_path], lambda path=model_path: joblib.load(path),
            lambda model: model.predict(ensemble_test), len(test), warmups, repeats,
        ))
    mlp_paths = [
        ROOT / "models" / "stage4" / "mlp_final.pt",
        ROOT / "models" / "stage4" / "final_preprocessor.joblib",
        ROOT / "models" / "stage4" / "model_metadata.json",
    ]
    rows.append(benchmark_model_inference(
        "mlp", mlp_paths,
        lambda: load_mlp_artifacts(*mlp_paths, device="cpu"),
        lambda bundle: predict_ratings(bundle[0], bundle[1], ensemble_test, device="cpu", batch_size=stage4_config["batch_size"]),
        len(test), warmups, repeats,
    ))
    efficiency = pd.DataFrame(rows)
    efficiency.to_csv(REPORT_DIR / "efficiency_benchmark.csv", index=False, encoding="utf-8-sig")
    environment = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "torch_version": torch.__version__,
        "os": platform.platform(),
        "cpu_logical_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "timing_scope": "不含CSV读取；端到端推理含必要特征转换",
    }
    write_json(environment, REPORT_DIR / "efficiency_environment.json")
    print("[7/10] 同一CPU环境效率基准完成")

    ranked = metrics.sort_values("rmse").reset_index(drop=True)
    recommended = str(ranked.iloc[0]["model"])
    if float(ranked.iloc[1]["rmse"] - ranked.iloc[0]["rmse"]) < 0.005:
        candidates = efficiency[efficiency["model"].isin(ranked.iloc[:2]["model"])]
        recommended = str(candidates.sort_values(["inference_time_mean_seconds", "artifact_size_bytes"]).iloc[0]["model"])
    artifact_map = {
        "mlp": "models/stage4/mlp_final.pt",
        "random_forest": "models/stage3/random_forest.joblib",
        "adaboost": "models/stage3/adaboost.joblib",
        "xgboost": "models/stage3/xgboost.joblib",
        "global_mean": "models/global_mean_baseline.joblib",
    }
    recommendation = {
        "recommended_default_model": recommended,
        "recommended_artifact": artifact_map[recommended],
        "primary_reason": "固定测试集RMSE最低；若差异小于0.005则进一步比较本机推理时间和文件大小",
        "secondary_reasons": ["同时复核MAE与R²", "考虑CPU端到端推理效率", "考虑Streamlit部署产物大小"],
        "best_ensemble_model": "random_forest",
        "best_neural_model": "mlp",
        "fallback_model": "global_mean",
        "streamlit_model_lab_models": ["global_mean", "random_forest", "xgboost", "mlp"],
        "limitations": "工程推荐使用了固定测试集结果，不能继续使用同一测试集调参；未来泛化需新验证数据或嵌套评估。",
        "selection_time": datetime.now(timezone.utc).isoformat(),
    }
    write_json(recommendation, REPORT_DIR / "model_recommendation.json")
    print(f"[8/10] 第6阶段工程建议模型：{recommended}")

    save_metric_chart(metrics)
    save_residual_chart(unified, rating_diagnostics)
    save_segment_chart(segment)
    save_importance_chart(detailed_importance, grouped_importance, mlp_importance)
    save_efficiency_chart(efficiency, metrics)
    print("[9/10] 五个Plotly图表生成完成")

    paper, summary = build_reports(
        metrics, bootstrap, segment, residual, detailed_importance,
        mlp_importance, efficiency, recommendation, segmentation,
    )
    (REPORT_DIR / "paper_results_material.md").write_text(paper, encoding="utf-8")
    (REPORT_DIR / "stage5_summary.md").write_text(summary, encoding="utf-8")

    after_hashes = source_hashes()
    hash_rows = []
    for path, before in before_hashes.items():
        after = after_hashes[path]
        hash_rows.append({"path": path, "before_sha256": before, "after_sha256": after, "unchanged": before == after})
    if not all(row["unchanged"] for row in hash_rows):
        raise RuntimeError("第5阶段执行期间源模型、预测或固定数据文件被修改")
    write_json({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "all_unchanged": True,
        "artifacts": hash_rows,
    }, REPORT_DIR / "source_artifact_hashes.json")
    print("[10/10] 源文件哈希复核通过；论文素材和阶段报告完成")
    print("\n=== 第5阶段结果摘要 ===")
    print(metrics[["model", "rmse", "mae", "r2", "rmse_rank"]].to_string(index=False))
    print(f"推荐第6阶段默认模型：{recommended}")
    print("第5阶段主程序执行完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
