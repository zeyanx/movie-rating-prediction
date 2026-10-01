"""执行第3阶段：无泄漏特征工程、三种集成模型与5折交叉验证。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from movie_rating.baseline import evaluate_predictions  # noqa: E402
from movie_rating.data import calculate_file_sha256, load_raw_data, validate_raw_data, write_json  # noqa: E402
from movie_rating.ensemble import (  # noqa: E402
    build_model_pipelines,
    evaluate_cv,
    extract_feature_importance,
    fit_final_models,
    summarize_cv,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "stage3_models.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "stage3"
REPORT_DIR = PROJECT_ROOT / "reports" / "stage3"
STAGE2_MANIFEST = PROCESSED_DIR / "split_manifest.json"
STAGE2_BASELINE = PROJECT_ROOT / "reports" / "stage2" / "baseline_metrics.json"
MODEL_ORDER = ["random_forest", "adaboost", "xgboost"]


def verify_stage2() -> None:
    """开始训练前使用当前解释器复验固定划分。"""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_stage2.py")],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0 or "第2阶段验证通过" not in result.stdout:
        raise RuntimeError("第2阶段复验失败：\n" + result.stdout + "\n" + result.stderr)
    print("第2阶段复验：通过")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fixed_interactions() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """读取固定划分，校验哈希，并与只读用户/电影元数据合并。"""
    manifest = load_json(STAGE2_MANIFEST)
    train_path = PROCESSED_DIR / "train_ratings.csv"
    test_path = PROCESSED_DIR / "test_ratings.csv"
    if calculate_file_sha256(train_path) != manifest["train_sha256"]:
        raise ValueError("固定训练集SHA-256已变化")
    if calculate_file_sha256(test_path) != manifest["test_sha256"]:
        raise ValueError("固定测试集SHA-256已变化")

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    if len(train) != 80_000 or len(test) != 20_000:
        raise ValueError("固定训练集或测试集规模异常")

    raw = load_raw_data(PROJECT_ROOT / "data" / "raw")
    validate_raw_data(raw)

    def enrich(interactions: pd.DataFrame) -> pd.DataFrame:
        merged = interactions.merge(raw.users, on="user_id", how="left", validate="many_to_one")
        merged = merged.merge(raw.movies, on="movie_id", how="left", validate="many_to_one")
        if len(merged) != len(interactions):
            raise ValueError("元数据合并改变了交互记录数")
        return merged

    train_merged = enrich(train)
    test_merged = enrich(test)
    y_train = train_merged.pop("rating").astype(float)
    y_test = test_merged.pop("rating").astype(float)
    return train_merged, y_train, test_merged, y_test


def prior_core_results() -> dict[str, pd.DataFrame] | None:
    """读取上次运行的核心指标，用于第二次运行时确认可重复性。"""
    cv_path = REPORT_DIR / "cv_summary.csv"
    test_path = REPORT_DIR / "test_metrics.csv"
    if not cv_path.is_file() or not test_path.is_file():
        return None
    cv = pd.read_csv(cv_path)
    test = pd.read_csv(test_path)
    cv_columns = ["model", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std"]
    test_columns = ["model", "rmse", "mae", "r2", "prediction_min", "prediction_max"]
    return {"cv": cv[cv_columns].sort_values("model").reset_index(drop=True),
            "test": test[test_columns].sort_values("model").reset_index(drop=True)}


def compare_core_results(
    previous: dict[str, pd.DataFrame] | None,
    cv_summary: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """忽略运行时间，比较两次运行的确定性指标。"""
    if previous is None:
        return {"previous_results_found": False, "core_metrics_match": None}
    current_cv = cv_summary[
        ["model", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std"]
    ].sort_values("model").reset_index(drop=True)
    current_test = test_metrics[
        ["model", "rmse", "mae", "r2", "prediction_min", "prediction_max"]
    ].sort_values("model").reset_index(drop=True)
    names_match = previous["cv"]["model"].equals(current_cv["model"]) and previous["test"]["model"].equals(current_test["model"])
    cv_match = np.allclose(
        previous["cv"].drop(columns="model"),
        current_cv.drop(columns="model"),
        rtol=0.0,
        atol=1e-10,
    )
    test_match = np.allclose(
        previous["test"].drop(columns="model"),
        current_test.drop(columns="model"),
        rtol=0.0,
        atol=1e-10,
    )
    return {
        "previous_results_found": True,
        "core_metrics_match": bool(names_match and cv_match and test_match),
    }


def save_charts(
    cv_summary: pd.DataFrame,
    test_metrics: pd.DataFrame,
    importance: pd.DataFrame,
    selected_model: str,
) -> None:
    """生成三个可离线打开的中文Plotly HTML图表。"""
    cv_plot = px.bar(
        cv_summary,
        x="model",
        y="rmse_mean",
        error_y="rmse_std",
        title="三种集成模型的5折交叉验证RMSE",
        labels={"model": "模型", "rmse_mean": "验证RMSE均值"},
        text_auto=".4f",
    )
    cv_plot.write_html(REPORT_DIR / "cv_rmse_comparison.html", include_plotlyjs=True)

    metric_long = test_metrics[["model", "rmse", "mae", "r2"]].melt(
        id_vars="model", var_name="metric", value_name="value"
    )
    test_plot = px.bar(
        metric_long,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        title="固定测试集指标对比",
        labels={"model": "模型", "value": "指标值", "metric": "指标"},
    )
    test_plot.write_html(REPORT_DIR / "test_metrics_comparison.html", include_plotlyjs=True)

    top = importance[importance["model"] == selected_model].nsmallest(20, "rank").sort_values(
        "normalized_importance"
    )
    importance_plot = px.bar(
        top,
        x="normalized_importance",
        y="feature",
        orientation="h",
        color="feature_group",
        title=f"最佳集成模型 {selected_model} 的前20项特征重要性",
        labels={
            "normalized_importance": "归一化重要性",
            "feature": "特征",
            "feature_group": "特征分组",
        },
    )
    importance_plot.write_html(REPORT_DIR / "feature_importance.html", include_plotlyjs=True)


def build_summary(
    config: dict[str, Any],
    cv_summary: pd.DataFrame,
    test_metrics: pd.DataFrame,
    importance: pd.DataFrame,
    selected: dict[str, Any],
    feature_count: int,
    baseline_rmse: float,
) -> str:
    """用实际计算结果生成可直接用于论文初稿的第3阶段报告。"""
    cv_rows = "\n".join(
        f"| {row.model} | {row.rmse_mean:.6f} ± {row.rmse_std:.6f} | "
        f"{row.mae_mean:.6f} ± {row.mae_std:.6f} | {row.r2_mean:.6f} ± {row.r2_std:.6f} |"
        for row in cv_summary.itertuples()
    )
    test_rows = "\n".join(
        f"| {row.model} | {row.rmse:.6f} | {row.mae:.6f} | {row.r2:.6f} | "
        f"{(baseline_rmse-row.rmse)/baseline_rmse*100:.2f}% |"
        for row in test_metrics.itertuples()
    )
    top_tables = []
    for name in MODEL_ORDER:
        top = importance[importance["model"] == name].nsmallest(10, "rank")
        lines = [f"### {name}", "", "| 排名 | 特征 | 分组 | 归一化重要性 |", "|---:|---|---|---:|"]
        lines.extend(
            f"| {int(row.rank)} | {row.feature} | {row.feature_group} | {row.normalized_importance:.6f} |"
            for row in top.itertuples()
        )
        top_tables.append("\n".join(lines))
    importance_text = "\n\n".join(top_tables)
    model_config = json.dumps(config["models"], ensure_ascii=False, indent=2)

    return rf"""# 第3阶段：特征工程与集成学习模型

## 1. 实验目标与数据划分

本阶段在第2阶段固定的80000条训练记录上构造无泄漏特征，对随机森林、AdaBoost和XGBoost进行5折交叉验证，再用交叉验证RMSE选择最佳集成模型。20000条固定测试记录只用于三个最终模型的一次性评估，没有参与特征参数学习、交叉验证或模型选择。

## 2. 特征工程

最终完整训练集Pipeline产生 **{feature_count}** 个数值特征。

| 特征组 | 主要内容 | 设计理由 |
|---|---|---|
| 用户属性 | 年龄、性别独热、职业独热 | 表达不同用户群体的评分差异 |
| 电影类型 | 19类电影多标签特征 | 表达影片内容类型 |
| 时间 | 上映年、评分年/月/星期/小时、影片年龄 | 表达年代和评分时间影响 |
| 训练统计 | 用户/电影平滑均值、对数计数、均值差 | 表达偏好与受欢迎程度 |

`user_id` 和 `movie_id` 是任意编号，直接作为连续数值会产生不存在的大小与距离关系，因此本阶段只将它们作为训练统计映射的键。`interaction_id`、评分目标、标题、IMDb链接和邮政编码均未进入特征矩阵。

## 3. 平滑统计与逐行留一

对验证集或测试集，用户平滑均值为：

$$
\mu_u = \frac{{S_u + \alpha_u\mu}}{{N_u+\alpha_u}}
$$

其中用户平滑系数为{config['feature_engineering']['user_smoothing']:.0f}，电影平滑系数为{config['feature_engineering']['movie_smoothing']:.0f}。未见用户或电影回退到当前训练部分的全局均值，对应计数为0。

对训练记录$j$，使用逐行留一统计：

$$
\mu_{{u,-j}} = \frac{{S_u-r_j+\alpha_u\mu_{{-j}}}}{{N_u-1+\alpha_u}}
$$

这保证当前记录自己的评分不会进入其目标统计特征。每一折Pipeline只在折内训练部分拟合类别、填充值和统计映射，再转换验证折，从而避免交叉验证泄漏。

## 4. 三种模型

- **随机森林**：通过多棵随机化决策树平均降低方差，具有稳定、可解释的重要性；缺点是模型较大、训练和预测成本较高。
- **AdaBoost**：依次关注前序弱学习器误差，以浅层回归树提升拟合能力；优点是结构清晰，缺点是对异常点和参数较敏感。
- **XGBoost**：采用梯度提升、子采样和L1/L2正则化逐步优化残差，通常具有较强预测能力；缺点是参数较多且需要控制复杂度。

固定参数如下：

```json
{model_config}
```

## 5. 五折交叉验证结果

使用 `KFold(n_splits=5, shuffle=True, random_state=42)`，每条训练记录恰好作为验证样本一次。最佳模型仅按平均验证RMSE选择。

| 模型 | CV RMSE | CV MAE | CV R² |
|---|---:|---:|---:|
{cv_rows}

## 6. 固定测试集结果

第2阶段全局均值基线测试RMSE为{baseline_rmse:.6f}。

| 模型 | RMSE | MAE | R² | 相对基线RMSE改善 |
|---|---:|---:|---:|---:|
{test_rows}

根据平均5折CV RMSE，选择 **{selected['selected_model']}** 作为最佳集成模型，其CV RMSE为{selected['selected_cv_rmse']:.6f}。该决定在查看测试指标之前完成，测试集结果不参与选择。

## 7. 特征重要性

以下重要性来自各模型在完整训练集上拟合后的原生 `feature_importances_`，归一化后各模型总和为1。重要性表示模型分裂贡献，不直接等价于因果影响。

{importance_text}

## 8. 冷启动处理与局限性

固定测试集中训练阶段未见电影使用训练全局均值作为电影统计回退值，计数特征为0；静态电影类型和时间特征仍可使用。该方法不能完全解决新电影冷启动，只是提供确定、无泄漏的回退方案。

本阶段只比较三种集成学习方法，参数为事先固定的课程实验配置，并未进行大规模超参数搜索。目标统计特征依赖历史显式评分，在线系统还需考虑时间顺序、概念漂移和新用户问题。集成学习与神经网络的最终对比必须等第4阶段神经网络和第5阶段统一实验完成后再下结论。
"""


def main() -> int:
    total_start = time.perf_counter()
    np.random.seed(42)
    verify_stage2()
    previous = prior_core_results()
    config = load_json(CONFIG_PATH)
    X_train, y_train, X_test, y_test = load_fixed_interactions()
    print(f"固定训练集：{len(X_train)}；固定测试集：{len(X_test)}")

    pipelines = build_model_pipelines(config)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    fold_results = evaluate_cv(pipelines, X_train, y_train, config["cv"])
    cv_summary = summarize_cv(fold_results)
    fold_results.to_csv(REPORT_DIR / "cv_fold_results.csv", index=False, encoding="utf-8-sig")
    cv_summary.to_csv(REPORT_DIR / "cv_summary.csv", index=False, encoding="utf-8-sig")

    selected_name = str(cv_summary.iloc[0]["model"])
    fitted_models, final_fit_times = fit_final_models(pipelines, X_train, y_train)
    prediction_table = X_test[["interaction_id", "user_id", "movie_id"]].copy()
    prediction_table["rating"] = y_test.to_numpy()
    metric_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    importance_frames: list[pd.DataFrame] = []

    for name in MODEL_ORDER:
        model = fitted_models[name]
        prediction_start = time.perf_counter()
        prediction = model.predict(X_test)
        prediction_time = time.perf_counter() - prediction_start
        predictions[name] = prediction
        metrics = evaluate_predictions(y_test, prediction)
        metric_rows.append(
            {
                "model": name,
                **metrics,
                "fit_time_seconds": final_fit_times[name],
                "prediction_time_seconds": prediction_time,
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
            }
        )
        prediction_table[f"{name}_prediction"] = prediction
        # 三级压缩显著减小随机森林模型体积，便于后续部署和缓存加载。
        joblib.dump(model, MODEL_DIR / f"{name}.joblib", compress=3)
        importance_frames.append(extract_feature_importance(name, model))

    test_metrics = pd.DataFrame(metric_rows)
    test_metrics.to_csv(REPORT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(
        {row["model"]: {key: value for key, value in row.items() if key != "model"} for row in metric_rows},
        REPORT_DIR / "test_metrics.json",
    )
    prediction_table.to_csv(
        REPORT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    importance = pd.concat(importance_frames, ignore_index=True)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    selected_row = cv_summary.iloc[0]
    selected = {
        "selection_metric": "mean_5fold_cv_rmse",
        "selected_model": selected_name,
        "selected_cv_rmse": float(selected_row["rmse_mean"]),
        "selected_cv_mae": float(selected_row["mae_mean"]),
        "selection_rule": "最低平均CV RMSE；若相同则比较MAE和训练时间",
        "model_artifact": f"models/stage3/{selected_name}.joblib",
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(selected, REPORT_DIR / "selected_model.json")
    joblib.dump(fitted_models[selected_name], MODEL_DIR / "best_ensemble.joblib", compress=3)

    final_transformer = fitted_models[selected_name].named_steps["features"]
    feature_names = final_transformer.get_feature_names_out().astype(str).tolist()
    feature_groups = final_transformer.get_feature_groups()
    write_json(
        {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "feature_groups": feature_groups,
            "excluded_fields": ["rating", "interaction_id", "title", "imdb_url", "zip_code", "user_id", "movie_id"],
            "target_statistics": "训练行使用逐行留一；验证/测试使用当前训练部分映射",
        },
        PROCESSED_DIR / "feature_schema.json",
    )

    save_charts(cv_summary, test_metrics, importance, selected_name)
    baseline_rmse = float(load_json(STAGE2_BASELINE)["test_metrics"]["rmse"])
    summary = build_summary(
        config,
        cv_summary,
        test_metrics,
        importance,
        selected,
        len(feature_names),
        baseline_rmse,
    )
    (REPORT_DIR / "stage3_summary.md").write_text(summary, encoding="utf-8")

    reproducibility = compare_core_results(previous, cv_summary, test_metrics)
    training_log = {
        "started_with_stage2_verification": True,
        "cv_total_fit_seconds": float(fold_results["fit_time_seconds"].sum()),
        "cv_total_score_seconds": float(fold_results["score_time_seconds"].sum()),
        "final_fit_seconds": final_fit_times,
        "total_elapsed_seconds": time.perf_counter() - total_start,
        "reproducibility": reproducibility,
        "train_sha256": calculate_file_sha256(PROCESSED_DIR / "train_ratings.csv"),
        "test_sha256": calculate_file_sha256(PROCESSED_DIR / "test_ratings.csv"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(training_log, REPORT_DIR / "training_log.json")

    print("\n=== 第3阶段结果摘要 ===")
    print(cv_summary.to_string(index=False))
    print("\n固定测试集：")
    print(test_metrics[["model", "rmse", "mae", "r2"]].to_string(index=False))
    print(f"\n按5折CV RMSE选择的最佳模型：{selected_name}")
    print(f"最终特征数：{len(feature_names)}")
    if reproducibility["previous_results_found"]:
        print(f"与上次运行核心指标一致：{reproducibility['core_metrics_match']}")
    print("第3阶段主程序执行完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
