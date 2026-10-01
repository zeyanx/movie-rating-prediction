"""独立验收第3阶段特征工程、交叉验证、模型与报告。"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from movie_rating.baseline import evaluate_predictions  # noqa: E402
from movie_rating.data import calculate_file_sha256, load_raw_data, validate_raw_data  # noqa: E402
from movie_rating.features import MovieFeatureEngineer  # noqa: E402


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "stage3"
REPORT_DIR = PROJECT_ROOT / "reports" / "stage3"
MODEL_NAMES = ["random_forest", "adaboost", "xgboost"]
TOLERANCE = 1e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size > 0, f"缺少或空文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_stage2_subprocess() -> None:
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
    if result.returncode != 0:
        raise AssertionError("第2阶段复验失败：\n" + result.stdout + "\n" + result.stderr)
    require("第2阶段验证通过" in result.stdout, "第2阶段缺少通过标记")
    print("第2阶段复验：通过")


def load_enriched_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
    manifest = load_json(PROCESSED_DIR / "split_manifest.json")
    train_path = PROCESSED_DIR / "train_ratings.csv"
    test_path = PROCESSED_DIR / "test_ratings.csv"
    require(calculate_file_sha256(train_path) == manifest["train_sha256"], "训练集哈希变化")
    require(calculate_file_sha256(test_path) == manifest["test_sha256"], "测试集哈希变化")
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    require(len(train) == 80_000 and len(test) == 20_000, "固定划分规模错误")
    raw = load_raw_data(PROJECT_ROOT / "data" / "raw")
    validate_raw_data(raw)

    def enrich(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.merge(raw.users, on="user_id", validate="many_to_one").merge(
            raw.movies, on="movie_id", validate="many_to_one"
        )

    train_full = enrich(train)
    test_full = enrich(test)
    y_train = train_full.pop("rating").astype(float)
    y_test = test_full.pop("rating").astype(float)
    return train_full, y_train, test_full, y_test, manifest


def verify_cv_outputs(X_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_results = pd.read_csv(REPORT_DIR / "cv_fold_results.csv", encoding="utf-8-sig")
    summary = pd.read_csv(REPORT_DIR / "cv_summary.csv", encoding="utf-8-sig")
    require(len(fold_results) == 15, "折级结果不是15行")
    require(set(fold_results["model"]) == set(MODEL_NAMES), "折级结果模型不完整")
    require(len(summary) == 3, "CV汇总不是3行")
    for name in MODEL_NAMES:
        rows = fold_results[fold_results["model"] == name]
        require(len(rows) == 5, f"{name}不是5折")
        require(set(rows["fold"]) == {1, 2, 3, 4, 5}, f"{name}折号异常")
        require((rows["train_rows"] == 64_000).all(), f"{name}折内训练规模错误")
        require((rows["validation_rows"] == 16_000).all(), f"{name}验证规模错误")
        values = rows[["rmse", "mae", "r2", "fit_time_seconds", "score_time_seconds"]].to_numpy()
        require(np.isfinite(values).all(), f"{name}的CV结果存在非有限值")

    # 独立重建KFold，确认每个训练样本恰好作为验证记录一次且折内无交叉。
    coverage = np.zeros(len(X_train), dtype=np.int8)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, validation_index in splitter.split(X_train):
        require(set(train_index).isdisjoint(set(validation_index)), "KFold折内索引交叉")
        coverage[validation_index] += 1
    require((coverage == 1).all(), "并非每条训练记录恰好验证一次")

    expected_order = summary.sort_values(
        ["rmse_mean", "mae_mean", "fit_time_mean"], kind="stable"
    ).reset_index(drop=True)
    require(expected_order["rank_by_cv_rmse"].tolist() == [1, 2, 3], "CV排名字段错误")
    return fold_results, summary


def verify_feature_engineering(X_train: pd.DataFrame, y_train: pd.Series) -> int:
    transformer = MovieFeatureEngineer(user_smoothing=10.0, movie_smoothing=20.0)
    clone(transformer)
    matrix = transformer.fit_transform(X_train, y_train)
    require(matrix.shape[0] == 80_000, "训练特征行数错误")
    require(np.isfinite(matrix).all(), "训练特征存在NaN或无穷大")
    names = transformer.get_feature_names_out().astype(str)
    require(matrix.shape[1] == len(names), "特征名称与矩阵列数不同")

    forbidden = {"rating", "interaction_id", "title", "imdb_url", "zip_code", "user_id", "movie_id"}
    require(forbidden.isdisjoint(set(names)), "禁止字段进入了特征矩阵")

    # 手工复算第一行用户和电影的留一平滑均值。
    row_index = 0
    row = X_train.iloc[row_index]
    target = float(y_train.iloc[row_index])
    loo_global = (float(y_train.sum()) - target) / (len(y_train) - 1)
    user_mask = X_train["user_id"].to_numpy() == row["user_id"]
    movie_mask = X_train["movie_id"].to_numpy() == row["movie_id"]
    expected_user = (
        float(y_train.to_numpy()[user_mask].sum()) - target + 10.0 * loo_global
    ) / (int(user_mask.sum()) - 1 + 10.0)
    expected_movie = (
        float(y_train.to_numpy()[movie_mask].sum()) - target + 20.0 * loo_global
    ) / (int(movie_mask.sum()) - 1 + 20.0)
    user_column = int(np.where(names == "user_smoothed_mean")[0][0])
    movie_column = int(np.where(names == "movie_smoothed_mean")[0][0])
    require(math.isclose(float(matrix[row_index, user_column]), expected_user, abs_tol=2e-6), "用户留一均值计算错误")
    require(math.isclose(float(matrix[row_index, movie_column]), expected_movie, abs_tol=2e-6), "电影留一均值计算错误")

    # 构造训练中未见的用户、电影和类别，验证全局均值与零计数回退。
    unseen = X_train.iloc[[0]].copy()
    unseen.loc[:, "user_id"] = 9_999_991
    unseen.loc[:, "movie_id"] = 9_999_992
    unseen.loc[:, "gender"] = "UNKNOWN"
    unseen.loc[:, "occupation"] = "unknown_new_occupation"
    transformed_unseen = transformer.transform(unseen)
    user_count_column = int(np.where(names == "user_rating_count_log")[0][0])
    movie_count_column = int(np.where(names == "movie_rating_count_log")[0][0])
    require(math.isclose(float(transformed_unseen[0, user_column]), transformer.global_mean_, abs_tol=2e-6), "未见用户未回退全局均值")
    require(math.isclose(float(transformed_unseen[0, movie_column]), transformer.global_mean_, abs_tol=2e-6), "未见电影未回退全局均值")
    require(transformed_unseen[0, user_count_column] == 0.0, "未见用户计数不为0")
    require(transformed_unseen[0, movie_count_column] == 0.0, "未见电影计数不为0")
    return len(names)


def verify_models_and_reports(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_summary: pd.DataFrame,
) -> None:
    test_metrics = pd.read_csv(REPORT_DIR / "test_metrics.csv", encoding="utf-8-sig")
    predictions = pd.read_csv(REPORT_DIR / "test_predictions.csv", encoding="utf-8-sig")
    importance = pd.read_csv(REPORT_DIR / "feature_importance.csv", encoding="utf-8-sig")
    selection = load_json(REPORT_DIR / "selected_model.json")
    schema = load_json(PROCESSED_DIR / "feature_schema.json")
    require(len(test_metrics) == 3 and set(test_metrics["model"]) == set(MODEL_NAMES), "测试指标模型不完整")
    require(len(predictions) == 20_000, "测试预测行数错误")

    loaded_models = {}
    for name in MODEL_NAMES:
        path = MODEL_DIR / f"{name}.joblib"
        require(path.is_file() and path.stat().st_size > 0, f"缺少模型：{name}")
        model = joblib.load(path)
        loaded_models[name] = model
        require(not model.named_steps["model"].estimator_.__class__.__module__.startswith("torch"), "发现神经网络模型")
        prediction = model.predict(X_test)
        require(prediction.shape == (20_000,), f"{name}预测数量错误")
        require(np.isfinite(prediction).all(), f"{name}预测存在非有限值")
        require(((prediction >= 1.0) & (prediction <= 5.0)).all(), f"{name}预测超出1至5")
        saved_column = f"{name}_prediction"
        require(saved_column in predictions.columns, f"预测文件缺少{saved_column}")
        require(np.allclose(prediction, predictions[saved_column], rtol=0.0, atol=TOLERANCE), f"{name}保存预测与模型不一致")

        recomputed = evaluate_predictions(y_test, prediction)
        row = test_metrics[test_metrics["model"] == name].iloc[0]
        for metric in ("rmse", "mae", "r2"):
            require(math.isfinite(float(row[metric])), f"{name}.{metric}不是有限值")
            require(math.isclose(float(row[metric]), recomputed[metric], rel_tol=0.0, abs_tol=TOLERANCE), f"{name}.{metric}复算不一致")

        feature_names = model.named_steps["features"].get_feature_names_out()
        raw_importance = model.named_steps["model"].feature_importances_
        require(len(feature_names) == len(raw_importance), f"{name}特征与重要性数量不一致")
        model_importance = importance[importance["model"] == name]
        require(len(model_importance) == len(feature_names), f"{name}重要性报告行数错误")
        require(math.isclose(float(model_importance["normalized_importance"].sum()), 1.0, abs_tol=1e-8), f"{name}归一化重要性之和不是1")
        require(set(model_importance["rank"]) == set(range(1, len(feature_names) + 1)), f"{name}重要性排名不完整")
        require(schema["feature_count"] == len(feature_names), "特征清单数量错误")
        require(schema["feature_names"] == feature_names.astype(str).tolist(), "特征清单名称错误")

    expected = cv_summary.sort_values(
        ["rmse_mean", "mae_mean", "fit_time_mean"], kind="stable"
    ).iloc[0]
    selected_name = str(expected["model"])
    require(selection["selection_metric"] == "mean_5fold_cv_rmse", "模型选择指标错误")
    require(selection["selected_model"] == selected_name, "最佳模型未按CV选择")
    require(math.isclose(selection["selected_cv_rmse"], float(expected["rmse_mean"]), abs_tol=TOLERANCE), "选择记录CV RMSE错误")

    best_path = MODEL_DIR / "best_ensemble.joblib"
    best_model = joblib.load(best_path)
    require(
        np.allclose(best_model.predict(X_test), loaded_models[selected_name].predict(X_test), rtol=0.0, atol=TOLERANCE),
        "best_ensemble与选中模型不一致",
    )

    for filename in [
        "test_metrics.json",
        "training_log.json",
        "stage3_summary.md",
        "cv_rmse_comparison.html",
        "test_metrics_comparison.html",
        "feature_importance.html",
    ]:
        path = REPORT_DIR / filename
        require(path.is_file() and path.stat().st_size > 0, f"缺少或空文件：{path}")


def main() -> int:
    print("=== 第3阶段独立验收 ===")
    verify_stage2_subprocess()
    X_train, y_train, X_test, y_test, manifest = load_enriched_data()
    fold_results, cv_summary = verify_cv_outputs(X_train)
    feature_count = verify_feature_engineering(X_train, y_train)
    verify_models_and_reports(X_test, y_test, cv_summary)

    training_log = load_json(REPORT_DIR / "training_log.json")
    require(training_log["train_sha256"] == manifest["train_sha256"], "训练日志哈希错误")
    require(training_log["test_sha256"] == manifest["test_sha256"], "测试日志哈希错误")
    require(len(fold_results) == 15, "CV折数最终检查失败")

    selected = load_json(REPORT_DIR / "selected_model.json")
    print(f"最终特征数：{feature_count}")
    print(f"完成CV拟合：{len(fold_results)}次")
    print(f"按CV RMSE选择的最佳模型：{selected['selected_model']}")
    if training_log["reproducibility"]["previous_results_found"]:
        require(training_log["reproducibility"]["core_metrics_match"] is True, "第二次运行核心指标不一致")
        print("第二次运行可重复性：通过")
    else:
        print("首次运行：已通过固定KFold和模型重载确定性检查")
    print("特征泄漏、模型加载、预测范围、指标复算、重要性与选择规则：全部通过")
    print("第3阶段验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
