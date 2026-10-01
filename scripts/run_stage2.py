"""执行第2阶段：数据质量检查、固定划分与全局均值基线。"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from movie_rating.baseline import GlobalMeanBaseline, evaluate_predictions  # noqa: E402
from movie_rating.data import (  # noqa: E402
    build_data_quality_report,
    calculate_file_sha256,
    cold_start_statistics,
    load_raw_data,
    rating_distribution,
    save_split,
    split_ratings,
    validate_raw_data,
    write_json,
)


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "global_mean_baseline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports" / "stage2"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_prediction_table(
    test: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """生成可逐行审计的测试集预测和误差表。"""
    result = test[["interaction_id", "user_id", "movie_id", "rating"]].copy()
    result["predicted_rating"] = predictions
    result["residual"] = result["rating"] - result["predicted_rating"]
    result["absolute_error"] = result["residual"].abs()
    result["squared_error"] = result["residual"].pow(2)
    return result


def build_summary_markdown(
    quality: dict,
    manifest: dict,
    metrics: dict,
) -> str:
    """根据实际运行结果生成可复用的中文论文素材。"""
    train_metrics = metrics["train_metrics"]
    test_metrics = metrics["test_metrics"]
    movies_missing = quality["tables"]["movies"]["missing_values"]
    distribution_rows = []
    for score in range(1, 6):
        key = str(score)
        full = manifest["rating_distribution_full"][key]
        train = manifest["rating_distribution_train"][key]
        test = manifest["rating_distribution_test"][key]
        distribution_rows.append(
            f"| {score} | {full['count']} | {train['count']} | {test['count']} |"
        )
    distribution_table = "\n".join(distribution_rows)

    return rf"""# 第2阶段：数据预处理、固定划分与全局均值基线

## 1. 数据集与字段

本研究采用 MovieLens 100K 稳定基准数据集，共包含100000条评分、943名用户和1682部电影。评分表包含用户编号、电影编号、1至5分的显式评分和Unix时间戳；用户表包含年龄、性别、职业和邮政编码；电影表包含标题、上映日期、IMDb链接和类型信息。

## 2. 数据质量与预处理原则

三张表的主键和引用完整性检查均通过，重复用户—电影评分数为{quality['duplicate_user_movie_pairs']}，非法时间戳数为{quality['invalid_rating_timestamps']}。用户年龄范围为{quality['user_age_range']['min']}至{quality['user_age_range']['max']}岁，性别编码为{', '.join(quality['gender_values'])}。电影表中 `release_date` 缺失{quality['missing_release_dates']}条，非空但无法解析的上映日期有{quality['unparseable_non_missing_release_dates']}条；`video_release_date` 缺失{movies_missing['video_release_date']}条，`imdb_url` 缺失{movies_missing['imdb_url']}条。

本阶段只清理字符串首尾空格，并将评分时间戳额外解析为UTC时间用于质量检查，不覆盖原始整数时间戳。对缺失字段不进行主观填补，`video_release_date` 本阶段不作为模型输入。本阶段未进行ID编码、独热编码、标准化或统计特征构造。

## 3. 训练集与测试集划分

采用分层随机划分：测试集比例为20%，随机种子固定为42，并按评分值分层。训练集包含{manifest['train_rows']}条记录，测试集包含{manifest['test_rows']}条记录。固定随机种子保证实验可重复，按评分分层可减少不同子集中评分分布的偶然偏差。划分结果已持久化，后续所有模型必须复用，不得重新随机划分。

| 评分 | 全量数量 | 训练集数量 | 测试集数量 |
|---:|---:|---:|---:|
{distribution_table}

测试集中训练阶段未出现的用户有{manifest['unseen_test_users']}个，未出现的电影有{manifest['unseen_test_movies']}部，涉及冷启动用户或电影的测试记录共{manifest['cold_start_test_rows']}条。这些记录保持在测试集中，以如实反映固定随机划分的边界情况。

## 4. 数据泄漏控制

全局评分均值只使用训练集计算，测试集既不参与均值计算，也不参与任何训练参数估计。原始评分文件、训练集和测试集均记录SHA-256，且使用稳定的 `interaction_id` 验证训练集和测试集无交叉、无丢失。

## 5. 全局均值基线

设训练集评分集合为 $R_{{train}}$，训练集全局平均评分为：

$$
\mu_{{train}} = \frac{{1}}{{|R_{{train}}|}} \sum_{{(u,i)\in R_{{train}}}} r_{{ui}}
$$

对任意用户—电影组合，基线预测为：

$$
\hat{{r}}_{{ui}} = \mu_{{train}}
$$

本次实验训练集全局平均评分为 **{metrics['global_mean']:.8f}**。

## 6. 评估指标

均方根误差、平均绝对误差和决定系数分别定义为：

$$RMSE = \sqrt{{\frac{{1}}{{n}}\sum_{{j=1}}^n(y_j-\hat{{y}}_j)^2}}$$

$$MAE = \frac{{1}}{{n}}\sum_{{j=1}}^n|y_j-\hat{{y}}_j|$$

$$R^2 = 1-\frac{{\sum_j(y_j-\hat{{y}}_j)^2}}{{\sum_j(y_j-\bar{{y}})^2}}$$

| 数据集 | RMSE | MAE | R² |
|---|---:|---:|---:|
| 训练集 | {train_metrics['rmse']:.6f} | {train_metrics['mae']:.6f} | {train_metrics['r2']:.6e} |
| 测试集 | {test_metrics['rmse']:.6f} | {test_metrics['mae']:.6f} | {test_metrics['r2']:.6e} |

## 7. 结果分析

全局均值模型为所有样本给出同一个预测值，因此无法表达用户偏好、电影质量差异或用户与电影之间的交互。它的主要作用是提供最低复杂度的参照系：后续模型只有在同一固定测试集上取得更低的RMSE和MAE，并结合R²进行解释，才能说明复杂模型带来了有效提升。

训练集R²理论上接近0，因为预测值正是训练集目标均值；测试集R²可能略低于0，这是训练均值与测试集均值存在轻微差异时的正常现象，并不表示指标计算错误。

## 8. 局限性

本阶段没有利用用户人口属性、电影类型、用户历史偏好或电影受欢迎程度，也没有处理冷启动问题。全局均值基线不能用于生成有区分度的个性化推荐，其结果仅作为后续集成学习和神经网络模型的比较基准。
"""


def main() -> int:
    np.random.seed(RANDOM_STATE)
    print("=== 第2阶段：加载并验证原始数据 ===")
    data = load_raw_data(RAW_DIR)
    validate_raw_data(data)
    quality = build_data_quality_report(data)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(quality, REPORT_DIR / "data_quality.json")

    print("=== 固定分层划分 ===")
    train, test = split_ratings(
        data.ratings,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    save_split(train, test, PROCESSED_DIR)
    train_path = PROCESSED_DIR / "train_ratings.csv"
    test_path = PROCESSED_DIR / "test_ratings.csv"

    cold_start = cold_start_statistics(train, test)
    manifest = {
        "source_file": "data/raw/ratings.csv",
        "source_sha256": calculate_file_sha256(RAW_DIR / "ratings.csv"),
        "split_method": "sklearn.model_selection.train_test_split",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "shuffle": True,
        "stratify_column": "rating",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_sha256": calculate_file_sha256(train_path),
        "test_sha256": calculate_file_sha256(test_path),
        "rating_distribution_full": rating_distribution(data.ratings),
        "rating_distribution_train": rating_distribution(train),
        "rating_distribution_test": rating_distribution(test),
        "train_unique_users": int(train["user_id"].nunique()),
        "test_unique_users": int(test["user_id"].nunique()),
        "train_unique_movies": int(train["movie_id"].nunique()),
        "test_unique_movies": int(test["movie_id"].nunique()),
        **cold_start,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest, PROCESSED_DIR / "split_manifest.json")

    print("=== 拟合全局均值基线（仅使用训练集） ===")
    model = GlobalMeanBaseline(random_state=RANDOM_STATE).fit(train["rating"])
    model.save(MODEL_PATH)
    train_predictions = model.predict_dataframe(train)
    test_predictions = model.predict_dataframe(test)
    train_metrics = evaluate_predictions(train["rating"], train_predictions)
    test_metrics = evaluate_predictions(test["rating"], test_predictions)

    metrics = {
        "model_name": model.model_name,
        "global_mean": model.global_mean_,
        "rating_min": model.rating_min,
        "rating_max": model.rating_max,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    write_json(metrics, REPORT_DIR / "baseline_metrics.json")

    prediction_table = build_prediction_table(test, test_predictions)
    prediction_table.to_csv(
        REPORT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    summary = build_summary_markdown(quality, manifest, metrics)
    (REPORT_DIR / "stage2_summary.md").write_text(summary, encoding="utf-8")

    print(f"训练集：{len(train)}条；测试集：{len(test)}条")
    print(f"训练集全局平均评分：{model.global_mean_:.8f}")
    print(
        "训练集指标："
        f"RMSE={train_metrics['rmse']:.6f}, "
        f"MAE={train_metrics['mae']:.6f}, "
        f"R²={train_metrics['r2']:.6e}"
    )
    print(
        "测试集指标："
        f"RMSE={test_metrics['rmse']:.6f}, "
        f"MAE={test_metrics['mae']:.6f}, "
        f"R²={test_metrics['r2']:.6e}"
    )
    print(
        "冷启动统计："
        f"未见用户={cold_start['unseen_test_users']}，"
        f"未见电影={cold_start['unseen_test_movies']}，"
        f"相关测试记录={cold_start['cold_start_test_rows']}"
    )
    print("第2阶段主程序执行完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
