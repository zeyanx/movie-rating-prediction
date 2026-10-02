"""验证中国电影数据库、模型产物和网页推理接口。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.web.model_service import predict_chinese_ratings  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    catalog = pd.read_csv(ROOT / "data" / "catalog" / "chinese_rated_movies.csv")
    summary = json.loads(
        (ROOT / "reports" / "chinese_training" / "dataset_summary.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (ROOT / "models" / "chinese" / "model_metadata.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(ROOT / "reports" / "chinese_training" / "metrics.csv")

    require(len(catalog) == 300, "公开中国电影目录应恰好有300部")
    require(catalog["catalog_id"].is_unique, "catalog_id存在重复")
    require(catalog["cn_movie_index"].is_unique, "模型影片索引存在重复")
    require(sorted(catalog["cn_movie_index"].tolist()) == list(range(1, 301)), "影片索引不连续")
    require("movieId" not in catalog.columns and "userId" not in catalog.columns,
            "公开目录不得再分发MovieLens原始标识")
    require(summary["ratings"] == 55_483, "训练评分数异常")
    require(summary["users"] == 5_973, "训练用户数异常")
    require(summary["movies"] == 300, "训练电影数异常")
    require(metadata["test_rows"] == 5_973, "固定测试集规模异常")
    require(set(metrics["model"]) == {"global_mean", "random_forest", "xgboost", "mlp"},
            "模型指标不完整")
    require(np.isfinite(metrics[["rmse", "mae", "r2"]].to_numpy()).all(), "指标包含非有限值")
    baseline = float(metrics.loc[metrics["model"] == "global_mean", "rmse"].iloc[0])
    xgb = float(metrics.loc[metrics["model"] == "xgboost", "rmse"].iloc[0])
    require(xgb < baseline, "XGBoost未优于全局均值基线")

    predictions = predict_chinese_ratings(1, [1, 2, 300])
    require(predictions.shape == (3, 3), "三模型推理输出形状异常")
    require(np.isfinite(predictions.to_numpy()).all(), "模型推理包含非有限值")
    require(((predictions >= 0.5) & (predictions <= 5.0)).all().all(), "预测超出评分范围")

    print(catalog.head(3).to_string(index=False))
    print(metrics.to_string(index=False))
    print("中国电影数据库与训练验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
