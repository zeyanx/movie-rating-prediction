"""在不含本地数据和数据库的部署副本中验证首次启动。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_rating.web.data_service import (
    build_prediction_frame, load_chinese_rated_catalog, load_raw_tables,
    search_chinese_catalog, search_chinese_rated_catalog, search_movies,
)
from movie_rating.web.database import get_chinese_movies, get_movie_localizations
from movie_rating.web.model_service import predict_chinese_ratings, predict_with_models
from movie_rating.web.recommendation import recommend_movies
from movie_rating.web.runtime_assets import ensure_runtime_assets


def main() -> int:
    result = ensure_runtime_assets(ROOT)
    ratings, movies, users = load_raw_tables()
    if (len(ratings), len(users), len(movies)) != (100_000, 943, 1_682):
        raise AssertionError("首次启动生成的数据规模异常")
    localizations = get_movie_localizations()
    chinese_catalog = get_chinese_movies()
    if (len(localizations), len(chinese_catalog)) != (80, 48):
        raise AssertionError("中文电影资料层规模异常")
    if search_movies("玩具总动员", limit=10)["movie_id"].astype(int).tolist() != [1]:
        raise AssertionError("MovieLens中文检索失败")
    if search_chinese_catalog("流浪地球", limit=10).iloc[0]["catalog_id"] != "cn034":
        raise AssertionError("中国电影扩展库检索失败")
    rated_catalog = load_chinese_rated_catalog()
    if len(rated_catalog) != 300:
        raise AssertionError("中国电影真实评分目录规模异常")
    if search_chinese_rated_catalog("重庆森林", limit=10).empty:
        raise AssertionError("中国电影真实评分目录检索失败")
    chinese_predictions = predict_chinese_ratings(1, [1, 2, 300])
    if chinese_predictions.shape != (3, 3) or not np.isfinite(chinese_predictions.to_numpy()).all():
        raise AssertionError("中国电影三模型推理失败")
    frame = build_prediction_frame(1, [1, 2, 3])
    predictions = predict_with_models(
        frame, ["global_mean", "random_forest", "xgboost", "mlp"]
    )
    if predictions.shape != (3, 4) or not np.isfinite(predictions.to_numpy()).all():
        raise AssertionError("干净副本四模型推理失败")
    recommendations, metadata = recommend_movies(1, top_n=5)
    if len(recommendations) != 5 or not recommendations["reasons"].map(lambda x: len(x) >= 2).all():
        raise AssertionError("干净副本推荐或理由生成失败")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_stage6_app.py", "-v"],
        cwd=ROOT, check=False, text=True, encoding="utf-8", errors="replace",
        capture_output=True,
    )
    if tests.returncode:
        raise AssertionError(f"干净副本页面测试失败：\n{tests.stdout}\n{tests.stderr}")
    health = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_streamlit.py")],
        cwd=ROOT, check=False, text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=90,
    )
    if health.returncode:
        raise AssertionError(f"干净副本健康检查失败：\n{health.stdout}\n{health.stderr}")
    payload = {
        "dataset": result["verification"],
        "model_predictions": {
            name: predictions[name].round(6).tolist() for name in predictions.columns
        },
        "recommendation_count": len(recommendations),
        "candidate_count": metadata["candidate_count"],
        "movielens_localizations": len(localizations),
        "chinese_catalog_movies": len(chinese_catalog),
        "chinese_rated_movies": len(rated_catalog),
        "chinese_model_predictions": {
            name: chinese_predictions[name].round(6).tolist()
            for name in chinese_predictions.columns
        },
        "app_test": "passed",
        "streamlit_health": "passed",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("干净部署副本验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
