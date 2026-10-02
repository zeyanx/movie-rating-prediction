"""第6阶段兼容验收：数据、推理、反馈、六页面与本地服务。"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.web.config import load_app_config
from movie_rating.web.data_service import build_prediction_frame, load_raw_tables, load_train_ratings
from movie_rating.web.database import (
    delete_interaction, get_interaction, initialize_database, upsert_interaction,
)
from movie_rating.web.model_service import predict_with_mlp
from movie_rating.web.recommendation import recommend_movies


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_checked(arguments: list[str], label: str, timeout: int = 180) -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        arguments, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False, timeout=timeout, env=environment,
    )
    if result.returncode != 0:
        raise AssertionError(f"{label}失败：\n{result.stdout}\n{result.stderr}")
    print(f"{label}：通过")


def main() -> int:
    print("=== 第6阶段独立验收 ===")
    require(sys.version_info[:2] == (3, 10), f"要求Python 3.10，当前为{sys.version.split()[0]}")

    # 第5阶段验证器只复核历史产物，不会重新训练模型。
    run_checked([sys.executable, str(ROOT / "scripts" / "verify_stage5.py")], "第5阶段复验")

    required_files = [
        "app.py", "configs/stage6_app.json", ".streamlit/config.toml",
        "app_pages/home.py", "app_pages/movie_detail.py", "app_pages/recommendations.py",
        "app_pages/my_movies.py", "app_pages/model_lab.py",
        "src/movie_rating/web/__init__.py", "src/movie_rating/web/config.py",
        "src/movie_rating/web/data_service.py", "src/movie_rating/web/model_service.py",
        "src/movie_rating/web/recommendation.py", "src/movie_rating/web/database.py",
        "src/movie_rating/web/ui.py", "scripts/init_app_db.py",
        "scripts/smoke_test_streamlit.py", "tests/test_stage6_services.py",
        "tests/test_stage6_app.py", "reports/stage6/stage6_summary.md",
    ]
    for name in required_files:
        path = ROOT / name
        require(path.is_file() and path.stat().st_size > 0, f"缺少或为空：{name}")

    config = load_app_config()
    require(abs(sum(config["recommendation_weights"].values()) - 1.0) < 1e-9, "推荐权重之和不为1")
    require(config["default_model"] == "mlp", "默认模型不是MLP")

    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    for token in ["st.navigation", "st.Page", ".run()", "selected_user_id", "profile_key"]:
        require(token in app_source, f"app.py缺少：{token}")
    for page in ["home.py", "movie_detail.py", "recommendations.py", "my_movies.py", "model_lab.py"]:
        require(page in app_source, f"入口未注册页面：{page}")
    require(not (ROOT / "pages").exists(), "不应存在与st.navigation冲突的根pages目录")

    data_source = (ROOT / "src" / "movie_rating" / "web" / "data_service.py").read_text(encoding="utf-8")
    model_source = (ROOT / "src" / "movie_rating" / "web" / "model_service.py").read_text(encoding="utf-8")
    recommendation_source = (ROOT / "src" / "movie_rating" / "web" / "recommendation.py").read_text(encoding="utf-8")
    require("@st.cache_data" in data_source, "静态数据没有使用st.cache_data")
    require("@st.cache_resource" in model_source, "模型没有使用st.cache_resource")
    require("test_ratings" not in recommendation_source + data_source, "网页推荐代码疑似读取固定测试集")
    for forbidden in [".fit(", "optimizer.step", ".backward("]:
        require(forbidden not in model_source + recommendation_source, f"网页代码疑似训练模型：{forbidden}")

    ratings, movies, users = load_raw_tables()
    train = load_train_ratings()
    require((len(ratings), len(users), len(movies), len(train)) == (100_000, 943, 1_682, 80_000), "数据规模异常")
    frame = build_prediction_frame(1, [1, 2])
    prediction = predict_with_mlp(frame)
    require(len(prediction) == 2 and np.isfinite(prediction).all(), "MLP实际推理失败")
    require(((prediction >= 1) & (prediction <= 5)).all(), "MLP预测超出1至5")

    with tempfile.TemporaryDirectory() as directory:
        db_path = Path(directory) / "verify.db"
        initialize_database(db_path)
        initialize_database(db_path)
        upsert_interaction("movielens:1", 1, 1, "want_to_watch", feedback="like", path=db_path)
        upsert_interaction("movielens:1", 1, 1, "watched", personal_rating=4.0, path=db_path)
        record = get_interaction("movielens:1", 1, db_path)
        require(record is not None and record["status"] == "watched", "SQLite新增/更新失败")
        require(delete_interaction("movielens:1", 1, db_path), "SQLite删除失败")
        require(get_interaction("movielens:1", 1, db_path) is None, "SQLite删除后仍有记录")

    actual_database = initialize_database()
    with sqlite3.connect(actual_database) as connection:
        require(connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_interactions'"
        ).fetchone() is not None, "真实数据库缺少user_interactions表")

    recommendations, metadata = recommend_movies(1, pd.DataFrame(), top_n=5)
    require(0 < len(recommendations) <= 5, "推荐数量异常")
    require(recommendations["movie_id"].is_unique, "推荐含重复电影")
    require(recommendations["reasons"].map(lambda values: len(values) >= 2).all(), "推荐理由不足两条")
    require(recommendations["final_score"].between(1, 5).all(), "推荐评分超出1至5")
    require(metadata["excluded_training_ratings"] > 0, "未排除训练期已评分电影")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for token in ["data/app/*.db", "data/app/*.db-wal", ".streamlit/secrets.toml"]:
        require(token in gitignore, f".gitignore缺少：{token}")
    streamlit_config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8").lower()
    for forbidden in ["password", "token", "secret", "api_key"]:
        require(forbidden not in streamlit_config, ".streamlit/config.toml疑似包含密钥")

    run_checked(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_stage6*.py", "-v"],
        "服务层与六页面AppTest", timeout=180,
    )
    run_checked([sys.executable, str(ROOT / "scripts" / "smoke_test_streamlit.py")], "Streamlit本地健康检查", timeout=60)

    print(f"数据规模：{len(ratings)}条评分、{len(users)}名用户、{len(movies)}部电影")
    print(f"实际MLP预测：{prediction.round(4).tolist()}")
    print(f"推荐验收：{len(recommendations)}部，候选{metadata['candidate_count']}部，每部至少2条理由")
    print(f"SQLite：{actual_database.relative_to(ROOT)}")
    print("第6阶段验证通过")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"第6阶段验证失败：{error}", file=sys.stderr)
        raise SystemExit(1)
