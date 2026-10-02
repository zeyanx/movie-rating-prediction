"""第6阶段 Streamlit 多页面应用唯一入口。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.web.config import load_app_config
from movie_rating.web.data_service import load_raw_tables
from movie_rating.web.database import initialize_database
from movie_rating.web.runtime_assets import ensure_runtime_assets_cached


config = load_app_config()
st.set_page_config(
    page_title=config["app_title"], page_icon="🎬", layout="wide",
    initial_sidebar_state="expanded",
)

try:
    with st.spinner("正在准备电影数据和评分模型……"):
        ensure_runtime_assets_cached()
    initialize_database()
    _, _, users = load_raw_tables()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"应用初始化失败：{exc}")
    st.info(
        "请检查GroupLens官方网络连接和模型文件。官方数据地址："
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    )
    st.stop()

if "selected_user_id" not in st.session_state:
    st.session_state["selected_user_id"] = int(config["default_user_id"])

user_ids = users["user_id"].astype(int).tolist()
current = int(st.session_state["selected_user_id"])
if current not in user_ids:
    current = int(config["default_user_id"])
with st.sidebar:
    st.header("🎬 电影导航")
    selected = st.selectbox(
        "当前用户",
        user_ids,
        index=user_ids.index(current),
        key="sidebar_user_selector",
        help="切换用户会改变个性化预测、推荐和本地观影记录。",
    )
    st.session_state["selected_user_id"] = int(selected)
    st.session_state["profile_key"] = f"movielens:{int(selected)}"
    st.caption("切换用户可查看不同的预测与推荐结果")

pages = [
    st.Page("app_pages/home.py", title="首页", icon="🏠", default=True),
    st.Page("app_pages/chinese_catalog.py", title="电影库", icon="🎞️"),
    st.Page("app_pages/movie_detail.py", title="评分预测", icon="⭐"),
    st.Page("app_pages/recommendations.py", title="个性化推荐", icon="✨"),
    st.Page("app_pages/my_movies.py", title="我的观影", icon="📚"),
    st.Page("app_pages/model_lab.py", title="模型分析", icon="📊"),
]

try:
    current_page = st.navigation(pages, position="sidebar")
    current_page.run()
except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
    st.error(f"页面运行失败：{exc}")
    st.info("请检查模型和报告文件是否齐全，并运行 python scripts/verify_stage6.py。")
