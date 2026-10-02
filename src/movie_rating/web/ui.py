"""五个页面共享的小型UI辅助函数。"""

from __future__ import annotations

from urllib.parse import urlparse

import streamlit as st

from .config import load_app_config


MODEL_LABELS = {
    "global_mean": "全局均值",
    "random_forest": "随机森林",
    "adaboost": "自适应提升",
    "xgboost": "梯度提升树",
    "mlp": "神经网络",
}
STATUS_LABELS = {"want_to_watch": "想看", "watched": "已看"}
FEEDBACK_LABELS = {
    None: "未选择",
    "like": "喜欢",
    "neutral": "中立",
    "dislike": "不喜欢",
    "not_interested": "不感兴趣",
}


def ensure_session_state() -> tuple[int, str]:
    config = load_app_config()
    if "selected_user_id" not in st.session_state:
        st.session_state["selected_user_id"] = int(config["default_user_id"])
    user_id = int(st.session_state["selected_user_id"])
    st.session_state["profile_key"] = f"movielens:{user_id}"
    return user_id, st.session_state["profile_key"]


def valid_external_url(value: object) -> bool:
    try:
        parsed = urlparse(str(value))
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def format_movie_option(row) -> str:
    title = row.get("display_title", row["title"])
    return f"{title} · ID {int(row['movie_id'])}"


def page_intro(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


def show_interaction_notice() -> None:
    st.info("切换当前用户后，预测评分和推荐结果会同步更新。")
