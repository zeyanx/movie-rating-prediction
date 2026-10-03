"""统一个性化推荐：中国电影与其他国家电影混合排序。"""

from __future__ import annotations

import streamlit as st

from movie_rating.web.config import load_app_config
from movie_rating.web.user_store import (
    get_catalog_interactions, get_interactions,
    upsert_catalog_interaction, upsert_interaction,
)
from movie_rating.web.ui import account_mode, ensure_session_state, page_intro
from movie_rating.web.unified_service import recommend_unified_movies, unified_genres


user_id, profile_key = ensure_session_state()
config = load_app_config()
page_intro("个性化推荐", "综合中国电影与其他国家电影，为当前用户生成带理由的推荐结果。")

with st.form("unified_recommendation_controls"):
    c1, c2 = st.columns(2)
    top_n = c1.slider("推荐数量", 2, int(config["max_top_n"]), int(config["default_top_n"]))
    genre = c2.selectbox("电影类型", ["全部", *unified_genres()])
    generate_col, refresh_col = st.columns(2)
    generate_clicked = generate_col.form_submit_button(
        "生成推荐", type="primary", width="stretch"
    )
    refresh_clicked = refresh_col.form_submit_button(
        "换一批推荐", width="stretch"
    )

# 推荐页码保存在当前浏览器会话中；切换用户、数量或类型时自动回到首批。
refresh_signature = f"{profile_key}|{user_id}|{int(top_n)}|{genre}"
if st.session_state.get("recommendation_refresh_signature") != refresh_signature:
    st.session_state["recommendation_refresh_signature"] = refresh_signature
    st.session_state["recommendation_refresh_page"] = 0
elif generate_clicked:
    st.session_state["recommendation_refresh_page"] = 0
elif refresh_clicked:
    st.session_state["recommendation_refresh_page"] = (
        int(st.session_state.get("recommendation_refresh_page", 0)) + 1
    )
refresh_page = int(st.session_state.get("recommendation_refresh_page", 0))

interactions = get_interactions(profile_key)
catalog_interactions = get_catalog_interactions(profile_key)
with st.spinner("正在计算预测评分并生成推荐理由……"):
    recommendations = recommend_unified_movies(
        user_id=user_id,
        interactions=interactions,
        top_n=int(top_n),
        genre=genre,
        catalog_interactions=catalog_interactions,
        refresh_page=refresh_page,
    )

# 当前筛选条件的候选已翻完时自动回到第一批，避免显示空白页面。
if recommendations.empty and refresh_page > 0:
    st.session_state["recommendation_refresh_page"] = 0
    refresh_page = 0
    recommendations = recommend_unified_movies(
        user_id=user_id,
        interactions=interactions,
        top_n=int(top_n),
        genre=genre,
        catalog_interactions=catalog_interactions,
        refresh_page=0,
    )
    st.info("当前筛选条件下的候选已经浏览完，已回到第一批推荐。")

if recommendations.empty:
    st.info("当前类型下没有可推荐的电影，请切换电影类型。")
    st.stop()

china_count = int((recommendations["model_space"] == "china").sum())
international_count = int((recommendations["model_space"] == "international").sum())
st.caption(f"本次推荐包含中国电影 {china_count} 部、其他国家电影 {international_count} 部。")
st.caption(f"当前为第 {refresh_page + 1} 批；点击“换一批推荐”可继续刷新。")

for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
    with st.container(border=True):
        title_col, score_col = st.columns([4, 1])
        title_col.subheader(f"{rank}. {row.title_zh}")
        title_col.caption(f"{row.origin} · {row.category} · 评分模型：{row.model_label}")
        score_col.metric("推荐分", f"{float(row.final_score):.2f}")
        st.markdown("**推荐理由**")
        for reason in row.reasons:
            st.write(f"- {reason}")

        if row.model_space == "international":
            b1, b2, b3, b4 = st.columns(4)
            actions = [
                (b1, "加入想看", "want_to_watch", None),
                (b2, "标记已看", "watched", None),
                (b3, "喜欢", "want_to_watch", "like"),
                (b4, "不感兴趣", "want_to_watch", "not_interested"),
            ]
            movie_id = int(row.movie_id)
            for column, label, status, feedback in actions:
                if column.button(label, key=f"unified_{label}_{movie_id}", width="stretch"):
                    upsert_interaction(
                        profile_key,
                        user_id,
                        movie_id,
                        status,
                        feedback=feedback,
                        predicted_rating=float(row.mlp_prediction),
                    )
                    st.toast(f"已保存：{row.title_zh} · {label}")
                    st.rerun()
        elif account_mode():
            b1, b2 = st.columns(2)
            actions = [(b1, "加入想看", "want_to_watch"), (b2, "标记已看", "watched")]
            for column, label, status in actions:
                if column.button(label, key=f"china_{label}_{row.catalog_id}", width="stretch"):
                    upsert_catalog_interaction(profile_key, str(row.catalog_id), status)
                    st.toast(f"已保存：{row.title_zh} · {label}")
                    st.rerun()
