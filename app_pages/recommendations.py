"""统一个性化推荐：中国电影与其他国家电影混合排序。"""

from __future__ import annotations

import streamlit as st

from movie_rating.web.config import load_app_config
from movie_rating.web.database import get_interactions, upsert_interaction
from movie_rating.web.ui import ensure_session_state, page_intro
from movie_rating.web.unified_service import recommend_unified_movies, unified_genres


user_id, profile_key = ensure_session_state()
config = load_app_config()
page_intro("个性化推荐", "综合中国电影与其他国家电影，为当前用户生成带理由的推荐结果。")

with st.form("unified_recommendation_controls"):
    c1, c2 = st.columns(2)
    top_n = c1.slider("推荐数量", 2, int(config["max_top_n"]), int(config["default_top_n"]))
    genre = c2.selectbox("电影类型", ["全部", *unified_genres()])
    st.form_submit_button("生成推荐", type="primary")

interactions = get_interactions(profile_key)
with st.spinner("正在计算预测评分并生成推荐理由……"):
    recommendations = recommend_unified_movies(
        user_id=user_id,
        interactions=interactions,
        top_n=int(top_n),
        genre=genre,
    )

if recommendations.empty:
    st.info("当前类型下没有可推荐的电影，请切换电影类型。")
    st.stop()

china_count = int((recommendations["model_space"] == "china").sum())
international_count = int((recommendations["model_space"] == "international").sum())
st.caption(f"本次推荐包含中国电影 {china_count} 部、其他国家电影 {international_count} 部。")

for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
    with st.container(border=True):
        title_col, score_col = st.columns([4, 1])
        title_col.subheader(f"{rank}. {row.title_zh}")
        title_col.caption(f"{row.origin} · {row.category} · 评分模型：{row.model_label}")
        score_col.metric("推荐分", f"{float(row.final_score):.2f}")
        st.markdown("**推荐理由**")
        for reason in row.reasons:
            st.write(f"- {reason}")

        # 其他国家电影沿用现有观影反馈表；中国电影推荐仍保持只读，避免混用两套影片主键。
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
