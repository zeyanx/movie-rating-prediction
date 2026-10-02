"""个性化推荐：批量MLP评分、稳定排序、理由和快捷反馈。"""

from __future__ import annotations

import streamlit as st

from movie_rating.web.config import load_app_config
from movie_rating.web.data_service import all_genres
from movie_rating.web.database import get_interactions, upsert_interaction
from movie_rating.web.recommendation import recommend_movies
from movie_rating.web.ui import ensure_session_state, page_intro


user_id, profile_key = ensure_session_state()
config = load_app_config()
page_intro("个性化推荐", "MLP个性化预测 + 类型偏好 + 贝叶斯热度，并为每部电影给出可核验理由。")
st.caption("云端反馈用于课程演示；Community Cloud重启或重新构建后SQLite记录可能重置。")

with st.form("recommendation_controls"):
    c1, c2, c3 = st.columns(3)
    top_n = c1.slider("推荐数量", 1, int(config["max_top_n"]), int(config["default_top_n"]))
    genre = c2.selectbox("电影类型", ["全部", *all_genres()])
    min_count = c3.number_input(
        "最少训练期评分数", min_value=0, max_value=500,
        value=int(config["minimum_popularity_count"]), step=1,
    )
    generate = st.form_submit_button("生成推荐", type="primary")

interactions = get_interactions(profile_key)
with st.spinner("正在批量预测候选电影并生成理由……"):
    recommendations, metadata = recommend_movies(
        user_id, interactions, top_n, genre, int(min_count)
    )

st.caption(
    f"已排除当前用户训练期评价过的 {metadata.get('excluded_training_ratings', 0)} 部电影，"
    "以及本地已看、不感兴趣和点踩记录。推荐排序不使用固定测试集真实评分。"
)
if metadata.get("relaxed_popularity"):
    st.warning("满足条件的候选不足，已仅放宽最低评分数量；明确排除项仍然有效。")

if recommendations.empty:
    st.info("当前筛选下没有候选电影。请切换类型或降低最低评分数量。")
    st.stop()

for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
    with st.container(border=True):
        title_col, score_col = st.columns([4, 1])
        shown_title = getattr(row, "display_title", row.title)
        title_col.subheader(f"{rank}. {shown_title}")
        title_col.caption(f"{row.genres} · {row.release_year if not str(row.release_year) == '<NA>' else '年份未知'} · 训练期评分 {row.rating_count} 条")
        score_col.metric("综合分", f"{row.final_score:.2f}", help=f"MLP预测 {row.mlp_prediction:.2f}")
        st.markdown("**推荐理由**")
        for reason in row.reasons:
            st.write(f"- {reason}")
        b1, b2, b3, b4 = st.columns(4)
        actions = [
            (b1, "加入想看", "want_to_watch", None),
            (b2, "标记已看", "watched", None),
            (b3, "喜欢", "want_to_watch", "like"),
            (b4, "不感兴趣", "want_to_watch", "not_interested"),
        ]
        for column, label, status, feedback in actions:
            if column.button(label, key=f"rec_{label}_{int(row.movie_id)}", width="stretch"):
                upsert_interaction(
                    profile_key, user_id, int(row.movie_id), status,
                    feedback=feedback, predicted_rating=float(row.mlp_prediction),
                )
                st.toast(f"已保存：{shown_title} · {label}")
                st.rerun()
