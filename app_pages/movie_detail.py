"""电影详情：检索、统计、预测与反馈保存。"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from movie_rating.web.data_service import (
    all_genres, build_prediction_frame, get_user_history, rating_distribution, search_movies,
)
from movie_rating.web.database import get_interaction, upsert_interaction
from movie_rating.web.model_service import predict_with_mlp, predict_with_models
from movie_rating.web.ui import (
    FEEDBACK_LABELS, MODEL_LABELS, STATUS_LABELS, ensure_session_state,
    format_movie_option, page_intro, valid_external_url,
)


user_id, profile_key = ensure_session_state()
page_intro("电影详情", "检索电影、查看训练期统计，并使用已训练模型预测当前用户评分。")

with st.form("detail_search_form"):
    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("标题关键词", value=st.session_state.get("detail_query", ""))
    genre = c2.selectbox("类型", ["全部", *all_genres()])
    year_text = c3.text_input("发行年份（可选）", placeholder="1995")
    search_clicked = st.form_submit_button("查找电影")
if search_clicked:
    st.session_state["detail_query"] = query

release_year = int(year_text) if year_text.strip().isdigit() else None
matches = search_movies(query, genre, release_year, limit=100)
if matches.empty:
    st.info("没有找到匹配电影。请清空年份或缩短标题关键词。")
    st.stop()

options = matches["movie_id"].astype(int).tolist()
selected_id = st.selectbox(
    "选择电影",
    options,
    format_func=lambda value: format_movie_option(matches[matches["movie_id"] == value].iloc[0]),
    key="detail_movie_selector",
)
movie = matches[matches["movie_id"] == int(selected_id)].iloc[0]

title_col, score_col = st.columns([3, 1])
title_col.subheader(str(movie["display_title"]))
if movie.get("title_zh"):
    title_col.caption(f"中文别名：{movie['aliases_zh'] or '—'}")
title_col.write(f"类型：{movie['genres']}  ·  发行日期：{movie['release_date']}  ·  ID：{int(movie['movie_id'])}")
model_input = build_prediction_frame(user_id, [int(selected_id)])
with st.spinner("正在加载缓存的MLP并预测……"):
    mlp_score = float(predict_with_mlp(model_input)[0])
score_col.metric("MLP预测", f"{mlp_score:.2f} / 5")

if valid_external_url(movie.get("imdb_url")):
    st.link_button("打开IMDb资料", str(movie["imdb_url"]))

stat1, stat2, stat3 = st.columns(3)
stat1.metric("训练期评分数", int(movie["rating_count"]))
stat2.metric("训练期平均分", f"{float(movie['rating_mean']):.2f}")
stat3.metric("贝叶斯平滑口碑", f"{float(movie['bayesian_popularity']):.2f}")

distribution = rating_distribution(int(selected_id))
fig = px.bar(distribution, x="评分", y="数量", title="训练期评分分布", text_auto=True)
fig.update_layout(height=330)
st.plotly_chart(fig, width="stretch")

history = get_user_history(user_id)
historic = history[history["movie_id"] == int(selected_id)]
if historic.empty:
    st.info("当前用户在固定训练集中没有评价过这部电影。")
else:
    st.success(f"当前用户在固定训练集中的评分：{int(historic.iloc[0]['rating'])} 分")

if st.checkbox("显示多模型预测比较（会按需加载集成模型）", key="detail_compare_models"):
    with st.spinner("正在延迟加载随机森林和XGBoost……"):
        compared = predict_with_models(
            model_input, ["global_mean", "random_forest", "xgboost", "mlp"]
        )
    chart = pd.DataFrame({
        "模型": [MODEL_LABELS[name] for name in compared.columns],
        "预测评分": [float(compared.iloc[0][name]) for name in compared.columns],
    })
    st.plotly_chart(px.bar(chart, x="模型", y="预测评分", range_y=[1, 5], text_auto=".2f"), width="stretch")

st.subheader("保存到我的观影")
st.caption("本地SQLite会保存反馈；Community Cloud重启、休眠或重新构建后记录可能重置。")
existing = get_interaction(profile_key, int(selected_id)) or {}
status_values = list(STATUS_LABELS)
feedback_values = list(FEEDBACK_LABELS)
with st.form("detail_feedback_form"):
    c1, c2 = st.columns(2)
    status = c1.selectbox(
        "观影状态", status_values,
        index=status_values.index(existing.get("status", "want_to_watch")),
        format_func=lambda value: STATUS_LABELS[value],
    )
    feedback = c2.selectbox(
        "反馈", feedback_values,
        index=feedback_values.index(existing.get("feedback")) if existing.get("feedback") in feedback_values else 0,
        format_func=lambda value: FEEDBACK_LABELS[value],
    )
    use_rating = st.checkbox("记录个人评分", value=existing.get("personal_rating") is not None)
    rating = st.slider(
        "个人评分", 1.0, 5.0,
        float(existing.get("personal_rating") or 3.0), 0.5,
        disabled=not use_rating,
    )
    note = st.text_area("备注（最多500字）", value=str(existing.get("note", "")), max_chars=500)
    save = st.form_submit_button("保存记录", type="primary")
if save:
    upsert_interaction(
        profile_key, user_id, int(selected_id), status,
        rating if use_rating else None, feedback, note, mlp_score,
    )
    st.success("观影记录和反馈已保存。个性化推荐会读取最新记录。")
