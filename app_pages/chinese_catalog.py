"""统一电影库：中国电影与其他国家电影共同检索和评分。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from movie_rating.web.ui import ensure_session_state, page_intro, valid_external_url
from movie_rating.web.unified_service import (
    predict_unified_item,
    recommend_unified_movies,
    search_unified_catalog,
    unified_catalog,
    unified_genres,
)


user_id, _ = ensure_session_state()
page_intro("电影库", "在同一个电影库中浏览中国电影与其他国家电影，并直接查看预测评分。")

catalog = unified_catalog()
china_count = int((catalog["model_space"] == "china").sum())
international_count = int((catalog["model_space"] == "international").sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("电影总数", f"{len(catalog):,}")
c2.metric("中国电影", f"{china_count:,}")
c3.metric("其他国家电影", f"{international_count:,}")
c4.metric("当前用户", f"#{user_id}")

with st.form("unified_movie_filters"):
    f1, f2, f3 = st.columns([2, 1, 2])
    query = f1.text_input("搜索电影", placeholder="输入中文片名，例如：流浪地球、玩具总动员")
    genre = f2.selectbox("电影类型", ["全部", *unified_genres()])
    years = f3.slider("上映年份", 1930, 2019, (1930, 2019))
    st.form_submit_button("查找电影", type="primary")

matches = search_unified_catalog(
    query=query,
    genre=genre,
    start_year=int(years[0]),
    end_year=int(years[1]),
    limit=500,
)
st.caption(f"找到 {len(matches)} 部电影")
if matches.empty:
    st.info("当前条件下没有匹配电影，请调整片名、类型或年份。")
    st.stop()

display = matches[[
    "title_zh", "origin", "category", "release_year", "rating_count", "rating_mean"
]].copy()
display["rating_mean"] = display["rating_mean"].map(lambda value: f"{float(value):.2f}")
st.dataframe(
    display.rename(columns={
        "title_zh": "电影名称",
        "origin": "国家或地区",
        "category": "电影类型",
        "release_year": "上映年份",
        "rating_count": "历史评分数",
        "rating_mean": "历史平均分",
    }),
    hide_index=True,
    width="stretch",
    height=430,
)

st.subheader("评分预测")
selected_key = st.selectbox(
    "选择电影",
    matches["item_key"].tolist(),
    format_func=lambda value: (
        f"{matches.loc[matches['item_key'] == value, 'title_zh'].iloc[0]} · "
        f"{matches.loc[matches['item_key'] == value, 'origin'].iloc[0]}"
    ),
    key="unified_catalog_movie",
)
if st.button("预测评分", type="primary", key="predict_unified_catalog"):
    with st.spinner("正在加载模型并计算评分……"):
        result = predict_unified_item(user_id, selected_key)
    st.metric("推荐预测分", f"{result['score']:.2f} / 5", help=f"默认使用{result['default_model']}")
    model_columns = st.columns(len(result["scores"]))
    for column, (model_name, value) in zip(model_columns, result["scores"].items()):
        column.metric(model_name, f"{value:.2f}")
    st.markdown("**评分依据**")
    for reason in result["reasons"]:
        st.write(f"- {reason}")
    source_url = result["movie"].get("source_url", "")
    if valid_external_url(source_url):
        st.link_button("查看电影资料", source_url)

st.subheader("综合推荐")
recommendation_count = st.slider("推荐数量", 2, 10, 6, key="library_recommendation_count")
if st.button("生成推荐", key="library_generate_recommendations"):
    with st.spinner("正在综合计算中国电影和其他国家电影……"):
        recommendations = recommend_unified_movies(
            user_id, pd.DataFrame(), recommendation_count, genre=genre
        )
    for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.subheader(f"{rank}. {row.title_zh}")
            left.caption(f"{row.origin} · {row.category} · {row.model_label}")
            right.metric("推荐分", f"{float(row.final_score):.2f}")
            for reason in row.reasons[:2]:
                st.write(f"- {reason}")
