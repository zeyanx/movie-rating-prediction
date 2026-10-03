"""统一电影库：中国电影与其他国家电影共同检索和评分。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from movie_rating.web.ui import ensure_session_state, page_intro, valid_external_url
from movie_rating.web.user_store import get_catalog_interactions
from movie_rating.web.ml25m_service import ensure_ml25m_catalog, recommend_ml25m_movies
from movie_rating.web.unified_service import (
    expanded_catalog,
    expanded_genres,
    large_catalog_ready,
    predict_expanded_item,
    predict_unified_item,
    recommend_unified_movies,
    search_expanded_catalog,
    search_unified_catalog,
    unified_catalog,
    unified_genres,
)


user_id, profile_key = ensure_session_state()
page_intro("电影库", "在同一个电影库中浏览中国电影与其他国家电影，并直接查看预测评分和推荐理由。")

large_ready = large_catalog_ready()
if not large_ready:
    with st.container(border=True):
        st.subheader("启用 MovieLens 25M 大型电影库")
        st.write("从 GroupLens 官方源下载并校验数据后，电影库将从 1,624 部扩展到 62,423 部。首次准备约需数分钟，完成后会复用本地缓存。")
        st.caption("原始数据不会打包进公开仓库；新增影片使用25M历史评分与账户类型偏好进行可解释预测。")
        if st.button("下载并启用大型电影库", type="primary", key="enable_ml25m"):
            try:
                with st.spinner("正在下载约250 MB官方数据并分块聚合2,500万条评分，请勿关闭页面……"):
                    ensure_ml25m_catalog()
                st.success("大型电影库准备完成")
                st.rerun()
            except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
                st.error(f"大型电影库准备失败：{exc}")
                st.info("可稍后重试；当前1,624部精选电影仍可正常使用。")

catalog = expanded_catalog() if large_ready else unified_catalog()

china_count = int((catalog["model_space"] == "china").sum())
international_count = len(catalog) - china_count
c1, c2, c3, c4 = st.columns(4)
c1.metric("电影总数", f"{len(catalog):,}")
c2.metric("中国电影", f"{china_count:,}")
c3.metric("其他国家电影", f"{international_count:,}")
c4.metric("数据规模", "25M大型库" if large_ready else "精选库")

with st.form("unified_movie_filters"):
    f1, f2, f3 = st.columns([2, 1, 2])
    query = f1.text_input("搜索电影", placeholder="输入中文片名，例如：流浪地球、玩具总动员")
    genres = expanded_genres() if large_ready else unified_genres()
    genre = f2.selectbox("电影类型", ["全部", *genres])
    valid_years = catalog["release_year"].dropna().astype(int)
    minimum_year = int(valid_years.min())
    maximum_year = int(valid_years.max())
    years = f3.slider("上映年份", minimum_year, maximum_year, (minimum_year, maximum_year))
    st.form_submit_button("查找电影", type="primary")

search_function = search_expanded_catalog if large_ready else search_unified_catalog
matches = search_function(
    query=query, genre=genre, start_year=int(years[0]), end_year=int(years[1]), limit=500
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
        result = (
            predict_expanded_item(user_id, selected_key)
            if large_ready else predict_unified_item(user_id, selected_key)
        )
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
if "library_refresh_page" not in st.session_state:
    st.session_state["library_refresh_page"] = 0
generate_column, refresh_column = st.columns(2)
generate = generate_column.button("生成推荐", key="library_generate_recommendations", width="stretch")
refresh = refresh_column.button("换一批", key="library_refresh_recommendations", width="stretch")
if refresh:
    st.session_state["library_refresh_page"] += 1
if generate:
    st.session_state["library_refresh_page"] = 0
if generate or refresh:
    with st.spinner("正在综合计算中国电影和其他国家电影……"):
        if large_ready:
            recommendations = recommend_ml25m_movies(
                user_id, recommendation_count, genre=genre,
                refresh_page=st.session_state["library_refresh_page"],
            )
        else:
            recommendations = recommend_unified_movies(
                user_id, pd.DataFrame(), recommendation_count, genre=genre,
                catalog_interactions=get_catalog_interactions(profile_key),
                refresh_page=st.session_state["library_refresh_page"],
            )
    for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.subheader(f"{rank}. {row.title_zh}")
            left.caption(f"{row.origin} · {row.category} · {row.model_label}")
            right.metric("推荐分", f"{float(row.final_score):.2f}")
            for reason in row.reasons[:2]:
                st.write(f"- {reason}")
