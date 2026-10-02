"""中文电影库：MovieLens中文检索与独立中国电影扩展目录。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from movie_rating.web.data_service import (
    build_prediction_frame, chinese_catalog_facets, chinese_rated_facets,
    load_chinese_rated_catalog, movie_statistics, search_chinese_catalog,
    search_chinese_rated_catalog, search_movies,
)
from movie_rating.web.database import (
    get_catalog_interactions, upsert_catalog_interaction,
)
from movie_rating.web.model_service import (
    load_cached_chinese_models, predict_chinese_ratings, predict_with_mlp,
)
from movie_rating.web.ui import ensure_session_state, page_intro, valid_external_url


user_id, profile_key = ensure_session_state()
page_intro(
    "中文电影库",
    "中文检索现有MovieLens影片，并浏览可独立扩充的中国电影目录。",
)

movielens = movie_statistics()
localized_count = int(movielens["title_zh"].ne("").sum())
catalog_all = search_chinese_catalog(limit=500)
c1, c2, c3, c4 = st.columns(4)
c1.metric("MovieLens中文片名", localized_count)
c2.metric("中国电影扩展库", len(catalog_all))
c3.metric("真实评分训练库", len(load_chinese_rated_catalog()))
c4.metric("当前匿名用户", f"#{user_id}")

st.info(
    "三类数据边界：MovieLens 100K中文映射使用主实验模型；300部真实评分库使用独立的"
    "MovieLens 25M中国电影模型；冷启动扩展库只提供资料检索与个人记录，不生成伪预测分。"
)

predictable_tab, rated_tab, catalog_tab = st.tabs([
    "MovieLens 100K中文检索", "中国电影真实评分库", "独立扩展库（冷启动）"
])

with predictable_tab:
    query = st.text_input(
        "输入中文、别名或英文片名",
        placeholder="例如：星球大战、北非谍影、Toy Story",
        key="localized_movie_query",
    )
    only_localized = st.checkbox("只显示已有中文片名的影片", value=True)
    matches = search_movies(query=query, limit=100)
    if only_localized:
        matches = matches[matches["title_zh"].ne("")].reset_index(drop=True)
    if matches.empty:
        st.info("当前索引没有匹配项；可改用英文片名，或在CSV中继续补充中文映射。")
    else:
        st.dataframe(
            matches[[
                "movie_id", "title_zh", "title", "aliases_zh", "genres",
                "release_year", "rating_count",
            ]].rename(columns={
                "movie_id": "MovieLens ID", "title_zh": "中文片名", "title": "英文片名",
                "aliases_zh": "中文别名", "genres": "类型", "release_year": "年份",
                "rating_count": "训练期评分数",
            }),
            hide_index=True, width="stretch", height=380,
        )
        selected_id = st.selectbox(
            "选择影片进行MLP预测",
            matches["movie_id"].astype(int).tolist(),
            format_func=lambda value: matches.loc[
                matches["movie_id"] == value, "display_title"
            ].iloc[0],
            key="localized_predict_movie",
        )
        if st.button("预测当前用户评分", type="primary"):
            prediction_frame = build_prediction_frame(user_id, [int(selected_id)])
            score = float(predict_with_mlp(prediction_frame)[0])
            st.metric("MLP预测评分", f"{score:.2f} / 5")
            st.caption("预测输入和模型均与论文实验保持一致；中文片名只用于检索与展示。")

with rated_tab:
    st.success(
        "该库由MovieLens 25M匿名真实评分与Wikidata元数据交叉构建："
        "共300部中国电影、55,483条评分、5,973名用户，并已独立训练随机森林、XGBoost和MLP。"
    )
    st.caption(
        "这里的用户编号属于独立中国电影实验，与侧边栏MovieLens 100K用户不是同一人；"
        "默认使用固定测试集RMSE最低的XGBoost。"
    )
    rated_origins, rated_languages = chinese_rated_facets()
    filter_form = st.form("rated_chinese_filters")
    f1, f2, f3, f4 = filter_form.columns([2, 1, 1, 1])
    rated_query = f1.text_input("中英文片名", placeholder="例如：重庆森林、Infernal Affairs")
    rated_origin = f2.selectbox("出品地区", ["全部", *rated_origins])
    rated_language = f3.selectbox("原始语言", ["全部", *rated_languages])
    rated_years = f4.slider("上映年份", 1930, 2019, (1930, 2019))
    filter_form.form_submit_button("筛选训练库")
    rated_movies = search_chinese_rated_catalog(
        query=rated_query,
        origin=rated_origin,
        language=rated_language,
        start_year=rated_years[0],
        end_year=rated_years[1],
        limit=300,
    )
    st.dataframe(
        rated_movies[["title_zh", "title_en", "release_year", "origins", "languages"]]
        .rename(columns={
            "title_zh": "中文片名", "title_en": "英文片名", "release_year": "年份",
            "origins": "出品地区", "languages": "原始语言",
        }),
        hide_index=True, width="stretch", height=360,
    )
    if rated_movies.empty:
        st.info("当前条件没有匹配影片。")
    else:
        p1, p2 = st.columns([2, 1])
        selected_index = p1.selectbox(
            "选择影片",
            rated_movies["cn_movie_index"].astype(int).tolist(),
            format_func=lambda value: rated_movies.loc[
                rated_movies["cn_movie_index"] == value, "title_zh"
            ].iloc[0],
            key="rated_chinese_movie",
        )
        chinese_user = int(p2.number_input(
            "中国电影实验用户", min_value=1, max_value=5973, value=1, step=1,
            help="独立匿名用户空间，仅用于中国电影扩展实验。",
        ))
        if st.button("预测中国电影评分", type="primary", key="predict_rated_chinese"):
            prediction = predict_chinese_ratings(chinese_user, [selected_index])
            row = prediction.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("XGBoost（默认）", f"{row['xgboost']:.2f} / 5")
            m2.metric("随机森林", f"{row['random_forest']:.2f} / 5")
            m3.metric("PyTorch MLP", f"{row['mlp']:.2f} / 5")
            loaded = load_cached_chinese_models()
            prep = loaded["metadata"]["preprocessing"]
            user_mean = float(prep["user_mean"][chinese_user])
            movie_mean = float(prep["movie_mean"][int(selected_index)])
            st.write(
                f"推荐依据：该用户训练期平均评分为 **{user_mean:.2f}**；"
                f"该影片训练期平均评分为 **{movie_mean:.2f}**；"
                "模型进一步结合用户、影片、交互次数及评分时间特征得到预测。"
            )
            selected_row = rated_movies[rated_movies["cn_movie_index"] == selected_index].iloc[0]
            if valid_external_url(selected_row["source_url"]):
                st.link_button("查看Wikidata资料", selected_row["source_url"])

with catalog_tab:
    origins, genres = chinese_catalog_facets()
    with st.form("chinese_catalog_filters"):
        q1, q2, q3, q4 = st.columns([2, 1, 1, 1])
        catalog_query = q1.text_input("片名或简介关键词", placeholder="例如：科幻、亲情、流浪地球")
        origin = q2.selectbox("地区", ["全部", *origins])
        genre = q3.selectbox("类型", ["全部", *genres])
        year_range = q4.slider("上映年份", 1930, 2025, (1930, 2025))
        st.form_submit_button("筛选扩展库")

    catalog = search_chinese_catalog(
        query=catalog_query,
        origin=origin,
        genre=genre,
        start_year=int(year_range[0]),
        end_year=int(year_range[1]),
        limit=100,
    )
    st.caption(f"找到 {len(catalog)} 部；基础目录可通过 data/catalog/chinese_movies.csv 继续扩充。")
    existing = get_catalog_interactions(profile_key)
    existing_ids = set(existing["catalog_id"]) if not existing.empty else set()
    if catalog.empty:
        st.info("当前筛选条件下没有影片。")
    for row in catalog.head(30).itertuples(index=False):
        with st.container(border=True):
            title_col, year_col = st.columns([4, 1])
            title_col.subheader(f"{row.title_zh} · {row.title_en}")
            title_col.caption(f"{row.origin} · {row.genres} · 目录ID {row.catalog_id}")
            year_col.metric("上映年份", int(row.release_year))
            st.write(row.overview_zh)
            if pd.notna(row.movielens_movie_id):
                st.success(f"已映射MovieLens ID {int(row.movielens_movie_id)}，可进入预测流程。")
            else:
                st.warning("冷启动影片：暂无MovieLens训练评分，不显示模型预测。")
            b1, b2, b3 = st.columns([1, 1, 2])
            if b1.button("加入想看", key=f"catalog_want_{row.catalog_id}"):
                upsert_catalog_interaction(profile_key, row.catalog_id, "want_to_watch")
                st.toast(f"已加入想看：{row.title_zh}")
                st.rerun()
            if b2.button("标记已看", key=f"catalog_seen_{row.catalog_id}"):
                upsert_catalog_interaction(profile_key, row.catalog_id, "watched")
                st.toast(f"已标记已看：{row.title_zh}")
                st.rerun()
            b3.caption("已保存到我的观影" if row.catalog_id in existing_ids else "尚未保存")
            if valid_external_url(row.source_url):
                st.link_button("查看资料来源", row.source_url)

st.caption(
    "资料说明：中文片名映射为课程项目人工校对索引；扩展库采用可编辑基础目录并逐条保留公开资料链接。"
    "事实字段可用于检索，不作为论文模型训练数据。"
)
