"""统一评分预测页。"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from movie_rating.web.ui import ensure_session_state, page_intro, valid_external_url
from movie_rating.web.unified_service import predict_unified_item, search_unified_catalog


user_id, _ = ensure_session_state()
page_intro("评分预测", "搜索中国电影或其他国家电影，查看模型评分、历史口碑和预测依据。")

with st.form("unified_prediction_search"):
    c1, c2 = st.columns([2, 1])
    query = c1.text_input("电影名称", placeholder="例如：流浪地球、星球大战")
    region = c2.selectbox("电影地区", ["全部", "中国电影", "其他国家和地区"])
    st.form_submit_button("查找电影", type="primary")

matches = search_unified_catalog(query=query, region=region, limit=200)
if matches.empty:
    st.info("没有找到匹配电影，请缩短关键词或切换电影地区。")
    st.stop()

selected_key = st.selectbox(
    "选择电影",
    matches["item_key"].tolist(),
    format_func=lambda value: (
        f"{matches.loc[matches['item_key'] == value, 'title_zh'].iloc[0]} · "
        f"{matches.loc[matches['item_key'] == value, 'origin'].iloc[0]}"
    ),
    key="unified_prediction_movie",
)
selected = matches[matches["item_key"] == selected_key].iloc[0]

title_col, info_col = st.columns([3, 1])
title_col.subheader(str(selected["title_zh"]))
title_col.write(
    f"国家或地区：{selected['origin']}  ·  "
    f"类型或语言：{selected['category']}  ·  "
    f"上映年份：{selected['release_year']}"
)
info_col.metric("历史平均分", f"{float(selected['rating_mean']):.2f} / 5")

if st.button("计算当前用户评分", type="primary"):
    with st.spinner("正在加载评分模型……"):
        result = predict_unified_item(user_id, selected_key)
    st.metric("推荐预测分", f"{result['score']:.2f} / 5", help=f"默认模型：{result['default_model']}")
    comparison = pd.DataFrame({
        "模型": list(result["scores"].keys()),
        "预测评分": list(result["scores"].values()),
    })
    st.plotly_chart(
        px.bar(
            comparison,
            x="模型",
            y="预测评分",
            range_y=[0.5, 5],
            text_auto=".2f",
            title="多模型评分比较",
        ),
        width="stretch",
    )
    st.markdown("**评分依据**")
    for reason in result["reasons"]:
        st.write(f"- {reason}")
    source_url = result["movie"].get("source_url", "")
    if valid_external_url(source_url):
        st.link_button("查看电影资料", source_url)
