"""首页：统一电影数据概览、搜索和推荐预览。"""

from __future__ import annotations

import streamlit as st

from movie_rating.web.config import PROJECT_ROOT
from movie_rating.web.data_service import load_stage5_reports
from movie_rating.web.database import get_interactions
from movie_rating.web.ui import MODEL_LABELS, ensure_session_state, page_intro
from movie_rating.web.unified_service import (
    recommend_unified_movies,
    search_unified_catalog,
    unified_catalog,
)


user_id, profile_key = ensure_session_state()
page_intro("智能电影评分与推荐系统", "集成学习与神经网络驱动的电影评分预测与个性化推荐。")

catalog = unified_catalog()
reports = load_stage5_reports()
metrics = reports["metrics"].sort_values("rmse")
china_count = int((catalog["model_space"] == "china").sum())
international_count = int((catalog["model_space"] == "international").sum())

st.subheader("电影数据概况")
c1, c2, c3, c4 = st.columns(4)
c1.metric("电影总数", f"{len(catalog):,}")
c2.metric("中国电影", f"{china_count:,}")
c3.metric("其他国家电影", f"{international_count:,}")
c4.metric("当前用户", f"#{user_id}")

left, right = st.columns([1, 1])
with left:
    st.subheader("快速搜索")
    query = st.text_input("电影名称", placeholder="例如：流浪地球、玩具总动员")
    matches = search_unified_catalog(query=query, limit=8)
    st.dataframe(
        matches[["title_zh", "origin", "release_year", "rating_mean"]].rename(columns={
            "title_zh": "电影名称",
            "origin": "国家或地区",
            "release_year": "上映年份",
            "rating_mean": "历史平均分",
        }),
        hide_index=True,
        width="stretch",
    )

with right:
    st.subheader("为你推荐")
    interactions = get_interactions(profile_key)
    with st.spinner("正在生成推荐预览……"):
        preview = recommend_unified_movies(user_id, interactions, top_n=4)
    for row in preview.itertuples(index=False):
        st.markdown(f"**{row.title_zh}** · 推荐 {float(row.final_score):.2f} 分")
        st.caption(f"{row.origin} · {row.reasons[0]}")

st.subheader("模型表现")
display_metrics = metrics[["model", "rmse", "mae", "r2"]].copy()
display_metrics["模型"] = display_metrics["model"].map(MODEL_LABELS)
st.dataframe(
    display_metrics[["模型", "rmse", "mae", "r2"]].rename(
        columns={"rmse": "均方根误差↓", "mae": "平均绝对误差↓", "r2": "决定系数↑"}
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("主要功能")
st.markdown(
    "- **电影库**：统一检索中国电影和其他国家电影。\n"
    "- **评分预测**：查看多模型预测评分和评分依据。\n"
    "- **个性化推荐**：获取混合电影推荐及推荐理由。\n"
    "- **我的观影**：管理评分历史与个人观影记录。\n"
    "- **模型分析**：比较不同模型的误差、效率和特征重要性。"
)

health = {
    "评分模型": (PROJECT_ROOT / "models" / "stage4" / "mlp_final.pt").is_file(),
    "中国电影模型": (PROJECT_ROOT / "models" / "chinese" / "xgboost.joblib").is_file(),
    "实验报告": (PROJECT_ROOT / "reports" / "stage5" / "metrics_comparison.csv").is_file(),
}
st.caption("系统状态：" + " · ".join(f"{name}{'正常' if ok else '缺失'}" for name, ok in health.items()))
