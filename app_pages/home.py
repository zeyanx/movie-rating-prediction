"""首页：项目概览、实验结论、搜索和推荐预览。"""

from __future__ import annotations

import streamlit as st

from movie_rating.web.config import PROJECT_ROOT
from movie_rating.web.data_service import (
    load_raw_tables, load_stage5_reports, search_movies,
)
from movie_rating.web.database import get_interactions
from movie_rating.web.recommendation import recommend_movies
from movie_rating.web.ui import MODEL_LABELS, ensure_session_state, page_intro, show_interaction_notice


user_id, profile_key = ensure_session_state()
page_intro("MovieLens 智能评分与推荐系统", "集成学习与神经网络电影评分预测对比研究 · 中文电影资料增强版")
show_interaction_notice()

ratings, movies, users = load_raw_tables()
reports = load_stage5_reports()
metrics = reports["metrics"].sort_values("rmse")
recommendation = reports["recommendation"]

st.subheader("数据与系统概况")
c1, c2, c3, c4 = st.columns(4)
c1.metric("原始评分", f"{len(ratings):,}")
c2.metric("匿名用户", f"{len(users):,}")
c3.metric("电影", f"{len(movies):,}")
c4.metric("当前用户", f"#{user_id}")

st.subheader("统一测试集模型表现")
display_metrics = metrics[["model", "rmse", "mae", "r2"]].copy()
display_metrics["模型"] = display_metrics["model"].map(MODEL_LABELS)
st.dataframe(
    display_metrics[["模型", "rmse", "mae", "r2"]].rename(
        columns={"rmse": "RMSE↓", "mae": "MAE↓", "r2": "R²↑"}
    ),
    hide_index=True,
    width="stretch",
)
st.success(
    f"默认模型：{MODEL_LABELS[recommendation['recommended_default_model']]}；"
    f"最佳集成模型：{MODEL_LABELS[recommendation['best_ensemble_model']]}。"
)
st.caption(recommendation["primary_reason"])

left, right = st.columns([1, 1])
with left:
    st.subheader("快速搜索")
    with st.form("home_search_form"):
        query = st.text_input("中英文电影标题", placeholder="例如：玩具总动员 / Toy Story")
        submitted = st.form_submit_button("搜索")
    if submitted or query:
        matches = search_movies(query=query, limit=10)
        if matches.empty:
            st.info("没有找到匹配电影，请缩短关键词或检查拼写。")
        else:
            st.dataframe(
                matches[["movie_id", "display_title", "genres", "release_year", "rating_count"]].rename(
                    columns={"display_title": "中英文片名"}
                ),
                hide_index=True, width="stretch",
            )

with right:
    st.subheader("为你推荐")
    interactions = get_interactions(profile_key)
    with st.spinner("正在使用MLP批量生成推荐预览……"):
        preview, metadata = recommend_movies(user_id, interactions, top_n=3)
    if preview.empty:
        st.info("当前没有可推荐电影，请在“我的观影”中检查排除项。")
    else:
        for row in preview.itertuples(index=False):
            shown_title = getattr(row, "display_title", row.title)
            st.markdown(f"**{shown_title}** · 预测 {row.mlp_prediction:.2f} 分")
            st.caption("；".join(row.reasons))
        if metadata["relaxed_popularity"]:
            st.caption("候选较少，预览已放宽最低热度条件。")

st.subheader("功能入口")
st.markdown(
    "- **电影详情**：搜索电影、查看历史统计并预测评分。\n"
    "- **中文电影库**：中文搜索MovieLens影片，并浏览独立中国电影扩展库。\n"
    "- **个性化推荐**：获取带理由的Top-N推荐并即时反馈。\n"
    "- **我的观影**：管理训练期历史与本地想看/已看记录。\n"
    "- **模型实验室**：对比五个模型、效率、Bootstrap和重要性。"
)

health = {
    "数据": len(ratings) == 100_000 and len(users) == 943 and len(movies) == 1_682,
    "MLP权重": (PROJECT_ROOT / "models" / "stage4" / "mlp_final.pt").is_file(),
    "第5阶段报告": (PROJECT_ROOT / "reports" / "stage5" / "metrics_comparison.csv").is_file(),
    "反馈数据库": (PROJECT_ROOT / "data" / "app" / "movie_app.db").is_file(),
}
st.caption("系统健康：" + " · ".join(f"{name}{'正常' if ok else '缺失'}" for name, ok in health.items()))
