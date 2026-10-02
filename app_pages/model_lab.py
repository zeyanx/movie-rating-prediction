"""模型实验室：复用第5阶段报告进行可交互展示。"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from movie_rating.web.data_service import build_prediction_frame, load_stage5_reports, search_movies
from movie_rating.web.model_service import predict_with_models
from movie_rating.web.ui import MODEL_LABELS, ensure_session_state, page_intro


user_id, _ = ensure_session_state()
page_intro("模型实验室", "读取第5阶段固定报告；本页面只做展示和推理，不重新训练模型。")
reports = load_stage5_reports()
metrics = reports["metrics"].copy()
metrics["模型"] = metrics["model"].map(MODEL_LABELS)

c1, c2, c3 = st.columns(3)
best = metrics.sort_values("rmse").iloc[0]
c1.metric("最低RMSE", f"{best['rmse']:.4f}", best["模型"])
c2.metric("最低MAE", f"{metrics['mae'].min():.4f}")
c3.metric("最高R²", f"{metrics['r2'].max():.4f}")

metric_name = st.radio("评价指标", ["rmse", "mae", "r2"], horizontal=True)
ascending = metric_name != "r2"
metric_chart = metrics.sort_values(metric_name, ascending=ascending)
st.plotly_chart(
    px.bar(metric_chart, x="模型", y=metric_name, color="模型", text_auto=".4f", title="五模型统一测试集对比"),
    width="stretch",
)
st.caption("RMSE、MAE越低越好；R²越高越好。测试集结果用于课程实验比较，不等同于真实线上推荐收益。")

tab1, tab2, tab3, tab4 = st.tabs(["效率", "Bootstrap", "特征重要性", "分段与误差"])
with tab1:
    efficiency = reports["efficiency"].copy()
    efficiency["模型"] = efficiency["model"].map(MODEL_LABELS)
    st.dataframe(
        efficiency[["模型", "artifact_size_mb", "load_time_mean_seconds", "inference_time_mean_seconds", "rows_per_second"]],
        hide_index=True, width="stretch",
    )
    st.plotly_chart(px.scatter(
        efficiency, x="artifact_size_mb", y="inference_time_mean_seconds",
        size="rows_per_second", color="模型", hover_name="模型",
        labels={"artifact_size_mb": "产物大小/MB", "inference_time_mean_seconds": "2万行推理时间/秒"},
    ), width="stretch")

with tab2:
    bootstrap = reports["bootstrap"].copy()
    bootstrap["集成模型"] = bootstrap["ensemble_model"].map(MODEL_LABELS)
    st.dataframe(
        bootstrap[["集成模型", "metric", "point_estimate_delta", "confidence_lower", "confidence_upper", "mlp_win_probability"]],
        hide_index=True, width="stretch",
    )
    st.caption("差值定义为MLP指标减集成模型指标；RMSE/MAE差值小于0代表MLP更优。")

with tab3:
    ensemble = reports["ensemble_importance"].copy()
    ensemble["模型"] = ensemble["model"].map(MODEL_LABELS)
    st.plotly_chart(px.bar(
        ensemble, x="feature_group", y="importance", color="模型", barmode="group",
        labels={"feature_group": "特征组", "importance": "树模型原生重要性"},
    ), width="stretch")
    mlp_imp = reports["mlp_importance"].sort_values("positive_normalized_importance", ascending=False)
    st.plotly_chart(px.bar(
        mlp_imp, x="feature_group", y="positive_normalized_importance",
        labels={"feature_group": "特征组", "positive_normalized_importance": "MLP排列重要性"},
    ), width="stretch")
    st.warning("树模型分裂重要性与MLP置换重要性的定义和评估数据不同，绝对数值不能直接互比，也不代表因果关系。")

with tab4:
    segments = reports["segments"]
    segment_type = st.selectbox("分段维度", sorted(segments["segment_type"].unique()))
    segment_view = segments[segments["segment_type"] == segment_type].copy()
    segment_view["模型"] = segment_view["model"].map(MODEL_LABELS)
    st.plotly_chart(px.bar(
        segment_view, x="segment_value", y="rmse", color="模型", barmode="group",
        labels={"segment_value": "分组", "rmse": "RMSE"},
    ), width="stretch")
    diagnostics = reports["rating_diagnostics"].copy()
    diagnostics["模型"] = diagnostics["model"].map(MODEL_LABELS)
    st.plotly_chart(px.line(
        diagnostics, x="actual_rating", y="mean_error", color="模型", markers=True,
        labels={"actual_rating": "真实评分", "mean_error": "平均误差"},
        title="不同真实评分下的平均误差",
    ), width="stretch")

recommendation = reports["recommendation"]
st.subheader("第5阶段工程建议")
st.success(
    f"默认：{MODEL_LABELS[recommendation['recommended_default_model']]}；"
    f"最佳集成：{MODEL_LABELS[recommendation['best_ensemble_model']]}；"
    f"降级：{MODEL_LABELS[recommendation['fallback_model']]}。"
)
st.caption(recommendation["limitations"])

st.subheader("当前用户 × 电影：多模型即时预测")
query = st.text_input("搜索一部电影", value="Toy Story", key="lab_movie_query")
movies = search_movies(query, limit=30)
if movies.empty:
    st.info("没有找到匹配电影。")
else:
    ids = movies["movie_id"].astype(int).tolist()
    movie_id = st.selectbox(
        "电影", ids,
        format_func=lambda value: f"{movies.loc[movies['movie_id'] == value, 'display_title'].iloc[0]} · ID {value}",
        key="lab_movie_selector",
    )
    if st.button("运行四模型预测", type="primary"):
        frame = build_prediction_frame(user_id, [int(movie_id)])
        with st.spinner("正在按需加载模型并推理……"):
            values = predict_with_models(frame, ["global_mean", "random_forest", "xgboost", "mlp"])
        prediction = pd.DataFrame({
            "模型": [MODEL_LABELS[name] for name in values.columns],
            "预测评分": [float(values.iloc[0][name]) for name in values.columns],
        })
        st.plotly_chart(px.bar(prediction, x="模型", y="预测评分", range_y=[1, 5], text_auto=".2f"), width="stretch")
