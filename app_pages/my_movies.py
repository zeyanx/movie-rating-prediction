"""我的观影：区分训练期历史和网页本地记录。"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from movie_rating.web.data_service import get_user_history, movie_statistics
from movie_rating.web.database import (
    delete_catalog_interaction, delete_interaction, get_catalog_interactions,
    get_interactions, upsert_catalog_interaction, upsert_interaction,
)
from movie_rating.web.recommendation import compute_user_genre_profile
from movie_rating.web.ui import FEEDBACK_LABELS, STATUS_LABELS, ensure_session_state, page_intro


user_id, profile_key = ensure_session_state()
page_intro("我的观影", "集中查看历史评分、想看电影、已看电影和个人偏好。")

history = get_user_history(user_id)
local = get_interactions(profile_key)
movies = movie_statistics()
local_display = local.merge(
    movies[["movie_id", "title", "title_zh", "display_title", "genres"]],
    on="movie_id", how="left", validate="many_to_one"
) if not local.empty else local.copy()
catalog_local = get_catalog_interactions(profile_key)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("训练期评分", len(history))
c2.metric("训练期均分", f"{history['rating'].mean():.2f}" if not history.empty else "—")
c3.metric("想看", int((local.get("status", pd.Series(dtype=str)) == "want_to_watch").sum()))
c4.metric("已看", int((local.get("status", pd.Series(dtype=str)) == "watched").sum()))
c5.metric("扩展库记录", len(catalog_local))

history_tab, local_tab, catalog_tab, preference_tab = st.tabs(
    ["历史评分", "我的电影", "中国电影记录", "类型偏好"]
)
with history_tab:
    if history.empty:
        st.info("该用户在固定训练集中没有历史评分。")
    else:
        min_rating = st.slider("最低历史评分", 1, 5, 1, key="history_min_rating")
        shown = history[history["rating"] >= min_rating]
        st.dataframe(
            shown[["movie_id", "display_title", "genres", "rating", "timestamp"]].rename(
                columns={"display_title": "电影名称"}
            ),
            hide_index=True, width="stretch", height=420,
        )

with local_tab:
    if local_display.empty:
        st.info("还没有网页记录。可在电影详情或个性化推荐中加入想看、已看和反馈。")
    else:
        status_filter = st.selectbox("状态筛选", ["全部", *STATUS_LABELS], format_func=lambda value: STATUS_LABELS.get(value, value))
        shown = local_display if status_filter == "全部" else local_display[local_display["status"] == status_filter]
        display = shown.copy()
        display["状态"] = display["status"].map(STATUS_LABELS)
        display["反馈"] = display["feedback"].map(FEEDBACK_LABELS)
        st.dataframe(
            display[["movie_id", "display_title", "genres", "状态", "personal_rating", "反馈", "note", "updated_at"]].rename(
                columns={"display_title": "电影名称"}
            ),
            hide_index=True, width="stretch", height=350,
        )
        st.download_button(
            "下载当前网页记录CSV",
            data=display.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"用户_{user_id}_观影记录.csv",
            mime="text/csv",
        )

        selected_movie = st.selectbox(
            "编辑记录", display["movie_id"].astype(int).tolist(),
            format_func=lambda value: f"{display.loc[display['movie_id'] == value, 'display_title'].iloc[0]} · 编号 {value}",
            key="my_movies_edit_selector",
        )
        record = local[local["movie_id"] == int(selected_movie)].iloc[0]
        status_values = list(STATUS_LABELS)
        feedback_values = list(FEEDBACK_LABELS)
        with st.form("edit_local_interaction"):
            status = st.selectbox("状态", status_values, index=status_values.index(record["status"]), format_func=lambda v: STATUS_LABELS[v])
            feedback_value = record["feedback"] if pd.notna(record["feedback"]) else None
            feedback = st.selectbox("反馈", feedback_values, index=feedback_values.index(feedback_value), format_func=lambda v: FEEDBACK_LABELS[v])
            use_rating = st.checkbox("记录个人评分", value=pd.notna(record["personal_rating"]))
            personal = st.slider("个人评分", 1.0, 5.0, float(record["personal_rating"] if pd.notna(record["personal_rating"]) else 3.0), 0.5, disabled=not use_rating)
            note = st.text_area("备注", value=str(record["note"]), max_chars=500)
            save = st.form_submit_button("保存修改", type="primary")
        if save:
            upsert_interaction(
                profile_key, user_id, int(selected_movie), status,
                personal if use_rating else None, feedback, note,
                float(record["predicted_rating"]) if pd.notna(record["predicted_rating"]) else None,
            )
            st.success("记录已更新。")
            st.rerun()

        confirm = st.checkbox("我确认删除所选网页记录", key="confirm_delete_local")
        if st.button("删除记录", disabled=not confirm, type="secondary"):
            if delete_interaction(profile_key, int(selected_movie)):
                st.success("记录已删除。")
                st.rerun()

with catalog_tab:
    if catalog_local.empty:
        st.info("还没有中国电影记录，可在“电影库”页面浏览电影。")
    else:
        catalog_display = catalog_local.copy()
        catalog_display["状态"] = catalog_display["status"].map(STATUS_LABELS)
        st.dataframe(
            catalog_display[[
                "catalog_id", "title_zh", "release_year", "origin",
                "genres", "状态", "personal_rating", "note", "updated_at",
            ]].rename(columns={
                "catalog_id": "电影编号", "title_zh": "电影名称",
                "release_year": "年份", "origin": "地区", "genres": "类型",
                "personal_rating": "个人评分", "note": "备注", "updated_at": "更新时间",
            }),
            hide_index=True, width="stretch", height=350,
        )
        selected_catalog = st.selectbox(
            "编辑扩展库记录",
            catalog_display["catalog_id"].tolist(),
            format_func=lambda value: catalog_display.loc[
                catalog_display["catalog_id"] == value, "title_zh"
            ].iloc[0],
            key="catalog_record_selector",
        )
        record = catalog_local[catalog_local["catalog_id"] == selected_catalog].iloc[0]
        with st.form("edit_catalog_interaction"):
            status_values = list(STATUS_LABELS)
            catalog_status = st.selectbox(
                "状态", status_values, index=status_values.index(record["status"]),
                format_func=lambda value: STATUS_LABELS[value],
            )
            use_rating = st.checkbox(
                "记录个人评分", value=pd.notna(record["personal_rating"]),
                key="catalog_use_rating",
            )
            personal = st.slider(
                "个人评分", 1.0, 5.0,
                float(record["personal_rating"] if pd.notna(record["personal_rating"]) else 3.0),
                0.5, disabled=not use_rating, key="catalog_personal_rating",
            )
            note = st.text_area("备注", value=str(record["note"]), max_chars=500, key="catalog_note")
            save_catalog = st.form_submit_button("保存扩展库记录", type="primary")
        if save_catalog:
            upsert_catalog_interaction(
                profile_key, selected_catalog, catalog_status,
                personal if use_rating else None, note,
            )
            st.success("扩展库记录已更新。")
            st.rerun()
        confirm_catalog_delete = st.checkbox("我确认删除所选扩展库记录")
        if st.button("删除扩展库记录", disabled=not confirm_catalog_delete):
            if delete_catalog_interaction(profile_key, selected_catalog):
                st.success("扩展库记录已删除。")
                st.rerun()

with preference_tab:
    train_columns = ["user_id", "movie_id", "rating", "timestamp"]
    train_like = history[train_columns] if not history.empty else pd.DataFrame(columns=train_columns)
    profile = compute_user_genre_profile(user_id, train_like, movies, local)
    if profile.empty:
        st.info("历史信息不足，暂时无法形成类型偏好。")
    else:
        fig = px.bar(
            profile.head(15), x="genre", y="affinity", color="support",
            labels={"genre": "类型", "affinity": "相对个人均分偏差", "support": "历史支持数"},
            title="类型偏好（训练期评分 + 本地反馈轻量修正）",
        )
        st.plotly_chart(fig, width="stretch")
