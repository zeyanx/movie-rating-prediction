"""我的观影：区分训练期历史和网页本地记录。"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from movie_rating.web.data_service import get_user_history, load_raw_tables
from movie_rating.web.database import delete_interaction, get_interactions, upsert_interaction
from movie_rating.web.recommendation import compute_user_genre_profile
from movie_rating.web.ui import FEEDBACK_LABELS, STATUS_LABELS, ensure_session_state, page_intro


user_id, profile_key = ensure_session_state()
page_intro("我的观影", "固定训练期历史与网页新增记录分开呈现；本地修改仅写入SQLite。")
st.warning("云端SQLite不保证永久持久化：应用重启、休眠或重新构建后网页记录可能重置。")

history = get_user_history(user_id)
local = get_interactions(profile_key)
_, movies, _ = load_raw_tables()
local_display = local.merge(
    movies[["movie_id", "title", "genres"]], on="movie_id", how="left", validate="many_to_one"
) if not local.empty else local.copy()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("训练期评分", len(history))
c2.metric("训练期均分", f"{history['rating'].mean():.2f}" if not history.empty else "—")
c3.metric("想看", int((local.get("status", pd.Series(dtype=str)) == "want_to_watch").sum()))
c4.metric("已看", int((local.get("status", pd.Series(dtype=str)) == "watched").sum()))
c5.metric("喜欢 / 不喜欢", f"{int((local.get('feedback', pd.Series(dtype=str)) == 'like').sum())} / {int((local.get('feedback', pd.Series(dtype=str)) == 'dislike').sum())}")

history_tab, local_tab, preference_tab = st.tabs(["训练期历史", "网页记录", "类型偏好"])
with history_tab:
    if history.empty:
        st.info("该用户在固定训练集中没有历史评分。")
    else:
        min_rating = st.slider("最低历史评分", 1, 5, 1, key="history_min_rating")
        shown = history[history["rating"] >= min_rating]
        st.dataframe(
            shown[["movie_id", "title", "genres", "rating", "timestamp"]],
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
            display[["movie_id", "title", "genres", "状态", "personal_rating", "反馈", "note", "updated_at"]],
            hide_index=True, width="stretch", height=350,
        )
        st.download_button(
            "下载当前网页记录CSV",
            data=display.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"movielens_user_{user_id}_interactions.csv",
            mime="text/csv",
        )

        selected_movie = st.selectbox(
            "编辑记录", display["movie_id"].astype(int).tolist(),
            format_func=lambda value: f"{display.loc[display['movie_id'] == value, 'title'].iloc[0]} · ID {value}",
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
