"""公开账户的资料与推荐偏好设置。"""

from __future__ import annotations

import streamlit as st

from movie_rating.web.auth_service import SESSION_PROFILE, update_current_profile
from movie_rating.web.ui import page_intro
from movie_rating.web.unified_service import unified_genres


profile = st.session_state.get(SESSION_PROFILE)
page_intro("账户设置", "管理昵称与冷启动类型偏好。修改后会重新匹配推荐参考画像。")

if not profile:
    st.info("当前处于本地演示模式；配置云端账户服务后可使用个人账户设置。")
    st.stop()

st.text_input("登录邮箱", value=str(profile.get("email", "")), disabled=True)
with st.form("account_settings_form"):
    display_name = st.text_input(
        "昵称", value=str(profile.get("display_name", "")), max_chars=40
    )
    favorite_genres = st.multiselect(
        "喜欢的电影类型",
        unified_genres(),
        default=list(profile.get("favorite_genres") or []),
        max_selections=6,
    )
    submitted = st.form_submit_button("保存账户设置", type="primary")

if submitted:
    if not display_name.strip():
        st.error("昵称不能为空。")
    elif not favorite_genres:
        st.error("请至少选择一种喜欢的电影类型。")
    else:
        with st.spinner("正在更新个性化画像……"):
            update_current_profile(display_name, favorite_genres)
        st.success("账户设置已保存，推荐画像已经更新。")
        st.rerun()

st.caption("账户数据由Supabase Auth与行级安全策略隔离；其他用户无法读取你的观影记录。")
