"""Supabase账户认证、注册与个人推荐画像管理。"""

from __future__ import annotations

import re
from typing import Any

import streamlit as st
from supabase import Client, create_client

from .account_personalization import choose_proxy_user_id
from .unified_service import unified_genres


SESSION_CLIENT = "_supabase_client"
SESSION_PROFILE = "account_profile"


def supabase_settings() -> dict[str, str] | None:
    """安全读取Streamlit Secrets；本地未配置时返回None。"""
    try:
        section = st.secrets.get("supabase")
    except (FileNotFoundError, KeyError):
        return None
    if not section:
        return None
    url = str(section.get("url", "")).strip().rstrip("/")
    key = str(section.get("anon_key", "")).strip()
    if not url or not key:
        return None
    return {"url": url, "anon_key": key}


def cloud_accounts_enabled() -> bool:
    return supabase_settings() is not None


def get_supabase_client() -> Client:
    settings = supabase_settings()
    if settings is None:
        raise RuntimeError("尚未配置Supabase账户服务")
    current = st.session_state.get(SESSION_CLIENT)
    current_url = st.session_state.get("_supabase_url")
    if current is None or current_url != settings["url"]:
        current = create_client(settings["url"], settings["anon_key"])
        st.session_state[SESSION_CLIENT] = current
        st.session_state["_supabase_url"] = settings["url"]
    return current


def _auth_message(exc: Exception) -> str:
    """将远端异常转换为适合公开页面的有限提示，不回显请求或密钥。"""
    text = str(exc).lower()
    if "email_not_confirmed" in text or "email not confirmed" in text:
        return "邮箱尚未验证，请先打开确认邮件完成验证后再登录。"
    if "invalid login" in text or "invalid_credentials" in text:
        return "邮箱或密码不正确。"
    duplicate_markers = (
        "already registered",
        "user_already_exists",
        "email_exists",
        "duplicate key value",
        "users_email_partial_key",
    )
    if any(marker in text for marker in duplicate_markers):
        return "该邮箱已创建账户或确认邮件已发送，请检查收件箱后登录，不要重复提交。"
    if "password" in text and ("weak" in text or "least" in text):
        return "密码强度不足，请至少使用8位并混合字母与数字。"
    if (
        "over_email_send_rate_limit" in text
        or "can only request this after" in text
        or ("rate" in text and "limit" in text)
    ):
        return "确认邮件发送过于频繁，请等待至少60秒，并检查收件箱或垃圾邮件。"
    return "账户服务暂时无法完成请求，请稍后重试。"


def _valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value.strip()))


def _valid_password(value: str) -> bool:
    return len(value) >= 8 and bool(re.search(r"[A-Za-z]", value)) and bool(re.search(r"\d", value))


def _current_session(client: Client):
    try:
        return client.auth.get_session()
    except Exception:
        return None


def _load_profile(client: Client, account_id: str) -> dict[str, Any] | None:
    response = client.table("profiles").select("*").eq("user_id", account_id).limit(1).execute()
    return dict(response.data[0]) if response.data else None


def _save_profile(
    client: Client,
    account_id: str,
    display_name: str,
    favorite_genres: list[str],
) -> dict[str, Any]:
    proxy_user_id = choose_proxy_user_id(favorite_genres)
    payload = {
        "user_id": account_id,
        "display_name": display_name.strip(),
        "model_user_id": proxy_user_id,
        "favorite_genres": favorite_genres,
    }
    response = client.table("profiles").upsert(payload, on_conflict="user_id").execute()
    if not response.data:
        raise RuntimeError("个人资料保存失败")
    return dict(response.data[0])


def _render_login(client: Client) -> None:
    st.title("登录电影推荐系统")
    st.caption("创建个人账户后，你的观影记录、反馈和推荐画像会安全地独立保存。")
    login_tab, signup_tab = st.tabs(["登录", "创建账户"])
    with login_tab:
        with st.form("account_login_form"):
            email = st.text_input("邮箱", key="login_email")
            password = st.text_input("密码", type="password", key="login_password")
            submitted = st.form_submit_button("登录", type="primary", width="stretch")
        if submitted:
            if not _valid_email(email) or not password:
                st.error("请输入有效邮箱和密码。")
            else:
                try:
                    client.auth.sign_in_with_password({"email": email.strip(), "password": password})
                    st.rerun()
                except Exception as exc:
                    st.error(_auth_message(exc))

    with signup_tab:
        with st.form("account_signup_form"):
            display_name = st.text_input("昵称", max_chars=40)
            email = st.text_input("邮箱", key="signup_email")
            password = st.text_input(
                "密码", type="password", key="signup_password",
                help="至少8位，并同时包含字母和数字。",
            )
            confirm = st.text_input("确认密码", type="password")
            accepted = st.checkbox("我同意仅将邮箱用于登录，并保存我的观影与推荐数据。")
            submitted = st.form_submit_button("创建账户", type="primary", width="stretch")
        if submitted:
            if not display_name.strip():
                st.error("请输入昵称。")
            elif not _valid_email(email):
                st.error("请输入有效邮箱。")
            elif not _valid_password(password):
                st.error("密码至少8位，并同时包含字母和数字。")
            elif password != confirm:
                st.error("两次输入的密码不一致。")
            elif not accepted:
                st.error("请先确认账户数据用途。")
            else:
                try:
                    response = client.auth.sign_up({
                        "email": email.strip(),
                        "password": password,
                        "options": {"data": {"display_name": display_name.strip()}},
                    })
                    if response.session is None:
                        st.success(
                            "账户已创建，确认邮件已发送。请勿重复注册；检查收件箱或垃圾邮件，"
                            "完成验证后再返回登录。"
                        )
                    else:
                        st.session_state["pending_display_name"] = display_name.strip()
                        st.rerun()
                except Exception as exc:
                    st.error(_auth_message(exc))

    st.info("密码由Supabase Auth处理，本应用不会读取或保存明文密码。")


def _render_onboarding(client: Client, account_id: str, default_name: str) -> None:
    st.title("设置你的推荐偏好")
    st.caption("首次登录只需选择喜欢的电影类型，系统将生成初始推荐画像；之后会结合反馈继续调整。")
    with st.form("account_onboarding_form"):
        display_name = st.text_input("昵称", value=default_name, max_chars=40)
        favorites = st.multiselect(
            "喜欢的电影类型",
            unified_genres(),
            default=["剧情", "喜剧"],
            max_selections=6,
        )
        submitted = st.form_submit_button("保存并开始使用", type="primary")
    if submitted:
        if not display_name.strip():
            st.error("昵称不能为空。")
        elif not favorites:
            st.error("请至少选择一种喜欢的电影类型。")
        else:
            try:
                profile = _save_profile(client, account_id, display_name, favorites)
                st.session_state[SESSION_PROFILE] = profile
                st.session_state.pop("pending_display_name", None)
                st.rerun()
            except Exception as exc:
                st.error(_auth_message(exc))


def authenticate_or_stop() -> dict[str, Any] | None:
    """云端启用账户时强制认证；本地演示模式返回None。"""
    if not cloud_accounts_enabled():
        st.session_state.pop(SESSION_PROFILE, None)
        return None
    client = get_supabase_client()
    session = _current_session(client)
    if session is None or session.user is None:
        st.session_state.pop(SESSION_PROFILE, None)
        _render_login(client)
        st.stop()
    account_id = str(session.user.id)
    try:
        profile = _load_profile(client, account_id)
    except Exception as exc:
        st.error(_auth_message(exc))
        st.stop()
    if profile is None:
        metadata = getattr(session.user, "user_metadata", {}) or {}
        default_name = str(
            st.session_state.get("pending_display_name")
            or metadata.get("display_name")
            or str(session.user.email or "电影用户").split("@")[0]
        )
        _render_onboarding(client, account_id, default_name)
        st.stop()
    profile["email"] = str(session.user.email or "")
    st.session_state[SESSION_PROFILE] = profile
    return profile


def update_current_profile(display_name: str, favorite_genres: list[str]) -> dict[str, Any]:
    profile = st.session_state.get(SESSION_PROFILE)
    if not profile:
        raise RuntimeError("当前没有登录账户")
    updated = _save_profile(
        get_supabase_client(), str(profile["user_id"]), display_name, favorite_genres
    )
    updated["email"] = profile.get("email", "")
    st.session_state[SESSION_PROFILE] = updated
    return updated


def sign_out() -> None:
    if cloud_accounts_enabled():
        try:
            get_supabase_client().auth.sign_out()
        except Exception:
            # 即使远端会话已失效，也必须清除当前浏览器会话中的认证状态。
            pass
        finally:
            for key in [SESSION_PROFILE, SESSION_CLIENT, "_supabase_url"]:
                st.session_state.pop(key, None)
