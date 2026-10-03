"""用户私有数据存储：云端Supabase或本地SQLite。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from . import database as local_db
from .auth_service import SESSION_PROFILE, cloud_accounts_enabled, get_supabase_client
from .chinese_service import load_chinese_rated_catalog


INTERACTION_COLUMNS = [
    "id", "profile_key", "user_id", "movie_id", "status", "personal_rating",
    "feedback", "note", "predicted_rating", "created_at", "updated_at",
]


def _cloud_profile() -> dict | None:
    profile = st.session_state.get(SESSION_PROFILE)
    return dict(profile) if cloud_accounts_enabled() and profile else None


def _cloud_mode(path: Path | str | None) -> bool:
    return path is None and _cloud_profile() is not None


def _normalize_interactions(rows: list[dict], profile: dict) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)
    frame["profile_key"] = str(profile["user_id"])
    frame["user_id"] = frame["model_user_id"].astype(int)
    return frame.reindex(columns=INTERACTION_COLUMNS)


def get_interactions(profile_key: str, path: Path | str | None = None) -> pd.DataFrame:
    profile = _cloud_profile()
    if not _cloud_mode(path) or profile is None:
        return local_db.get_interactions(profile_key, path)
    response = (
        get_supabase_client().table("account_interactions").select("*")
        .eq("account_id", str(profile["user_id"]))
        .order("updated_at", desc=True).execute()
    )
    return _normalize_interactions(response.data or [], profile)


def upsert_interaction(
    profile_key: str,
    user_id: int | None,
    movie_id: int,
    status: str,
    personal_rating: float | None = None,
    feedback: str | None = None,
    note: str = "",
    predicted_rating: float | None = None,
    path: Path | str | None = None,
) -> None:
    profile = _cloud_profile()
    if not _cloud_mode(path) or profile is None:
        return local_db.upsert_interaction(
            profile_key, user_id, movie_id, status, personal_rating,
            feedback, note, predicted_rating, path,
        )
    local_db._validate_interaction(  # 复用经过测试的输入边界。
        profile_key, user_id, movie_id, status, personal_rating,
        feedback, note, predicted_rating,
    )
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "account_id": str(profile["user_id"]),
        "model_user_id": int(user_id or profile["model_user_id"]),
        "movie_id": int(movie_id),
        "status": status,
        "personal_rating": personal_rating,
        "feedback": feedback,
        "note": str(note),
        "predicted_rating": predicted_rating,
        "updated_at": now,
    }
    get_supabase_client().table("account_interactions").upsert(
        payload, on_conflict="account_id,movie_id"
    ).execute()


def delete_interaction(
    profile_key: str, movie_id: int, path: Path | str | None = None
) -> bool:
    profile = _cloud_profile()
    if not _cloud_mode(path) or profile is None:
        return local_db.delete_interaction(profile_key, movie_id, path)
    response = (
        get_supabase_client().table("account_interactions").delete()
        .eq("account_id", str(profile["user_id"]))
        .eq("movie_id", int(movie_id)).execute()
    )
    return bool(response.data)


def get_feedback_fingerprint(profile_key: str, path: Path | str | None = None) -> str:
    frame = get_interactions(profile_key, path)
    payload = frame.to_json(orient="records", date_format="iso")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_catalog_interactions(
    profile_key: str, path: Path | str | None = None
) -> pd.DataFrame:
    profile = _cloud_profile()
    if not _cloud_mode(path) or profile is None:
        return local_db.get_catalog_interactions(profile_key, path)
    response = (
        get_supabase_client().table("account_catalog_interactions").select("*")
        .eq("account_id", str(profile["user_id"]))
        .order("updated_at", desc=True).execute()
    )
    frame = pd.DataFrame(response.data or [])
    if frame.empty:
        return frame
    base_catalog = local_db.get_chinese_movies()[[
        "catalog_id", "title_zh", "title_en", "release_year", "origin", "genres"
    ]]
    rated_catalog = load_chinese_rated_catalog()[[
        "catalog_id", "title_zh", "title_en", "release_year", "origins", "genres"
    ]].rename(columns={"origins": "origin"})
    catalog = pd.concat([rated_catalog, base_catalog], ignore_index=True).drop_duplicates(
        "catalog_id", keep="first"
    )
    return frame.merge(
        catalog[["catalog_id", "title_zh", "title_en", "release_year", "origin", "genres"]],
        on="catalog_id", how="left", validate="many_to_one",
    )


def upsert_catalog_interaction(
    profile_key: str,
    catalog_id: str,
    status: str,
    personal_rating: float | None = None,
    note: str = "",
    path: Path | str | None = None,
) -> None:
    profile = _cloud_profile()
    if not _cloud_mode(path) or profile is None:
        return local_db.upsert_catalog_interaction(
            profile_key, catalog_id, status, personal_rating, note, path
        )
    if status not in local_db.VALID_STATUSES:
        raise ValueError(f"非法观影状态：{status}")
    if personal_rating is not None and not 1 <= float(personal_rating) <= 5:
        raise ValueError("个人评分必须位于1至5")
    if len(str(note)) > 500:
        raise ValueError("备注不能超过500字符")
    valid_ids = set(local_db.get_chinese_movies()["catalog_id"].astype(str))
    valid_ids.update(load_chinese_rated_catalog()["catalog_id"].astype(str))
    if str(catalog_id) not in valid_ids:
        raise ValueError(f"扩展电影ID不存在：{catalog_id}")
    payload = {
        "account_id": str(profile["user_id"]),
        "catalog_id": str(catalog_id),
        "status": status,
        "personal_rating": personal_rating,
        "note": str(note),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    get_supabase_client().table("account_catalog_interactions").upsert(
        payload, on_conflict="account_id,catalog_id"
    ).execute()


def delete_catalog_interaction(
    profile_key: str, catalog_id: str, path: Path | str | None = None
) -> bool:
    profile = _cloud_profile()
    if not _cloud_mode(path) or profile is None:
        return local_db.delete_catalog_interaction(profile_key, catalog_id, path)
    response = (
        get_supabase_client().table("account_catalog_interactions").delete()
        .eq("account_id", str(profile["user_id"]))
        .eq("catalog_id", str(catalog_id)).execute()
    )
    return bool(response.data)
