"""MovieLens 25M 大型电影库的官方下载、校验、聚合和个性化统计服务。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import urllib.error
import urllib.request
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .config import PROJECT_ROOT
from .data_service import load_raw_tables, load_train_ratings
from .recommendation import compute_user_genre_profile


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
MD5_URL = DATASET_URL + ".md5"
README_URL = "https://files.grouplens.org/datasets/movielens/ml-25m-README.html"
OFFICIAL_MD5 = "6b51fb2759a8657d3bfcbfc42b592ada"
EXPECTED_RATINGS = 25_000_095
EXPECTED_USERS = 162_541
EXPECTED_MOVIES = 62_423
USER_AGENT = "MovieRatingCourseProject/1.0 (https://github.com/zeyanx/movie-rating-prediction)"
_BUILD_LOCK = threading.RLock()
GENRE_ZH = {
    "unknown": "未知", "Action": "动作", "Adventure": "冒险", "Animation": "动画",
    "Children": "儿童", "Children's": "儿童", "Comedy": "喜剧", "Crime": "犯罪",
    "Documentary": "纪录片", "Drama": "剧情", "Fantasy": "奇幻", "Film-Noir": "黑色电影",
    "Horror": "恐怖", "Musical": "歌舞", "Mystery": "悬疑", "Romance": "爱情",
    "Sci-Fi": "科幻", "Thriller": "惊悚", "War": "战争", "Western": "西部",
    "(no genres listed)": "未分类",
}


def translate_genres(value: object) -> str:
    labels = [GENRE_ZH.get(token, token) for token in str(value).split("|") if token]
    return "、".join(labels) if labels else "未分类"


def runtime_paths(root: Path | str = PROJECT_ROOT) -> dict[str, Path]:
    """返回大型库的本地缓存路径，不依赖盘符。"""
    project = Path(root).resolve()
    external = project / "data" / "external" / "ml-25m-source"
    generated = project / "data" / "ml25m_runtime"
    return {
        "archive": external / "ml-25m.zip",
        "md5": external / "ml-25m.zip.md5",
        "catalog": generated / "catalog.csv",
        "info": generated / "dataset_info.json",
    }


def calculate_md5(path: Path) -> str:
    """MD5仅用于核对GroupLens公布的文件完整性。"""
    digest = hashlib.md5()  # noqa: S324 - 必须与上游MD5值比较
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, timeout: float = 120.0) -> None:
    """仅从两个固定GroupLens地址下载，使用临时文件避免留下残包。"""
    if url not in {DATASET_URL, MD5_URL}:
        raise ValueError(f"拒绝非GroupLens官方地址：{url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            expected = response.headers.get("Content-Length")
        if expected is not None and temporary.stat().st_size != int(expected):
            raise OSError(f"下载长度异常：{temporary.stat().st_size} != {expected}")
        os.replace(temporary, destination)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"无法从GroupLens官方下载：{url}；请检查网络后重试") from exc


def _prepare_archive(
    root: Path | str = PROJECT_ROOT,
    archive_path: Path | str | None = None,
    force: bool = False,
) -> Path:
    paths = runtime_paths(root)
    archive = paths["archive"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive_path is not None:
        supplied = Path(archive_path).resolve()
        if not supplied.is_file():
            raise FileNotFoundError(f"指定压缩包不存在：{supplied}")
        if supplied != archive.resolve():
            shutil.copy2(supplied, archive)
    elif force or not archive.is_file():
        _download(DATASET_URL, archive)
    actual = calculate_md5(archive)
    if actual != OFFICIAL_MD5:
        archive.unlink(missing_ok=True)
        raise ValueError(f"MovieLens 25M MD5校验失败：{actual} != {OFFICIAL_MD5}")
    if force or not paths["md5"].is_file():
        try:
            _download(MD5_URL, paths["md5"])
        except RuntimeError:
            paths["md5"].write_text(f"{OFFICIAL_MD5}  ml-25m.zip\n", encoding="ascii")
    return archive


def _read_zip_table(bundle: zipfile.ZipFile, name: str, **kwargs: Any) -> pd.DataFrame:
    with bundle.open(name) as stream:
        return pd.read_csv(stream, **kwargs)


def _rating_statistics(bundle: zipfile.ZipFile) -> tuple[pd.DataFrame, dict[str, Any]]:
    """分块聚合2500万条评分，避免一次性占用大量内存。"""
    counts: pd.Series | None = None
    sums: pd.Series | None = None
    user_ids: set[int] = set()
    row_count = 0
    rating_min = float("inf")
    rating_max = float("-inf")
    with bundle.open("ml-25m/ratings.csv") as stream:
        chunks = pd.read_csv(
            stream,
            usecols=["userId", "movieId", "rating"],
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
            chunksize=1_000_000,
        )
        for chunk in chunks:
            row_count += len(chunk)
            rating_min = min(rating_min, float(chunk["rating"].min()))
            rating_max = max(rating_max, float(chunk["rating"].max()))
            user_ids.update(chunk["userId"].unique().astype(int).tolist())
            grouped = chunk.groupby("movieId")["rating"].agg(["count", "sum"])
            counts = grouped["count"] if counts is None else counts.add(grouped["count"], fill_value=0)
            sums = grouped["sum"] if sums is None else sums.add(grouped["sum"], fill_value=0)
    if counts is None or sums is None:
        raise ValueError("ratings.csv为空")
    if (row_count, len(user_ids), rating_min, rating_max) != (
        EXPECTED_RATINGS, EXPECTED_USERS, 0.5, 5.0
    ):
        raise ValueError(
            "MovieLens 25M评分规模异常："
            f"ratings={row_count}, users={len(user_ids)}, range={rating_min}-{rating_max}"
        )
    stats = pd.DataFrame({
        "movieId": counts.index.astype("int64"),
        "rating_count": counts.astype("int64").to_numpy(),
        "rating_sum": sums.astype("float64").to_numpy(),
    })
    stats["rating_mean"] = stats["rating_sum"] / stats["rating_count"]
    return stats, {
        "rating_count": row_count,
        "user_count": len(user_ids),
        "rating_min": rating_min,
        "rating_max": rating_max,
        "global_mean": float(stats["rating_sum"].sum() / row_count),
    }


def _imdb_number(value: object) -> int | None:
    match = re.search(r"tt(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def _title_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    text = re.sub(r"\s*\(\d{4}(?:-\d{4})?\)\s*$", "", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _title_year(value: object) -> int | None:
    match = re.search(r"\((\d{4})(?:-\d{4})?\)\s*$", str(value or ""))
    return int(match.group(1)) if match else None


def _model_mappings(
    root: Path,
) -> tuple[dict[tuple[str, int | None], dict[str, Any]], dict[int, dict[str, Any]]]:
    """用IMDb ID把25M影片对齐到两个已经训练好的模型空间。"""
    _, old_movies, _ = load_raw_tables(str(root))
    localization = pd.read_csv(root / "data" / "catalog" / "movie_localizations.csv")
    old = old_movies.merge(localization, on="movie_id", how="inner", validate="one_to_one")
    old["title_key"] = old["title"].map(_title_key)
    old["year_key"] = old["title"].map(_title_year)
    international = {
        (str(row.title_key), int(row.year_key) if pd.notna(row.year_key) else None): {
            "model_space": "international", "local_id": int(row.movie_id),
            "title_zh": str(row.title_zh),
            "aliases_zh": "" if pd.isna(row.aliases_zh) else str(row.aliases_zh),
        }
        for row in old.itertuples(index=False)
    }
    chinese_path = root / "data" / "catalog" / "chinese_rated_movies.csv"
    chinese = pd.read_csv(chinese_path, dtype={"imdb_id": "string"})
    chinese["imdb_number"] = chinese["imdb_id"].map(_imdb_number)
    china = {
        int(row.imdb_number): {
            "model_space": "china", "local_id": int(row.cn_movie_index),
            "title_zh": str(row.title_zh), "aliases_zh": str(row.title_en or ""),
            "catalog_id": str(row.catalog_id),
        }
        for row in chinese.itertuples(index=False) if pd.notna(row.imdb_number)
    }
    return international, china


def _build_catalog(root: Path, archive: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    with zipfile.ZipFile(archive) as bundle:
        damaged = bundle.testzip()
        if damaged is not None:
            raise zipfile.BadZipFile(f"ZIP内部文件损坏：{damaged}")
        required = {"ml-25m/movies.csv", "ml-25m/links.csv", "ml-25m/ratings.csv"}
        missing = required - set(bundle.namelist())
        if missing:
            raise ValueError(f"压缩包缺少文件：{sorted(missing)}")
        movies = _read_zip_table(bundle, "ml-25m/movies.csv")
        links = _read_zip_table(bundle, "ml-25m/links.csv")
        stats, info = _rating_statistics(bundle)
    if len(movies) != EXPECTED_MOVIES or movies["movieId"].duplicated().any():
        raise ValueError(f"MovieLens 25M电影规模或主键异常：{len(movies)}")
    catalog = movies.merge(links, on="movieId", how="left", validate="one_to_one")
    catalog = catalog.merge(stats.drop(columns="rating_sum"), on="movieId", how="left", validate="one_to_one")
    catalog["rating_count"] = catalog["rating_count"].fillna(0).astype("int64")
    catalog["rating_mean"] = catalog["rating_mean"].fillna(info["global_mean"]).astype(float)
    catalog["release_year"] = pd.to_numeric(
        catalog["title"].str.extract(r"\((\d{4})\)\s*$", expand=False), errors="coerce"
    ).astype("Int64")
    catalog["title_original"] = catalog["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

    international, china = _model_mappings(root)
    rows: list[dict[str, Any]] = []
    for row in catalog.itertuples(index=False):
        imdb = int(row.imdbId) if pd.notna(row.imdbId) else None
        title_year_key = (_title_key(row.title), int(row.release_year) if pd.notna(row.release_year) else None)
        mapping = china.get(imdb) or international.get(title_year_key) or {}
        model_space = mapping.get("model_space", "ml25m")
        local_id = int(mapping.get("local_id", row.movieId))
        title_zh = str(mapping.get("title_zh", row.title_original))
        aliases = str(mapping.get("aliases_zh", ""))
        # 大型目录始终使用25M影片ID作为唯一页面键；推理时再路由到相应模型空间。
        item_key = f"ml25m:{int(row.movieId)}"
        rows.append({
            "item_key": item_key,
            "model_space": model_space,
            "local_id": local_id,
            "ml25m_movie_id": int(row.movieId),
            "catalog_id": mapping.get("catalog_id", ""),
            "title_zh": title_zh,
            "title_original": str(row.title_original),
            "search_title": f"{title_zh} {row.title_original} {aliases}".strip(),
            "origin": "中国电影" if model_space == "china" else "其他国家和地区",
            "genres": str(row.genres),
            "release_year": row.release_year,
            "rating_count": int(row.rating_count),
            "rating_mean": float(row.rating_mean),
            "source_url": f"https://www.imdb.com/title/tt{imdb:07d}/" if imdb else "",
        })
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_MOVIES or result["ml25m_movie_id"].duplicated().any():
        raise ValueError("生成的大型电影库规模异常")
    info.update({
        "dataset_name": "MovieLens 25M",
        "official_download_url": DATASET_URL,
        "official_readme_url": README_URL,
        "archive_md5": OFFICIAL_MD5,
        "movie_count": len(result),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "columns": result.columns.tolist(),
    })
    return result, info


def _cache_valid(root: Path) -> bool:
    paths = runtime_paths(root)
    try:
        info = json.loads(paths["info"].read_text(encoding="utf-8"))
        return (
            paths["catalog"].is_file()
            and int(info["movie_count"]) == EXPECTED_MOVIES
            and int(info["rating_count"]) == EXPECTED_RATINGS
            and info["archive_md5"] == OFFICIAL_MD5
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def ensure_ml25m_catalog(
    root: Path | str = PROJECT_ROOT,
    archive_path: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """下载或读取官方ZIP，生成可供网页快速检索的聚合目录。"""
    project = Path(root).resolve()
    with _BUILD_LOCK:
        paths = runtime_paths(project)
        if not force and _cache_valid(project):
            return json.loads(paths["info"].read_text(encoding="utf-8"))
        archive = _prepare_archive(project, archive_path, force)
        catalog, info = _build_catalog(project, archive)
        paths["catalog"].parent.mkdir(parents=True, exist_ok=True)
        temporary_csv = paths["catalog"].with_suffix(".csv.part")
        catalog.to_csv(temporary_csv, index=False, encoding="utf-8-sig")
        os.replace(temporary_csv, paths["catalog"])
        temporary_json = paths["info"].with_suffix(".json.part")
        temporary_json.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_json, paths["info"])
        load_ml25m_catalog.clear()
        return info


def ml25m_catalog_ready(root: Path | str = PROJECT_ROOT) -> bool:
    return _cache_valid(Path(root).resolve())


@st.cache_data(show_spinner=False)
def load_ml25m_catalog(root: str = str(PROJECT_ROOT)) -> pd.DataFrame:
    """加载已生成的62,423部电影聚合目录。"""
    project = Path(root).resolve()
    if not _cache_valid(project):
        raise FileNotFoundError("MovieLens 25M大型电影库尚未准备")
    frame = pd.read_csv(runtime_paths(project)["catalog"], encoding="utf-8-sig")
    frame["release_year"] = pd.to_numeric(frame["release_year"], errors="coerce").astype("Int64")
    frame["catalog_id"] = frame["catalog_id"].fillna("")
    return frame


def _genre_adjustment(user_id: int, genres: str, root: str) -> tuple[float, list[str]]:
    train = load_train_ratings(root)
    _, movies, _ = load_raw_tables(root)
    profile = compute_user_genre_profile(int(user_id), train, movies)
    affinity = profile.set_index("genre")["affinity"].to_dict() if not profile.empty else {}
    tokens = [token for token in str(genres).split("|") if token and token != "(no genres listed)"]
    values = [(token, float(affinity[token])) for token in tokens if token in affinity]
    if not values:
        return 0.0, []
    values.sort(key=lambda item: item[1], reverse=True)
    return float(np.mean([value for _, value in values])), [name for name, _ in values[:2]]


def _genre_adjustment_from_affinity(
    genres: object, affinity: dict[str, float]
) -> tuple[float, list[str]]:
    """使用已经计算好的账户画像批量评分，避免对每部电影重复读取数据。"""
    tokens = [token for token in str(genres).split("|") if token and token != "(no genres listed)"]
    values = [(token, float(affinity[token])) for token in tokens if token in affinity]
    if not values:
        return 0.0, []
    values.sort(key=lambda item: item[1], reverse=True)
    return float(np.mean([value for _, value in values])), [name for name, _ in values[:2]]


def predict_ml25m_item(user_id: int, item_key: str, root: str = str(PROJECT_ROOT)) -> dict[str, Any]:
    """为未进入旧模型空间的影片生成可解释个性化统计预测。"""
    catalog = load_ml25m_catalog(root)
    matched = catalog[catalog["item_key"] == str(item_key)]
    if matched.empty:
        raise ValueError(f"大型电影库中不存在：{item_key}")
    movie = matched.iloc[0]
    info = json.loads(runtime_paths(root)["info"].read_text(encoding="utf-8"))
    prior_count = 50.0
    bayesian = (
        float(movie["rating_count"]) * float(movie["rating_mean"])
        + prior_count * float(info["global_mean"])
    ) / (float(movie["rating_count"]) + prior_count)
    affinity, preferred = _genre_adjustment(user_id, str(movie["genres"]), root)
    score = float(np.clip(bayesian + 0.35 * affinity, 0.5, 5.0))
    preferred_zh = [GENRE_ZH.get(name, name) for name in preferred]
    reasons = [
        f"25M贝叶斯历史评分为 {bayesian:.2f} 分",
        f"统计依据包含 {int(movie['rating_count']):,} 条该片评分",
    ]
    if preferred:
        reasons.append(f"结合当前账户对 {'、'.join(preferred_zh)} 类型的历史偏好进行修正")
    else:
        reasons.append("当前账户类型历史较少，本次以全体用户统计为主")
    return {
        "movie": movie.to_dict(), "score": score, "default_model": "个性化统计模型",
        "scores": {"个性化统计模型": score, "历史平均分": float(movie["rating_mean"])},
        "reasons": reasons,
    }


def recommend_ml25m_movies(
    user_id: int,
    top_n: int = 10,
    genre: str = "全部",
    refresh_page: int = 0,
    root: str = str(PROJECT_ROOT),
) -> pd.DataFrame:
    """从大型目录按贝叶斯口碑和账户类型偏好推荐，可稳定刷新下一批。"""
    if not 1 <= int(top_n) <= 20 or int(refresh_page) < 0:
        raise ValueError("推荐数量须为1至20，刷新页码不能为负数")
    catalog = load_ml25m_catalog(root).copy()
    if genre and genre != "全部":
        reverse_labels = {
            "未知": "unknown", "动作": "Action", "冒险": "Adventure", "动画": "Animation",
            "儿童": "Children", "喜剧": "Comedy", "犯罪": "Crime", "纪录片": "Documentary",
            "剧情": "Drama", "奇幻": "Fantasy", "黑色电影": "Film-Noir", "恐怖": "Horror",
            "歌舞": "Musical", "悬疑": "Mystery", "爱情": "Romance", "科幻": "Sci-Fi",
            "惊悚": "Thriller", "战争": "War", "西部": "Western", "未分类": "(no genres listed)",
        }
        token = reverse_labels.get(genre, genre)
        catalog = catalog[catalog["genres"].str.split("|").map(lambda values: token in values)]
    # 过滤极少评分条目，降低小样本偶然高分；仍保留足够宽的候选集。
    catalog = catalog[catalog["rating_count"] >= 20].copy()
    info = json.loads(runtime_paths(root)["info"].read_text(encoding="utf-8"))
    catalog["bayesian"] = (
        catalog["rating_count"] * catalog["rating_mean"] + 50.0 * float(info["global_mean"])
    ) / (catalog["rating_count"] + 50.0)
    train = load_train_ratings(root)
    _, movies, _ = load_raw_tables(root)
    profile = compute_user_genre_profile(int(user_id), train, movies)
    affinity = profile.set_index("genre")["affinity"].to_dict() if not profile.empty else {}
    adjustments = catalog["genres"].map(
        lambda value: _genre_adjustment_from_affinity(value, affinity)
    )
    catalog["genre_adjustment"] = adjustments.map(lambda value: value[0])
    catalog["preferred_genres"] = adjustments.map(lambda value: value[1])
    catalog["final_score"] = (catalog["bayesian"] + 0.35 * catalog["genre_adjustment"]).clip(0.5, 5.0)
    catalog["model_label"] = "25M个性化统计"
    catalog["category"] = catalog["genres"].map(translate_genres)
    catalog["reasons"] = catalog.apply(
        lambda row: [
            f"25M贝叶斯口碑得分 {float(row['bayesian']):.2f}",
            f"参考 {int(row['rating_count']):,} 条真实历史评分",
            (f"符合你对 {'、'.join(GENRE_ZH.get(name, name) for name in row['preferred_genres'])} 类型的偏好"
             if row["preferred_genres"] else "类型历史较少，以全体口碑为主"),
        ], axis=1,
    )
    ranked = catalog.sort_values(
        ["final_score", "rating_count", "ml25m_movie_id"],
        ascending=[False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    start = int(refresh_page) * int(top_n)
    return ranked.iloc[start:start + int(top_n)].reset_index(drop=True)
