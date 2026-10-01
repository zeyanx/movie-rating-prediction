"""为公开部署准备 MovieLens 运行时数据，不在仓库中重新分发原始数据。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from movie_rating.data import (
    calculate_file_sha256,
    load_raw_data,
    split_ratings,
    validate_raw_data,
)

from .config import PROJECT_ROOT


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MD5_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip.md5"
README_URL = "https://files.grouplens.org/datasets/movielens/ml-100k-README.txt"
OFFICIAL_MD5 = "0e33842e24a9c977be4e0107933c0723"
GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
USERS_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]
MOVIES_COLUMNS = [
    "movie_id", "title", "release_date", "video_release_date", "imdb_url", "genres",
]

_ASSET_LOCK = threading.RLock()


def calculate_md5(path: Path) -> str:
    """MD5仅用于核对GroupLens官方发布的文件完整性。"""
    digest = hashlib.md5()  # noqa: S324 - 与上游公开校验值保持一致。
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_official_file(url: str, destination: Path, timeout: float = 60.0) -> None:
    """只从明确传入的GroupLens官方URL下载，并使用临时文件原子替换。"""
    if url not in {DATASET_URL, MD5_URL}:
        raise ValueError(f"拒绝非GroupLens官方运行资源URL：{url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "MovieRatingCourseProject/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        os.replace(temporary, destination)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"无法从GroupLens官方下载运行数据：{url}。"
            "请检查网络后重试，不要改用未知镜像。"
        ) from exc


def read_expected_md5(md5_path: Path) -> str:
    content = Path(md5_path).read_text(encoding="ascii", errors="ignore")
    match = re.search(r"[0-9a-fA-F]{32}", content)
    if not match:
        raise ValueError("GroupLens MD5文件格式异常")
    expected = match.group(0).lower()
    if expected != OFFICIAL_MD5:
        raise ValueError(f"官方MD5与项目记录不一致：{expected} != {OFFICIAL_MD5}")
    return expected


def verify_archive_md5(archive_path: Path, expected: str = OFFICIAL_MD5) -> str:
    actual = calculate_md5(archive_path)
    if actual != expected:
        raise ValueError(f"MovieLens ZIP MD5校验失败：{actual} != {expected}")
    return actual


def safe_extract_zip(archive_path: Path, destination: Path) -> Path:
    """检查全部成员后再解压，拒绝绝对路径和目录穿越。"""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise zipfile.BadZipFile(f"ZIP内部文件损坏：{damaged}")
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(f"ZIP包含不安全路径：{member.filename}") from exc
        archive.extractall(destination)
    extracted = destination / "ml-100k"
    required = [extracted / "u.data", extracted / "u.item", extracted / "u.user"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"官方ZIP缺少必要文件：{missing}")
    return extracted


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def convert_official_files(extracted_dir: Path, raw_dir: Path, archive_md5: str) -> None:
    """按第1阶段完全相同的字段和编码生成三张规范CSV。"""
    ratings = pd.read_csv(
        extracted_dir / "u.data", sep="\t", names=RATINGS_COLUMNS, header=None,
        dtype={"user_id": "int64", "movie_id": "int64", "rating": "int64", "timestamp": "int64"},
    )
    users = pd.read_csv(
        extracted_dir / "u.user", sep="|", names=USERS_COLUMNS, header=None,
        encoding="latin-1",
        dtype={"user_id": "int64", "age": "int64", "gender": "string", "occupation": "string", "zip_code": "string"},
    )
    item_columns = ["movie_id", "title", "release_date", "video_release_date", "imdb_url", *GENRE_NAMES]
    items = pd.read_csv(
        extracted_dir / "u.item", sep="|", names=item_columns, header=None, encoding="latin-1"
    )
    items["genres"] = items[GENRE_NAMES].apply(
        lambda row: "|".join(name for name in GENRE_NAMES if int(row[name]) == 1) or "unknown",
        axis=1,
    )
    movies = items[MOVIES_COLUMNS].copy()
    if (len(ratings), len(users), len(movies)) != (100_000, 943, 1_682):
        raise ValueError("官方文件转换后的MovieLens 100K规模异常")
    _atomic_csv(ratings, raw_dir / "ratings.csv")
    _atomic_csv(users, raw_dir / "users.csv")
    _atomic_csv(movies, raw_dir / "movies.csv")
    _atomic_json(
        {
            "dataset_name": "MovieLens 100K",
            "official_download_url": DATASET_URL,
            "official_readme_url": README_URL,
            "converted_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_md5": archive_md5,
            "rating_count": 100_000,
            "user_count": 943,
            "movie_count": 1_682,
            "rating_min": 1,
            "rating_max": 5,
            "columns": {
                "ratings.csv": RATINGS_COLUMNS,
                "users.csv": USERS_COLUMNS,
                "movies.csv": MOVIES_COLUMNS,
            },
        },
        raw_dir / "dataset_info.json",
    )


def _raw_data_valid(root: Path) -> bool:
    try:
        data = load_raw_data(root / "data" / "raw")
        validate_raw_data(data)
        manifest = json.loads(
            (root / "data" / "processed" / "split_manifest.json").read_text(encoding="utf-8-sig")
        )
        return calculate_file_sha256(root / "data" / "raw" / "ratings.csv") == manifest["source_sha256"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def ensure_raw_data(
    root: Path | str = PROJECT_ROOT,
    archive_path: Path | str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """数据缺失时从官方源准备；已有且正确时不下载、不覆盖。"""
    target_root = Path(root).resolve()
    if _raw_data_valid(target_root):
        return {"downloaded": False, "archive_md5": OFFICIAL_MD5}
    external = target_root / "data" / "external"
    raw_dir = target_root / "data" / "raw"
    external.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = external / "ml-100k.zip"
    md5_file = external / "ml-100k.zip.md5"
    if archive_path is not None:
        supplied = Path(archive_path).resolve()
        if not supplied.is_file():
            raise FileNotFoundError(f"指定MovieLens压缩包不存在：{supplied}")
        if supplied != archive.resolve():
            shutil.copy2(supplied, archive)
        if not md5_file.exists():
            md5_file.write_text(f"{OFFICIAL_MD5}  ml-100k.zip\n", encoding="ascii")
    else:
        download_official_file(MD5_URL, md5_file, timeout)
        if not archive.exists():
            download_official_file(DATASET_URL, archive, timeout)
    expected = read_expected_md5(md5_file)
    actual = verify_archive_md5(archive, expected)
    extracted = safe_extract_zip(archive, external)
    convert_official_files(extracted, raw_dir, actual)
    if not _raw_data_valid(target_root):
        raise RuntimeError("运行时MovieLens CSV转换后验证失败")
    return {"downloaded": archive_path is None, "archive_md5": actual}


def ensure_fixed_split(root: Path | str = PROJECT_ROOT) -> dict[str, Any]:
    """只执行确定性数据划分，不训练任何模型，并核对第2阶段哈希。"""
    target_root = Path(root).resolve()
    processed = target_root / "data" / "processed"
    manifest_path = processed / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("部署包缺少固定划分manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    train_path = processed / "train_ratings.csv"
    test_path = processed / "test_ratings.csv"
    if train_path.is_file() and test_path.is_file():
        if (
            calculate_file_sha256(train_path) == manifest["train_sha256"]
            and calculate_file_sha256(test_path) == manifest["test_sha256"]
        ):
            return {"generated": False, "train_rows": 80_000, "test_rows": 20_000}
    data = load_raw_data(target_root / "data" / "raw")
    validate_raw_data(data)
    train, test = split_ratings(data.ratings, test_size=0.20, random_state=42)
    processed.mkdir(parents=True, exist_ok=True)
    train_tmp = train_path.with_suffix(".csv.part")
    test_tmp = test_path.with_suffix(".csv.part")
    train.to_csv(train_tmp, index=False, encoding="utf-8-sig")
    test.to_csv(test_tmp, index=False, encoding="utf-8-sig")
    if calculate_file_sha256(train_tmp) != manifest["train_sha256"]:
        train_tmp.unlink(missing_ok=True)
        test_tmp.unlink(missing_ok=True)
        raise ValueError("重新生成的训练集哈希与第2阶段不一致")
    if calculate_file_sha256(test_tmp) != manifest["test_sha256"]:
        train_tmp.unlink(missing_ok=True)
        test_tmp.unlink(missing_ok=True)
        raise ValueError("重新生成的测试集哈希与第2阶段不一致")
    os.replace(train_tmp, train_path)
    os.replace(test_tmp, test_path)
    return {"generated": True, "train_rows": len(train), "test_rows": len(test)}


def verify_runtime_asset_hashes(root: Path | str = PROJECT_ROOT) -> dict[str, Any]:
    target_root = Path(root).resolve()
    data = load_raw_data(target_root / "data" / "raw")
    validate_raw_data(data)
    manifest = json.loads(
        (target_root / "data" / "processed" / "split_manifest.json").read_text(encoding="utf-8-sig")
    )
    actual = {
        "source_sha256": calculate_file_sha256(target_root / "data" / "raw" / "ratings.csv"),
        "train_sha256": calculate_file_sha256(target_root / "data" / "processed" / "train_ratings.csv"),
        "test_sha256": calculate_file_sha256(target_root / "data" / "processed" / "test_ratings.csv"),
    }
    for key, value in actual.items():
        if value != manifest[key]:
            raise ValueError(f"运行时数据哈希异常：{key}")
    return {
        **actual,
        "ratings": len(data.ratings),
        "users": len(data.users),
        "movies": len(data.movies),
    }


def ensure_runtime_assets(
    root: Path | str = PROJECT_ROOT,
    archive_path: Path | str | None = None,
) -> dict[str, Any]:
    """线程安全地完成官方数据准备、固定划分与最终哈希审计。"""
    with _ASSET_LOCK:
        raw = ensure_raw_data(root, archive_path)
        split = ensure_fixed_split(root)
        hashes = verify_runtime_asset_hashes(root)
        return {"raw": raw, "split": split, "verification": hashes}


@st.cache_resource(show_spinner=False)
def ensure_runtime_assets_cached(root: str = str(PROJECT_ROOT)) -> dict[str, Any]:
    """Streamlit进程级缓存；返回内容只读使用。"""
    return ensure_runtime_assets(Path(root))
