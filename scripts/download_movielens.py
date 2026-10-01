"""下载、校验并转换 MovieLens 100K 数据集。

默认从 GroupLens 官方地址下载；网络不可用时，可以通过 ``--archive``
传入手动下载的 ml-100k.zip。脚本可重复执行，并提供 ``--force`` 强制重建。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE_PATH = EXTERNAL_DIR / "ml-100k.zip"
MD5_PATH = EXTERNAL_DIR / "ml-100k.zip.md5"
EXTRACTED_DIR = EXTERNAL_DIR / "ml-100k"

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MD5_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip.md5"
README_URL = "https://files.grouplens.org/datasets/movielens/ml-100k-README.txt"
# 官方 md5 文件中的固定校验值；保留该值可支持完全离线的 --archive 模式。
OFFICIAL_MD5 = "0e33842e24a9c977be4e0107933c0723"

RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
USERS_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]
MOVIES_COLUMNS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "genres",
]
GENRE_NAMES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def calculate_md5(path: Path) -> str:
    """分块计算文件 MD5，避免将整个压缩包一次性读入内存。"""
    digest = hashlib.md5()  # noqa: S324 - 此处仅用于文件完整性校验
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, retries: int = 3) -> None:
    """从官方地址下载文件，失败时重试并给出明确错误。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MovieRatingCourseProject/1.0"},
    )

    for attempt in range(1, retries + 1):
        try:
            print(f"正在下载：{url}（第 {attempt}/{retries} 次）")
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            temporary_path.replace(destination)
            print(f"下载完成：{destination}")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if temporary_path.exists():
                temporary_path.unlink()
            if attempt == retries:
                raise RuntimeError(
                    f"无法从官方地址下载 {url}。请检查网络或代理设置。原始错误：{exc}"
                ) from exc
            time.sleep(attempt * 2)


def read_official_md5() -> str:
    """读取官方校验文件；网络不可用时使用内置官方校验值。"""
    if not MD5_PATH.exists():
        try:
            download_file(MD5_URL, MD5_PATH)
        except RuntimeError as exc:
            print(f"警告：{exc}")
            print("将使用脚本内置的官方 MD5 继续校验。")
            return OFFICIAL_MD5

    content = MD5_PATH.read_text(encoding="ascii", errors="ignore")
    match = re.search(r"[0-9a-fA-F]{32}", content)
    if not match:
        raise ValueError(f"官方 MD5 文件格式异常：{MD5_PATH}")
    expected = match.group(0).lower()
    if expected != OFFICIAL_MD5:
        raise ValueError(
            "下载到的官方 MD5 与脚本记录不一致，请暂停并检查数据源："
            f"{expected} != {OFFICIAL_MD5}"
        )
    return expected


def prepare_archive(source_archive: Path | None, force: bool) -> str:
    """准备标准位置的 ZIP，并返回已经确认正确的 MD5。"""
    expected_md5 = read_official_md5()

    if source_archive is not None:
        source_archive = source_archive.expanduser().resolve()
        if not source_archive.is_file():
            raise FileNotFoundError(f"指定的压缩包不存在：{source_archive}")
        if source_archive != ARCHIVE_PATH.resolve():
            print(f"复制本地压缩包：{source_archive} -> {ARCHIVE_PATH}")
            shutil.copy2(source_archive, ARCHIVE_PATH)
    elif force or not ARCHIVE_PATH.exists():
        download_file(DATASET_URL, ARCHIVE_PATH)

    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"压缩包不存在：{ARCHIVE_PATH}")

    actual_md5 = calculate_md5(ARCHIVE_PATH)
    if actual_md5 != expected_md5:
        raise ValueError(
            "数据压缩包 MD5 校验失败，文件可能不完整或来源不正确："
            f"{actual_md5} != {expected_md5}"
        )
    print(f"MD5 校验通过：{actual_md5}")
    return actual_md5


def safe_extract(archive: Path, destination: Path) -> None:
    """先检查全部成员路径，再安全解压 ZIP，防止路径穿越。"""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        bad_member = zip_file.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"ZIP 内部文件损坏：{bad_member}")
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError(f"ZIP 包含不安全路径：{member.filename}")
        zip_file.extractall(destination)

    required = [EXTRACTED_DIR / "u.data", EXTRACTED_DIR / "u.item", EXTRACTED_DIR / "u.user"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"解压后缺少必要文件：{missing}")
    print(f"解压完成：{EXTRACTED_DIR}")


def join_genres(row: pd.Series) -> str:
    """将19个类型标志转换成管道符连接的类型字符串。"""
    selected = [genre for genre in GENRE_NAMES if int(row[genre]) == 1]
    return "|".join(selected) if selected else "unknown"


def convert_to_csv(archive_md5: str) -> None:
    """读取官方原始文件并生成三个统一格式的 CSV。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(
        EXTRACTED_DIR / "u.data",
        sep="\t",
        names=RATINGS_COLUMNS,
        header=None,
        dtype={"user_id": "int64", "movie_id": "int64", "rating": "int64", "timestamp": "int64"},
    )

    users = pd.read_csv(
        EXTRACTED_DIR / "u.user",
        sep="|",
        names=USERS_COLUMNS,
        header=None,
        encoding="latin-1",
        dtype={
            "user_id": "int64",
            "age": "int64",
            "gender": "string",
            "occupation": "string",
            "zip_code": "string",
        },
    )

    item_columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        *GENRE_NAMES,
    ]
    items = pd.read_csv(
        EXTRACTED_DIR / "u.item",
        sep="|",
        names=item_columns,
        header=None,
        encoding="latin-1",
    )
    items["genres"] = items[GENRE_NAMES].apply(join_genres, axis=1)
    movies = items[MOVIES_COLUMNS].copy()

    ratings.to_csv(RAW_DIR / "ratings.csv", index=False, encoding="utf-8-sig")
    users.to_csv(RAW_DIR / "users.csv", index=False, encoding="utf-8-sig")
    movies.to_csv(RAW_DIR / "movies.csv", index=False, encoding="utf-8-sig")

    dataset_info = {
        "dataset_name": "MovieLens 100K",
        "official_download_url": DATASET_URL,
        "official_readme_url": README_URL,
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_md5": archive_md5,
        "rating_count": int(len(ratings)),
        "user_count": int(len(users)),
        "movie_count": int(len(movies)),
        "rating_min": int(ratings["rating"].min()),
        "rating_max": int(ratings["rating"].max()),
        "columns": {
            "ratings.csv": RATINGS_COLUMNS,
            "users.csv": USERS_COLUMNS,
            "movies.csv": MOVIES_COLUMNS,
        },
    }
    (RAW_DIR / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"CSV 转换完成：{RAW_DIR}")


def processed_files_are_valid() -> bool:
    """快速判断已有CSV是否完整，完整时允许幂等跳过。"""
    try:
        ratings = pd.read_csv(RAW_DIR / "ratings.csv", encoding="utf-8-sig")
        users = pd.read_csv(RAW_DIR / "users.csv", encoding="utf-8-sig")
        movies = pd.read_csv(RAW_DIR / "movies.csv", encoding="utf-8-sig")
    except (OSError, ValueError, UnicodeError):
        return False

    return (
        list(ratings.columns) == RATINGS_COLUMNS
        and list(users.columns) == USERS_COLUMNS
        and list(movies.columns) == MOVIES_COLUMNS
        and len(ratings) == 100_000
        and len(users) == 943
        and len(movies) == 1_682
        and (RAW_DIR / "dataset_info.json").is_file()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并转换 MovieLens 100K 数据集")
    parser.add_argument(
        "--archive",
        type=Path,
        help="使用手动下载的 ml-100k.zip；适用于网络或代理受限环境",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（未指定 --archive 时）、解压并转换",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.force and args.archive is None and processed_files_are_valid():
        print("已存在通过快速检查的规范 CSV；无需重复下载和转换。")
        print("如需重建，请使用 --force。")
        return 0

    archive_md5 = prepare_archive(args.archive, args.force)
    safe_extract(ARCHIVE_PATH, EXTERNAL_DIR)
    convert_to_csv(archive_md5)
    print("MovieLens 100K 下载与转换成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
