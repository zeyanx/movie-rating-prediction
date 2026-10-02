"""构建可训练的中国电影评分子集。

数据来源：MovieLens 25M 的匿名评分与IMDb映射，以及 Wikidata 的出品国家、
原始语言和中英文片名。只保留中国大陆、香港、台湾、澳门出品且原始语言为
中文语种的影片；不抓取豆瓣，也不使用 Youku 视频标题充当评分标签。
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
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external" / "ml-25m-source"
ARCHIVE = EXTERNAL / "ml-25m.zip"
MD5_FILE = EXTERNAL / "ml-25m.zip.md5"
EXTRACTED = EXTERNAL / "ml-25m"
LOCAL_OUTPUT = ROOT / "data" / "chinese_training"
PUBLIC_CATALOG = ROOT / "data" / "catalog" / "chinese_rated_movies.csv"
REPORT_DIR = ROOT / "reports" / "chinese_training"

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
MD5_URL = DATA_URL + ".md5"
OFFICIAL_MD5 = "6b51fb2759a8657d3bfcbfc42b592ada"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "MovieRatingCourseProject/1.0 (https://github.com/zeyanx/movie-rating-prediction)"

COUNTRY_QIDS = ["Q148", "Q8646", "Q865", "Q14773", "Q13426199"]
LANGUAGE_QIDS = ["Q7850", "Q9192", "Q9186", "Q727694", "Q34290", "Q1624231", "Q36495"]
COUNTRY_NAMES = {
    "Q148": "中国大陆",
    "Q8646": "中国香港",
    "Q865": "中国台湾",
    "Q14773": "中国澳门",
    "Q13426199": "中华民国（1912—1949）",
}
LANGUAGE_NAMES = {
    "Q7850": "中文",
    "Q9192": "官话",
    "Q9186": "粤语",
    "Q727694": "现代标准汉语",
    "Q34290": "吴语",
    "Q1624231": "闽南语",
    "Q36495": "闽南语族",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - 仅用于官方文件完整性校验
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """使用标准库下载，先写入.part，避免中断后留下伪完整文件。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=600) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    except (OSError, urllib.error.URLError) as exc:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(f"官方文件下载失败：{url}；{exc}") from exc


def ensure_archive(archive: Path | None, force: bool) -> Path:
    """准备并校验MovieLens 25M压缩包。"""
    source = archive.resolve() if archive else ARCHIVE
    if archive:
        if not source.is_file():
            raise FileNotFoundError(f"手动压缩包不存在：{source}")
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        if source != ARCHIVE.resolve():
            shutil.copy2(source, ARCHIVE)
        source = ARCHIVE
    elif force or not source.is_file() or source.stat().st_size == 0:
        download(DATA_URL, source)
    actual = md5(source)
    if actual != OFFICIAL_MD5:
        raise ValueError(f"MovieLens 25M MD5不匹配：期望{OFFICIAL_MD5}，实际{actual}")
    if force or not MD5_FILE.exists():
        try:
            download(MD5_URL, MD5_FILE)
        except RuntimeError:
            MD5_FILE.write_text(f"{OFFICIAL_MD5}  ml-25m.zip\n", encoding="ascii")
    return source


def safe_extract_required(archive: Path, force: bool) -> None:
    """只解压训练必需文件，并阻止ZIP路径穿越。"""
    required = {
        "ml-25m/links.csv",
        "ml-25m/movies.csv",
        "ml-25m/ratings.csv",
        "ml-25m/README.txt",
    }
    if not force and all((EXTERNAL / name).is_file() for name in required):
        return
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    base = EXTERNAL.resolve()
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        missing = required - names
        if missing:
            raise ValueError(f"压缩包缺少必要文件：{sorted(missing)}")
        for name in required:
            target = (EXTERNAL / name).resolve()
            if base not in target.parents:
                raise ValueError(f"检测到不安全ZIP路径：{name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def sparql(query: str, retries: int = 5) -> dict[str, Any]:
    """访问Wikidata官方SPARQL端点，带指数退避重试。"""
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode("utf-8")
    request = urllib.request.Request(
        WIKIDATA_ENDPOINT,
        data=data,
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Wikidata查询失败：{exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def binding_rows(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {name: value["value"] for name, value in binding.items()}
        for binding in payload["results"]["bindings"]
    ]


def discover_chinese_imdb_ids(force: bool) -> dict[str, str]:
    """发现满足国家和中文语种双重条件的IMDb ID及Wikidata实体。"""
    cache = EXTERNAL / "wikidata_chinese_imdb.json"
    if cache.is_file() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))
    countries = " ".join(f"wd:{qid}" for qid in COUNTRY_QIDS)
    languages = " ".join(f"wd:{qid}" for qid in LANGUAGE_QIDS)
    query = f"""
SELECT DISTINCT ?imdb ?film WHERE {{
  ?film wdt:P345 ?imdb; wdt:P495 ?country; wdt:P364 ?language.
  VALUES ?country {{ {countries} }}
  VALUES ?language {{ {languages} }}
}}
"""
    mapping = {
        row["imdb"]: row["film"].rsplit("/", 1)[-1]
        for row in binding_rows(sparql(query))
    }
    cache.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapping


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def fetch_details(imdb_ids: list[str], force: bool) -> pd.DataFrame:
    """分批获取匹配影片的CC0元数据，防止响应过大被代理截断。"""
    cache = EXTERNAL / "wikidata_chinese_details.csv"
    if cache.is_file() and not force:
        return pd.read_csv(cache, encoding="utf-8-sig", dtype=str).fillna("")
    rows: list[dict[str, str]] = []
    for start in range(0, len(imdb_ids), 100):
        values = " ".join(json.dumps(value) for value in imdb_ids[start : start + 100])
        query = f"""
SELECT ?imdb ?film ?titleZhHans ?titleZh ?titleEn ?country ?language ?date WHERE {{
  VALUES ?imdb {{ {values} }}
  ?film wdt:P345 ?imdb; wdt:P495 ?country; wdt:P364 ?language.
  OPTIONAL {{ ?film rdfs:label ?titleZhHans FILTER(LANG(?titleZhHans) = "zh-hans") }}
  OPTIONAL {{ ?film rdfs:label ?titleZh FILTER(LANG(?titleZh) = "zh") }}
  OPTIONAL {{ ?film rdfs:label ?titleEn FILTER(LANG(?titleEn) = "en") }}
  OPTIONAL {{ ?film wdt:P577 ?date }}
}}
"""
        rows.extend(binding_rows(sparql(query)))
        time.sleep(0.15)

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        imdb = row["imdb"]
        item = grouped.setdefault(
            imdb,
            {"imdb_id": imdb, "wikidata_id": _qid(row["film"]), "title_zh": "",
             "title_en": "", "origins": set(), "languages": set(), "dates": []},
        )
        if row.get("titleZhHans"):
            item["title_zh"] = row["titleZhHans"]
        elif row.get("titleZh") and not item["title_zh"]:
            item["title_zh"] = row["titleZh"]
        if row.get("titleEn"):
            item["title_en"] = row["titleEn"]
        country = _qid(row.get("country", ""))
        language = _qid(row.get("language", ""))
        if country in COUNTRY_NAMES:
            item["origins"].add(COUNTRY_NAMES[country])
        if language in LANGUAGE_NAMES:
            item["languages"].add(LANGUAGE_NAMES[language])
        if row.get("date"):
            item["dates"].append(row["date"])

    output = []
    for item in grouped.values():
        years = [int(match.group()) for value in item.pop("dates")
                 if (match := re.match(r"\d{4}", value))]
        item["release_year"] = min(years) if years else ""
        item["origins"] = "|".join(sorted(item["origins"]))
        item["languages"] = "|".join(sorted(item["languages"]))
        output.append(item)
    frame = pd.DataFrame(output)
    frame.to_csv(cache, index=False, encoding="utf-8-sig")
    return frame.fillna("")


def iterative_filter(ratings: pd.DataFrame, min_user: int, min_movie: int) -> pd.DataFrame:
    """反复过滤，直到每个保留用户和影片都满足最小交互数。"""
    result = ratings.copy()
    for _ in range(50):
        before = len(result)
        user_counts = result["userId"].value_counts()
        movie_counts = result["movieId"].value_counts()
        result = result[
            result["userId"].isin(user_counts[user_counts >= min_user].index)
            & result["movieId"].isin(movie_counts[movie_counts >= min_movie].index)
        ]
        if len(result) == before:
            return result.reset_index(drop=True)
    raise RuntimeError("交互过滤超过50轮仍未收敛")


def build_dataset(force: bool, min_user: int, min_movie: int) -> dict[str, Any]:
    links = pd.read_csv(EXTRACTED / "links.csv", dtype={"imdbId": str})
    links["imdb_id"] = "tt" + links["imdbId"].str.zfill(7)
    imdb_to_qid = discover_chinese_imdb_ids(force)
    matched = links[links["imdb_id"].isin(imdb_to_qid)].copy()
    movie_ids = set(matched["movieId"].astype(int))

    chunks = []
    for chunk in pd.read_csv(EXTRACTED / "ratings.csv", chunksize=1_000_000):
        selected = chunk[chunk["movieId"].isin(movie_ids)]
        if not selected.empty:
            chunks.append(selected)
    raw_subset = pd.concat(chunks, ignore_index=True)
    filtered = iterative_filter(raw_subset, min_user, min_movie)

    kept_movie_ids = sorted(filtered["movieId"].unique())
    kept_user_ids = sorted(filtered["userId"].unique())
    movie_index = {movie_id: index + 1 for index, movie_id in enumerate(kept_movie_ids)}
    user_index = {user_id: index + 1 for index, user_id in enumerate(kept_user_ids)}
    filtered["cn_user_id"] = filtered["userId"].map(user_index)
    filtered["cn_movie_index"] = filtered["movieId"].map(movie_index)

    details = fetch_details(sorted(matched["imdb_id"].unique()), force)
    movie_map = matched[["movieId", "imdb_id"]].drop_duplicates().merge(
        details, on="imdb_id", how="left", validate="many_to_one"
    )
    movie_map = movie_map[movie_map["movieId"].isin(kept_movie_ids)].copy()
    movie_map["cn_movie_index"] = movie_map["movieId"].map(movie_index)
    movie_map["catalog_id"] = "cnwd_" + movie_map["wikidata_id"].str.lower()
    movie_map["title_zh"] = movie_map["title_zh"].where(
        movie_map["title_zh"].astype(str).str.strip().ne(""), movie_map["title_en"]
    )
    if movie_map["wikidata_id"].eq("").any() or movie_map["title_zh"].eq("").any():
        raise ValueError("部分保留影片缺少Wikidata实体或可显示片名")

    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    local_ratings = filtered[["cn_user_id", "cn_movie_index", "rating", "timestamp"]]
    local_ratings.to_csv(LOCAL_OUTPUT / "ratings.csv", index=False, encoding="utf-8-sig")
    local_movies = movie_map.sort_values("cn_movie_index").reset_index(drop=True)
    local_movies.to_csv(LOCAL_OUTPUT / "movies.csv", index=False, encoding="utf-8-sig")

    # 公开目录只写入Wikidata CC0字段及本项目内部索引，不分发MovieLens原始ID或评分。
    public_columns = [
        "catalog_id", "cn_movie_index", "title_zh", "title_en", "release_year",
        "origins", "languages", "wikidata_id", "imdb_id",
    ]
    public = local_movies[public_columns].copy()
    public["source_url"] = "https://www.wikidata.org/wiki/" + public["wikidata_id"]
    PUBLIC_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(PUBLIC_CATALOG, index=False, encoding="utf-8-sig")

    summary = {
        "dataset_name": "Chinese-language MovieLens 25M research subset",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "movielens_url": DATA_URL,
        "movielens_md5": OFFICIAL_MD5,
        "wikidata_endpoint": WIKIDATA_ENDPOINT,
        "selection_rule": "target production country AND target Chinese original language",
        "minimum_user_ratings": min_user,
        "minimum_movie_ratings": min_movie,
        "candidate_imdb_ids": len(imdb_to_qid),
        "matched_movielens_movies": int(matched["movieId"].nunique()),
        "raw_subset_ratings": len(raw_subset),
        "ratings": len(filtered),
        "users": len(kept_user_ids),
        "movies": len(kept_movie_ids),
        "rating_min": float(filtered["rating"].min()),
        "rating_max": float(filtered["rating"].max()),
    }
    (LOCAL_OUTPUT / "dataset_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建中国电影真实评分训练集")
    parser.add_argument("--archive", type=Path, help="网络不可用时指定手动下载的ml-25m.zip")
    parser.add_argument("--force", action="store_true", help="重新下载缓存之外的处理与Wikidata查询")
    parser.add_argument("--min-user-ratings", type=int, default=5)
    parser.add_argument("--min-movie-ratings", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_user_ratings < 2 or args.min_movie_ratings < 2:
        print("最小交互数必须至少为2。", file=sys.stderr)
        return 2
    try:
        archive = ensure_archive(args.archive, args.force)
        safe_extract_required(archive, args.force)
        summary = build_dataset(args.force, args.min_user_ratings, args.min_movie_ratings)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("中国电影评分训练集构建完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
