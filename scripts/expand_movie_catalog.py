"""扩充电影中文片名并为中国电影补齐标准类型。

中文片名来自 Wikidata CC0；中国电影类型来自本项目已使用的 MovieLens 25M
影片元数据。脚本不会导出评分、原始用户编号或 MovieLens 25M 影片编号。
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_MOVIES = ROOT / "data" / "raw" / "movies.csv"
LOCALIZATIONS = ROOT / "data" / "catalog" / "movie_localizations.csv"
CHINESE_CATALOG = ROOT / "data" / "catalog" / "chinese_rated_movies.csv"
ML25_DIR = ROOT / "data" / "external" / "ml-25m-source" / "ml-25m"
CHINESE_LOCAL = ROOT / "data" / "chinese_training" / "movies.csv"
CACHE = ROOT / "data" / "external" / "unified-wikidata-titles.csv"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "MovieRatingResearch/1.1 (https://github.com/zeyanx/movie-rating-prediction)"

# MovieLens 25M中极少数影片没有类型标注；这里记录可审计的人工补充项。
GENRE_OVERRIDES = {"tt4613272": "Drama"}  # 《路边野餐》


def normalize_title(value: object) -> str:
    """生成跨MovieLens版本匹配使用的片名键。"""
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    text = re.sub(r"\s*\(\d{4}(?:-\d{4})?\)\s*$", "", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def extract_year(value: object) -> str:
    match = re.search(r"\((\d{4})(?:-\d{4})?\)\s*$", str(value or ""))
    return match.group(1) if match else ""


def sparql(query: str, retries: int = 5) -> dict[str, Any]:
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


def match_imdb_ids() -> pd.DataFrame:
    """把MovieLens 100K影片匹配到25M版本的IMDb编号。"""
    raw = pd.read_csv(RAW_MOVIES, encoding="utf-8-sig")
    ml25 = pd.read_csv(ML25_DIR / "movies.csv")
    links = pd.read_csv(ML25_DIR / "links.csv", dtype={"imdbId": str})
    raw["title_key"] = raw["title"].map(normalize_title)
    raw["year_key"] = raw["title"].map(extract_year)
    ml25["title_key"] = ml25["title"].map(normalize_title)
    ml25["year_key"] = ml25["title"].map(extract_year)

    # 同名同年份只在目标表唯一时匹配，避免续集和重名影片误关联。
    candidates = ml25.groupby(["title_key", "year_key"], as_index=False).filter(
        lambda group: len(group) == 1
    )
    matched = raw.merge(
        candidates[["movieId", "title_key", "year_key"]],
        on=["title_key", "year_key"], how="left", validate="many_to_one",
    )
    matched = matched.merge(links[["movieId", "imdbId"]], on="movieId", how="left")
    matched["imdb_id"] = matched["imdbId"].map(
        lambda value: f"tt{str(value).zfill(7)}" if pd.notna(value) else ""
    )
    return matched[["movie_id", "imdb_id"]]


def fetch_chinese_titles(imdb_ids: list[str], force: bool) -> pd.DataFrame:
    if CACHE.is_file() and not force:
        return pd.read_csv(CACHE, encoding="utf-8-sig", dtype=str).fillna("")
    rows: list[dict[str, str]] = []
    for start in range(0, len(imdb_ids), 80):
        values = " ".join(json.dumps(value) for value in imdb_ids[start : start + 80])
        query = f"""
SELECT ?imdb ?film ?titleZhHans ?titleZh WHERE {{
  VALUES ?imdb {{ {values} }}
  ?film wdt:P345 ?imdb.
  OPTIONAL {{ ?film rdfs:label ?titleZhHans FILTER(LANG(?titleZhHans) = "zh-hans") }}
  OPTIONAL {{ ?film rdfs:label ?titleZh FILTER(LANG(?titleZh) = "zh") }}
}}
"""
        payload = sparql(query)
        for binding in payload["results"]["bindings"]:
            imdb = binding["imdb"]["value"]
            title = binding.get("titleZhHans", binding.get("titleZh", {})).get("value", "")
            if title:
                rows.append({
                    "imdb_id": imdb,
                    "title_zh": title,
                    "source_url": binding["film"]["value"],
                })
        print(f"已查询中文片名：{min(start + 80, len(imdb_ids))}/{len(imdb_ids)}")
        time.sleep(0.2)
    frame = pd.DataFrame(rows).drop_duplicates("imdb_id", keep="first")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE, index=False, encoding="utf-8-sig")
    return frame


def expand_localizations(force: bool) -> int:
    existing = pd.read_csv(LOCALIZATIONS, encoding="utf-8-sig").fillna("")
    matched = match_imdb_ids()
    imdb_ids = sorted(value for value in matched["imdb_id"].unique() if value)
    wikidata = fetch_chinese_titles(imdb_ids, force)
    additions = matched.merge(wikidata, on="imdb_id", how="inner")
    additions = additions[["movie_id", "title_zh", "source_url"]].copy()
    additions["aliases_zh"] = ""
    additions["source_name"] = "维基数据"
    additions = additions[["movie_id", "title_zh", "aliases_zh", "source_name", "source_url"]]

    # 人工校对片名优先于Wikidata自动匹配结果。
    combined = pd.concat([existing, additions], ignore_index=True)
    combined = combined.drop_duplicates("movie_id", keep="first").sort_values("movie_id")
    combined.to_csv(LOCALIZATIONS, index=False, encoding="utf-8-sig")
    return len(combined)


def add_chinese_genres() -> int:
    catalog = pd.read_csv(CHINESE_CATALOG, encoding="utf-8-sig", dtype=str).fillna("")
    local = pd.read_csv(CHINESE_LOCAL, encoding="utf-8-sig", dtype=str).fillna("")
    ml25 = pd.read_csv(ML25_DIR / "movies.csv", dtype={"movieId": str}).fillna("")
    genres = local[["movieId", "imdb_id"]].merge(
        ml25[["movieId", "genres"]], on="movieId", how="left", validate="one_to_one"
    )[["imdb_id", "genres"]]
    catalog = catalog.drop(columns=["genres"], errors="ignore").merge(
        genres, on="imdb_id", how="left", validate="one_to_one"
    )
    catalog["genres"] = catalog.apply(
        lambda row: GENRE_OVERRIDES.get(row["imdb_id"], row["genres"]), axis=1
    )
    if catalog["genres"].fillna("").isin(["", "(no genres listed)"]).any():
        raise ValueError("部分中国电影缺少类型信息")
    ordered = [
        "catalog_id", "cn_movie_index", "title_zh", "title_en", "release_year",
        "origins", "languages", "genres", "wikidata_id", "imdb_id", "source_url",
    ]
    catalog[ordered].to_csv(CHINESE_CATALOG, index=False, encoding="utf-8-sig")
    return len(catalog)


def main() -> int:
    parser = argparse.ArgumentParser(description="扩充统一电影中文目录并补齐中国电影类型")
    parser.add_argument("--force", action="store_true", help="忽略Wikidata缓存并重新查询")
    args = parser.parse_args()
    for path in [RAW_MOVIES, LOCALIZATIONS, CHINESE_CATALOG, ML25_DIR / "movies.csv",
                 ML25_DIR / "links.csv", CHINESE_LOCAL]:
        if not path.is_file():
            raise FileNotFoundError(f"缺少扩充目录所需文件：{path}")
    localized = expand_localizations(args.force)
    chinese = add_chinese_genres()
    print(f"具有中文片名的其他国家电影：{localized}")
    print(f"已补齐类型的中国电影：{chinese}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
