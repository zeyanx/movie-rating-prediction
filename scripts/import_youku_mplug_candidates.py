"""从 Youku-mPLUG 分类标注中提取与电影有关的待审核候选标题。

重要边界：Youku-mPLUG 是视频—文本数据集，不是规范电影资料库。其“电影剪辑”和
“电影周边（预告/杂谈）”记录只能作为发现线索，不能直接写入正式中国电影目录。
本脚本只读取标注文件，不读取、复制或发布视频文件。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "imports" / "youku_mplug_movie_candidates.csv"
SOURCE_PROJECT = "X-PLUG/Youku-mPLUG"
SOURCE_URL = "https://github.com/X-PLUG/Youku-mPLUG"

# 来自官方 classname.json。两类均是视频片段/周边内容，而不是电影正片目录。
OFFICIAL_MOVIE_LABELS = {
    "电影周边（预告/杂谈）": 11,
    "电影剪辑": 34,
}
TITLE_FIELDS = ("title", "caption", "video_title", "text")
LABEL_FIELDS = ("label", "category", "class", "class_name")
VIDEO_ID_FIELDS = ("video_id:FILE", "video_id", "clip_name:FILE", "clip_name", "id")
OUTPUT_FIELDS = [
    "candidate_id",
    "source_title",
    "source_category",
    "source_video_ids",
    "duplicate_count",
    "review_status",
    "canonical_title_zh",
    "release_year",
    "origin",
    "genres",
    "overview_zh",
    "source_project",
    "source_url",
    "import_note",
]


def _first_value(record: dict[str, Any], fields: Iterable[str]) -> Any:
    """按候选字段顺序取得第一个非空值。"""
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return value
    return ""


def _normalise_title(value: Any) -> str:
    """仅清理空白字符，不擅自把短视频标题改写成电影片名。"""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe_key(title: str) -> str:
    """生成保守的去重键；保留中文、字母和数字，忽略常见标点及空白。"""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", title.casefold())


def _load_class_map(path: Path | None) -> dict[str, int]:
    """加载官方类别映射；未指定时使用脚本内置的两个电影相关类别。"""
    if path is None:
        return dict(OFFICIAL_MOVIE_LABELS)
    if not path.is_file():
        raise FileNotFoundError(f"类别文件不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("classname.json 顶层必须是对象。")
    mapping = {str(name): int(index) for name, index in payload.items()}
    missing = set(OFFICIAL_MOVIE_LABELS) - set(mapping)
    if missing:
        raise ValueError(f"类别文件缺少官方电影相关类别：{sorted(missing)}")
    return mapping


def _read_records(path: Path) -> list[dict[str, Any]]:
    """读取官方下游任务常见的 CSV、JSONL 或 JSON 标注格式。"""
    if not path.is_file():
        raise FileNotFoundError(f"标注文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path} 第 {line_number} 行不是JSON对象。")
                records.append(record)
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("records", []))
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"{path} 必须是对象数组，或包含 data/records 数组。")
        return payload
    raise ValueError(f"不支持的标注格式：{path.suffix}；仅支持 CSV、JSONL、JSON。")


def _resolve_label(raw_label: Any, class_map: dict[str, int]) -> str | None:
    """兼容官方CSV中的中文标签和JSONL中的整数标签。"""
    text = str(raw_label or "").strip()
    if text in OFFICIAL_MOVIE_LABELS:
        return text
    try:
        index = int(float(text))
    except (TypeError, ValueError):
        return None
    index_to_name = {index_value: name for name, index_value in class_map.items()}
    resolved = index_to_name.get(index)
    return resolved if resolved in OFFICIAL_MOVIE_LABELS else None


def extract_candidates(
    annotation_paths: Iterable[Path], class_map: dict[str, int]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """筛选两个电影相关视频类别并按原始标题去重。"""
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    total_records = 0
    movie_related_records = 0
    empty_title_records = 0

    for path in annotation_paths:
        for record in _read_records(path):
            total_records += 1
            label = _resolve_label(_first_value(record, LABEL_FIELDS), class_map)
            if label is None:
                continue
            movie_related_records += 1
            title = _normalise_title(_first_value(record, TITLE_FIELDS))
            if not title:
                empty_title_records += 1
                continue
            key = _dedupe_key(title)
            if not key:
                empty_title_records += 1
                continue
            video_id = _normalise_title(_first_value(record, VIDEO_ID_FIELDS))
            if key not in grouped:
                digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
                grouped[key] = {
                    "candidate_id": f"yk_{digest}",
                    "source_title": title,
                    "source_category": set(),
                    "source_video_ids": [],
                    "duplicate_count": 0,
                }
            item = grouped[key]
            item["source_category"].add(label)
            if video_id and video_id not in item["source_video_ids"]:
                item["source_video_ids"].append(video_id)
            item["duplicate_count"] += 1

    candidates: list[dict[str, Any]] = []
    for item in grouped.values():
        candidates.append({
            "candidate_id": item["candidate_id"],
            "source_title": item["source_title"],
            "source_category": "|".join(sorted(item["source_category"])),
            "source_video_ids": "|".join(item["source_video_ids"][:20]),
            "duplicate_count": item["duplicate_count"],
            "review_status": "pending",
            "canonical_title_zh": "",
            "release_year": "",
            "origin": "",
            "genres": "",
            "overview_zh": "",
            "source_project": SOURCE_PROJECT,
            "source_url": SOURCE_URL,
            "import_note": "仅为视频标题线索；须核实为电影正片并补全元数据后再进入正式目录",
        })

    stats = {
        "total_records": total_records,
        "movie_related_records": movie_related_records,
        "empty_title_records": empty_title_records,
        "unique_candidates": len(candidates),
    }
    return candidates, stats


def write_candidates(candidates: list[dict[str, Any]], output: Path, force: bool) -> Path:
    """写出UTF-8 BOM CSV，便于在Windows Excel中直接审核中文。"""
    if output.exists() and not force:
        raise FileExistsError(f"输出已存在：{output}；为保护人工审核结果，请加 --force 才能覆盖。")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(candidates)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提取Youku-mPLUG电影相关视频标题，生成待人工审核候选CSV。"
    )
    parser.add_argument("annotations", nargs="+", type=Path, help="官方标注CSV/JSONL/JSON路径")
    parser.add_argument("--classname", type=Path, help="可选：官方classname.json路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="候选CSV输出路径")
    parser.add_argument("--force", action="store_true", help="覆盖现有候选CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        class_map = _load_class_map(args.classname)
        candidates, stats = extract_candidates(args.annotations, class_map)
        output = write_candidates(candidates, args.output.resolve(), args.force)
        summary = {
            "dataset": "Youku-mPLUG annotation candidates",
            "source_url": SOURCE_URL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_files": [str(path.resolve()) for path in args.annotations],
            **stats,
            "important_note": "候选不是规范电影记录，未自动写入data/catalog/chinese_movies.csv。",
        }
        summary_path = output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"导入失败：{exc}", file=sys.stderr)
        return 1

    print(f"已读取 {stats['total_records']} 条标注。")
    print(f"电影相关视频标注 {stats['movie_related_records']} 条，去重候选 {stats['unique_candidates']} 条。")
    print(f"待审核文件：{output}")
    print("注意：未修改正式中国电影目录，候选必须人工核实后才能入库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
