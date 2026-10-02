"""Youku-mPLUG电影相关视频候选导入测试。"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.import_youku_mplug_candidates import (
    OFFICIAL_MOVIE_LABELS,
    _load_class_map,
    extract_candidates,
    main,
)


class YoukuMplugCandidateImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_csv_chinese_labels_are_filtered_and_deduplicated(self) -> None:
        source = self.root / "classification.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id:FILE", "title", "label"])
            writer.writeheader()
            writer.writerows([
                {"video_id:FILE": "a.mp4", "title": "流浪地球 预告", "label": "电影周边（预告/杂谈）"},
                {"video_id:FILE": "b.mp4", "title": "流浪地球　预告", "label": "电影周边（预告/杂谈）"},
                {"video_id:FILE": "c.mp4", "title": "霸王别姬经典片段", "label": "电影剪辑"},
                {"video_id:FILE": "d.mp4", "title": "装修技巧", "label": "房产装修"},
            ])
        candidates, stats = extract_candidates([source], dict(OFFICIAL_MOVIE_LABELS))
        self.assertEqual(stats["total_records"], 4)
        self.assertEqual(stats["movie_related_records"], 3)
        self.assertEqual(stats["unique_candidates"], 2)
        self.assertEqual(candidates[0]["duplicate_count"], 2)
        self.assertEqual(candidates[0]["review_status"], "pending")
        self.assertEqual(candidates[0]["canonical_title_zh"], "")

    def test_jsonl_numeric_labels_use_official_class_map(self) -> None:
        source = self.root / "classification.jsonl"
        records = [
            {"video_id": "a", "caption": "电影片段甲", "label": 34},
            {"video_id": "b", "caption": "篮球集锦", "label": 12},
            {"video_id": "c", "caption": "电影预告乙", "label": "11"},
        ]
        source.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in records),
            encoding="utf-8",
        )
        candidates, stats = extract_candidates([source], _load_class_map(None))
        self.assertEqual(stats["unique_candidates"], 2)
        self.assertEqual({row["source_category"] for row in candidates}, set(OFFICIAL_MOVIE_LABELS))

    def test_cli_writes_review_file_and_protects_existing_work(self) -> None:
        source = self.root / "records.json"
        source.write_text(
            json.dumps([{"video_id": "x", "title": "测试电影混剪", "label": "电影剪辑"}], ensure_ascii=False),
            encoding="utf-8",
        )
        output = self.root / "candidates.csv"
        self.assertEqual(main([str(source), "--output", str(output)]), 0)
        self.assertTrue(output.is_file())
        self.assertTrue(output.with_suffix(".summary.json").is_file())
        self.assertEqual(main([str(source), "--output", str(output)]), 1)
        self.assertEqual(main([str(source), "--output", str(output), "--force"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
