"""中文电影资料层、扩展目录和独立反馈闭环测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.web.data_service import search_chinese_catalog, search_movies
from movie_rating.web.database import (
    delete_catalog_interaction, get_catalog_interactions, get_chinese_movies,
    get_movie_localizations, initialize_database, upsert_catalog_interaction,
)


class ChineseCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "catalog.db"
        initialize_database(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_seed_counts_and_primary_keys(self) -> None:
        localizations = get_movie_localizations(self.db_path)
        catalog = get_chinese_movies(self.db_path)
        self.assertEqual(len(localizations), 80)
        self.assertEqual(len(catalog), 48)
        self.assertTrue(localizations["movie_id"].is_unique)
        self.assertTrue(catalog["catalog_id"].is_unique)

    def test_chinese_search_does_not_change_movie_ids(self) -> None:
        result = search_movies("玩具总动员", limit=10)
        self.assertEqual(result["movie_id"].astype(int).tolist(), [1])
        self.assertEqual(result.iloc[0]["title"], "Toy Story (1995)")
        alias = search_movies("北非谍影", limit=10)
        self.assertIn(483, alias["movie_id"].astype(int).tolist())

    def test_independent_catalog_filters(self) -> None:
        science_fiction = search_chinese_catalog("流浪地球", limit=10)
        self.assertEqual(science_fiction.iloc[0]["catalog_id"], "cn034")
        hong_kong = search_chinese_catalog(origin="中国香港", genre="犯罪", limit=100)
        self.assertIn("无间道", hong_kong["title_zh"].tolist())

    def test_catalog_interaction_crud(self) -> None:
        upsert_catalog_interaction(
            "movielens:1", "cn034", "want_to_watch", path=self.db_path
        )
        upsert_catalog_interaction(
            "movielens:1", "cn034", "watched", personal_rating=4.5,
            note="课堂演示", path=self.db_path,
        )
        rows = get_catalog_interactions("movielens:1", self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["title_zh"], "流浪地球")
        self.assertEqual(float(rows.iloc[0]["personal_rating"]), 4.5)
        with self.assertRaises(ValueError):
            upsert_catalog_interaction(
                "movielens:1", "missing", "watched", path=self.db_path
            )
        self.assertTrue(delete_catalog_interaction("movielens:1", "cn034", self.db_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
