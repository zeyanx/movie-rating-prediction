"""MovieLens 25M大型电影库服务回归测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from movie_rating.web import ml25m_service


class MovieLens25MServiceTests(unittest.TestCase):
    def test_official_source_and_checksum_are_fixed(self) -> None:
        self.assertEqual(
            ml25m_service.DATASET_URL,
            "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        )
        self.assertEqual(ml25m_service.OFFICIAL_MD5, "6b51fb2759a8657d3bfcbfc42b592ada")

    def test_genres_are_translated_to_chinese(self) -> None:
        self.assertEqual(
            ml25m_service.translate_genres("Action|Adventure|Comedy"),
            "动作、冒险、喜剧",
        )
        self.assertEqual(ml25m_service.translate_genres("(no genres listed)"), "未分类")

    def test_cache_validation_requires_exact_official_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ml25m_service.runtime_paths(root)
            paths["catalog"].parent.mkdir(parents=True)
            paths["catalog"].write_text("movie\n1\n", encoding="utf-8")
            paths["info"].write_text(json.dumps({
                "movie_count": 62_423,
                "rating_count": 25_000_095,
                "archive_md5": ml25m_service.OFFICIAL_MD5,
            }), encoding="utf-8")
            self.assertTrue(ml25m_service.ml25m_catalog_ready(root))
            payload = json.loads(paths["info"].read_text(encoding="utf-8"))
            payload["rating_count"] -= 1
            paths["info"].write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(ml25m_service.ml25m_catalog_ready(root))

    def test_recommendations_refresh_without_overlap(self) -> None:
        sample = pd.DataFrame({
            "item_key": [f"ml25m:{value}" for value in range(1, 7)],
            "ml25m_movie_id": range(1, 7),
            "title_zh": [f"电影{value}" for value in range(1, 7)],
            "genres": ["Adventure|Comedy"] * 6,
            "rating_count": [200, 180, 160, 140, 120, 100],
            "rating_mean": [4.5, 4.4, 4.3, 4.2, 4.1, 4.0],
        })
        profile = pd.DataFrame({"genre": ["Adventure"], "affinity": [0.2], "support": [10]})
        info = {"global_mean": 3.5}
        with tempfile.TemporaryDirectory() as directory:
            info_path = ml25m_service.runtime_paths(directory)["info"]
            info_path.parent.mkdir(parents=True)
            info_path.write_text(json.dumps(info), encoding="utf-8")
            with (
                patch.object(ml25m_service, "load_ml25m_catalog", return_value=sample),
                patch.object(ml25m_service, "load_train_ratings", return_value=pd.DataFrame()),
                patch.object(
                    ml25m_service, "load_raw_tables",
                    return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
                ),
                patch.object(ml25m_service, "compute_user_genre_profile", return_value=profile),
            ):
                first = ml25m_service.recommend_ml25m_movies(1, 2, refresh_page=0, root=directory)
                second = ml25m_service.recommend_ml25m_movies(1, 2, refresh_page=1, root=directory)
        self.assertEqual(len(first), 2)
        self.assertTrue(set(first["item_key"]).isdisjoint(set(second["item_key"])))
        self.assertTrue(first["category"].eq("冒险、喜剧").all())


if __name__ == "__main__":
    unittest.main()
