"""第6阶段服务层与推荐闭环测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.web.config import load_app_config
from movie_rating.web.data_service import (
    build_prediction_frame, get_reference_timestamp, load_raw_tables,
    load_train_ratings, search_movies,
)
from movie_rating.web.database import (
    delete_interaction, get_interaction, get_interactions, initialize_database,
    upsert_interaction,
)
from movie_rating.web.model_service import predict_with_mlp
from movie_rating.web.recommendation import build_candidate_frame, recommend_movies


class Stage6ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_app_config()
        cls.train = load_train_ratings()
        cls.ratings, cls.movies, cls.users = load_raw_tables()

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "feedback.db"
        initialize_database(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_config_and_data(self) -> None:
        self.assertAlmostEqual(sum(self.config["recommendation_weights"].values()), 1.0)
        self.assertEqual((len(self.ratings), len(self.users), len(self.movies)), (100_000, 943, 1_682))
        self.assertEqual(len(self.train), 80_000)

    def test_literal_case_insensitive_search(self) -> None:
        lower = search_movies("toy story", limit=10)
        upper = search_movies("TOY STORY", limit=10)
        self.assertEqual(lower["movie_id"].tolist(), upper["movie_id"].tolist())
        self.assertIn(1, lower["movie_id"].tolist())
        self.assertTrue(search_movies("[", limit=10).empty)  # 若被当正则会报错或产生错误结果。

    def test_reference_timestamp(self) -> None:
        user_rows = self.train[self.train["user_id"] == 1]
        self.assertEqual(get_reference_timestamp(1, self.train), int(user_rows["timestamp"].max()))
        self.assertEqual(get_reference_timestamp(9999, self.train), int(self.train["timestamp"].max()))

    def test_mlp_single_and_batch_predictions(self) -> None:
        frame = build_prediction_frame(1, [1, 2, 3])
        prediction = predict_with_mlp(frame)
        self.assertEqual(len(prediction), 3)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(((prediction >= 1) & (prediction <= 5)).all())
        self.assertEqual(len(predict_with_mlp(frame.iloc[:1])), 1)

    def test_database_crud_and_validation(self) -> None:
        initialize_database(self.db_path)  # 重复初始化必须安全。
        upsert_interaction("movielens:1", 1, 1, "want_to_watch", feedback="like", path=self.db_path)
        upsert_interaction("movielens:1", 1, 1, "watched", personal_rating=4.5, path=self.db_path)
        rows = get_interactions("movielens:1", self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(get_interaction("movielens:1", 1, self.db_path)["status"], "watched")
        with self.assertRaises(ValueError):
            upsert_interaction("movielens:1", 1, 1, "invalid", path=self.db_path)
        with self.assertRaises(ValueError):
            upsert_interaction("movielens:1", 1, 1, "watched", personal_rating=6, path=self.db_path)
        with self.assertRaises(ValueError):
            upsert_interaction("movielens:1", 1, 1, "watched", feedback="bad", path=self.db_path)
        self.assertTrue(delete_interaction("movielens:1", 1, self.db_path))
        self.assertIsNone(get_interaction("movielens:1", 1, self.db_path))

    def test_candidate_exclusions(self) -> None:
        from movie_rating.web.data_service import movie_statistics

        stats = movie_statistics()
        rated_movie = int(self.train[self.train["user_id"] == 1].iloc[0]["movie_id"])
        unobserved = int(stats[~stats["movie_id"].isin(
            self.train.loc[self.train["user_id"] == 1, "movie_id"]
        )].iloc[0]["movie_id"])
        local = pd.DataFrame([{
            "movie_id": unobserved, "status": "want_to_watch", "feedback": "not_interested"
        }])
        candidates, _ = build_candidate_frame(1, stats, self.train, local, minimum_popularity_count=0)
        ids = set(candidates["movie_id"].astype(int))
        self.assertNotIn(rated_movie, ids)
        self.assertNotIn(unobserved, ids)

    def test_recommendation_determinism_reasons_and_range(self) -> None:
        empty = pd.DataFrame()
        first, _ = recommend_movies(1, empty, top_n=5)
        second, _ = recommend_movies(1, empty, top_n=5)
        self.assertLessEqual(len(first), 5)
        self.assertTrue(first["movie_id"].is_unique)
        self.assertEqual(first["movie_id"].tolist(), second["movie_id"].tolist())
        self.assertTrue(first["final_score"].between(1, 5).all())
        self.assertTrue(first["mlp_prediction"].between(1, 5).all())
        self.assertTrue(first["reasons"].map(lambda values: len(values) >= 2 and all(values)).all())


if __name__ == "__main__":
    unittest.main()
