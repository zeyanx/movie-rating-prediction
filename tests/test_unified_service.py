"""统一电影库、评分预测和混合推荐回归测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from movie_rating.web.unified_service import (
    predict_unified_item,
    recommend_unified_movies,
    search_unified_catalog,
    unified_catalog,
    unified_genres,
)


class UnifiedServiceTests(unittest.TestCase):
    def test_catalog_contains_both_regions(self) -> None:
        catalog = unified_catalog()
        counts = catalog.groupby("model_space").size().to_dict()
        self.assertEqual(counts, {"china": 300, "international": 1324})
        self.assertEqual(len(catalog), 1624)
        self.assertFalse(catalog["title_zh"].eq("").any())
        self.assertFalse(catalog.loc[catalog["model_space"] == "china", "category"].eq("未分类").any())

    def test_chinese_search_for_both_sources(self) -> None:
        china = search_unified_catalog("流浪地球")
        international = search_unified_catalog("玩具总动员")
        self.assertEqual(china.iloc[0]["model_space"], "china")
        self.assertEqual(international.iloc[0]["model_space"], "international")

    def test_predictions_are_finite(self) -> None:
        for item_key in ["china:300", "international:1"]:
            result = predict_unified_item(1, item_key)
            self.assertTrue(np.isfinite(result["score"]))
            self.assertTrue(0.5 <= result["score"] <= 5.0)
            self.assertEqual(len(result["reasons"]), 3)

    def test_genre_filter_replaces_region_filter(self) -> None:
        genres = unified_genres()
        self.assertIn("冒险", genres)
        self.assertIn("喜剧", genres)
        for genre in ["冒险", "喜剧"]:
            result = search_unified_catalog(genre=genre)
            self.assertFalse(result.empty)
            self.assertTrue(result["category"].str.split("、").map(lambda x: genre in x).all())
        comedy = search_unified_catalog(genre="喜剧")
        self.assertEqual(set(comedy["model_space"]), {"china", "international"})

    def test_combined_recommendations_include_both_sources(self) -> None:
        result = recommend_unified_movies(1, pd.DataFrame(), top_n=6)
        self.assertEqual(len(result), 6)
        self.assertEqual(set(result["model_space"]), {"china", "international"})
        self.assertTrue(result["final_score"].between(0.5, 5.0).all())
        self.assertTrue(result["category"].str.len().gt(0).all())
        self.assertTrue(result["reasons"].map(lambda values: len(values) >= 2).all())
        reason_text = " ".join(reason for values in result["reasons"] for reason in values)
        for english_label in ["Drama", "Romance", "War", "MLP"]:
            self.assertNotIn(english_label, reason_text)

    def test_genre_filtered_recommendations(self) -> None:
        result = recommend_unified_movies(1, pd.DataFrame(), top_n=6, genre="冒险")
        self.assertEqual(len(result), 6)
        self.assertTrue(result["category"].str.split("、").map(lambda x: "冒险" in x).all())


if __name__ == "__main__":
    unittest.main()
