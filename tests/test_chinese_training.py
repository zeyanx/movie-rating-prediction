"""中国电影真实评分数据库和模型推理回归测试。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.web.model_service import predict_chinese_ratings


class ChineseTrainingTests(unittest.TestCase):
    def test_catalog_and_summary(self) -> None:
        catalog = pd.read_csv(ROOT / "data" / "catalog" / "chinese_rated_movies.csv")
        summary = json.loads(
            (ROOT / "reports" / "chinese_training" / "dataset_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(catalog), 300)
        self.assertTrue(catalog["catalog_id"].is_unique)
        self.assertIn("genres", catalog.columns)
        self.assertTrue(catalog["genres"].fillna("").str.len().gt(0).all())
        self.assertEqual(summary["ratings"], 55_483)
        self.assertEqual(summary["users"], 5_973)

    def test_saved_models_predict_in_range(self) -> None:
        result = predict_chinese_ratings(1, [1, 100, 300])
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertTrue(((result >= 0.5) & (result <= 5.0)).all().all())

    def test_xgboost_beats_baseline(self) -> None:
        metrics = pd.read_csv(ROOT / "reports" / "chinese_training" / "metrics.csv")
        values = metrics.set_index("model")["rmse"]
        self.assertLess(values["xgboost"], values["global_mean"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
