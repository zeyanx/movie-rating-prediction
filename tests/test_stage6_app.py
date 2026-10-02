"""使用Streamlit官方AppTest检查五个页面可运行。"""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class Stage6AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = AppTest.from_file(ROOT / "app.py", default_timeout=60).run(timeout=60)

    def _assert_page(self, page_path: str, expected_title: str) -> None:
        self.app.switch_page(page_path).run(timeout=60)
        self.assertEqual(len(self.app.exception), 0, [item.value for item in self.app.exception])
        titles = [item.value for item in self.app.title]
        self.assertIn(expected_title, titles)
        self.assertEqual(self.app.session_state["selected_user_id"], 1)
        self.assertEqual(self.app.session_state["profile_key"], "movielens:1")

    def test_home(self) -> None:
        self.assertEqual(len(self.app.exception), 0, [item.value for item in self.app.exception])
        self.assertIn("MovieLens 智能评分与推荐系统", [item.value for item in self.app.title])

    def test_movie_detail(self) -> None:
        self._assert_page("app_pages/movie_detail.py", "电影详情")

    def test_chinese_catalog(self) -> None:
        self._assert_page("app_pages/chinese_catalog.py", "中文电影库")

    def test_recommendations(self) -> None:
        self._assert_page("app_pages/recommendations.py", "个性化推荐")

    def test_my_movies(self) -> None:
        self._assert_page("app_pages/my_movies.py", "我的观影")

    def test_model_lab(self) -> None:
        self._assert_page("app_pages/model_lab.py", "模型实验室")


if __name__ == "__main__":
    unittest.main()
