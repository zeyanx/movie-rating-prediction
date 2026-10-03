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
        self.assertIn("智能电影评分与推荐系统", [item.value for item in self.app.title])

    def test_movie_detail(self) -> None:
        self._assert_page("app_pages/movie_detail.py", "评分预测")

    def test_chinese_catalog(self) -> None:
        self._assert_page("app_pages/chinese_catalog.py", "电影库")

    def test_recommendations(self) -> None:
        self._assert_page("app_pages/recommendations.py", "个性化推荐")
        first_titles = [item.value for item in self.app.subheader]
        refresh_button = next(
            button for button in self.app.button if button.label == "换一批推荐"
        )
        refresh_button.click().run(timeout=60)
        self.assertEqual(len(self.app.exception), 0, [item.value for item in self.app.exception])
        second_titles = [item.value for item in self.app.subheader]
        self.assertNotEqual(first_titles, second_titles)
        self.assertTrue(
            any("当前为第 2 批" in item.value for item in self.app.caption)
        )

    def test_my_movies(self) -> None:
        self._assert_page("app_pages/my_movies.py", "我的观影")

    def test_model_lab(self) -> None:
        self._assert_page("app_pages/model_lab.py", "模型分析")

    def test_account_settings_local_fallback(self) -> None:
        self._assert_page("app_pages/account.py", "账户设置")


if __name__ == "__main__":
    unittest.main()
