"""公开账户冷启动画像和安全配置测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.web.account_personalization import choose_proxy_user_id


class AccountPersonalizationTests(unittest.TestCase):
    def test_proxy_user_is_deterministic_and_valid(self) -> None:
        first = choose_proxy_user_id(["冒险", "喜剧"])
        second = choose_proxy_user_id(["冒险", "喜剧"])
        self.assertEqual(first, second)
        self.assertTrue(1 <= first <= 943)

    def test_preferences_are_required(self) -> None:
        with self.assertRaises(ValueError):
            choose_proxy_user_id([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
