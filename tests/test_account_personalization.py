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
from movie_rating.web.auth_service import _auth_message


class AccountPersonalizationTests(unittest.TestCase):
    def test_proxy_user_is_deterministic_and_valid(self) -> None:
        first = choose_proxy_user_id(["冒险", "喜剧"])
        second = choose_proxy_user_id(["冒险", "喜剧"])
        self.assertEqual(first, second)
        self.assertTrue(1 <= first <= 943)

    def test_preferences_are_required(self) -> None:
        with self.assertRaises(ValueError):
            choose_proxy_user_id([])

    def test_duplicate_signup_has_actionable_message(self) -> None:
        message = _auth_message(
            RuntimeError('duplicate key value violates unique constraint "users_email_partial_key"')
        )
        self.assertIn("确认邮件", message)
        self.assertIn("不要重复提交", message)

    def test_email_rate_limit_has_actionable_message(self) -> None:
        message = _auth_message(
            RuntimeError("over_email_send_rate_limit: can only request this after 39 seconds")
        )
        self.assertIn("60秒", message)
        self.assertIn("垃圾邮件", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
