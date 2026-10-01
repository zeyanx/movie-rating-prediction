"""初始化第6阶段本地SQLite反馈数据库。"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.web.database import default_database_path, initialize_database


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化MovieLens网页反馈数据库")
    parser.add_argument("--path", type=Path, help="可选数据库路径，测试时可指定临时文件")
    args = parser.parse_args()
    path = initialize_database(args.path)
    with sqlite3.connect(path) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_interactions'"
        ).fetchone()
    if table is None:
        raise RuntimeError("user_interactions表创建失败")
    print(f"SQLite反馈数据库已就绪：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
