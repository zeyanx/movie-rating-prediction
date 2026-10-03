"""下载并生成MovieLens 25M大型电影目录。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.web.ml25m_service import ensure_ml25m_catalog, runtime_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="准备MovieLens 25M大型电影库")
    parser.add_argument("--archive", type=Path, help="使用手动下载的官方ml-25m.zip")
    parser.add_argument("--force", action="store_true", help="强制重新校验并生成")
    args = parser.parse_args()
    info = ensure_ml25m_catalog(ROOT, args.archive, args.force)
    paths = runtime_paths(ROOT)
    print(f"电影：{info['movie_count']:,} 部")
    print(f"评分：{info['rating_count']:,} 条")
    print(f"用户：{info['user_count']:,} 名")
    print(f"MD5：{info['archive_md5']}")
    print(f"目录：{paths['catalog']}")
    print("MovieLens 25M大型电影库准备完成")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"准备失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
