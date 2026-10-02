"""第7阶段运行时数据、校验与安全解压测试。"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from movie_rating.web.runtime_assets import (
    GENRE_NAMES, OFFICIAL_MD5, calculate_md5, ensure_runtime_assets,
    safe_extract_zip, verify_archive_md5,
)


class RuntimeAssetsTest(unittest.TestCase):
    def test_official_archive_builds_exact_dataset_and_split(self) -> None:
        source = ROOT / "data" / "external" / "ml-100k.zip"
        self.assertTrue(source.is_file(), "本地集成测试需要第1阶段官方ZIP")
        self.assertEqual(calculate_md5(source), OFFICIAL_MD5)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "data" / "processed").mkdir(parents=True)
            shutil.copy2(ROOT / "data" / "processed" / "split_manifest.json", target / "data" / "processed" / "split_manifest.json")
            result = ensure_runtime_assets(target, archive_path=source)
            self.assertEqual(result["verification"]["ratings"], 100_000)
            self.assertEqual(result["verification"]["users"], 943)
            self.assertEqual(result["verification"]["movies"], 1_682)
            self.assertEqual(result["verification"]["train_sha256"], "d2c8ac22231cc761a3a522e01e1a955c9d7cca847c0deb2ef1e8c9ed863c9053")
            # 第二次执行必须幂等，不重新转换有效文件。
            second = ensure_runtime_assets(target, archive_path=source)
            self.assertFalse(second["raw"]["downloaded"])
            self.assertFalse(second["split"]["generated"])
            movies = __import__("pandas").read_csv(target / "data/raw/movies.csv")
            genre_union = set().union(*(str(value).split("|") for value in movies["genres"]))
            self.assertEqual(genre_union, set(GENRE_NAMES))
            with ThreadPoolExecutor(max_workers=2) as pool:
                repeated = list(pool.map(lambda _: ensure_runtime_assets(target, archive_path=source), range(2)))
            self.assertTrue(all(item["verification"]["ratings"] == 100_000 for item in repeated))

    def test_wrong_md5_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.zip"
            bad.write_bytes(b"not the official archive")
            with self.assertRaises(ValueError):
                verify_archive_md5(bad)

    def test_zip_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "禁止写出目录")
            with self.assertRaises(ValueError):
                safe_extract_zip(archive, root / "extract")


if __name__ == "__main__":
    unittest.main(verbosity=2)
