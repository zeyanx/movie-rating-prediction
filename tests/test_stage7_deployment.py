"""第7阶段部署边界与Git合规测试。"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git_lines(*arguments: str) -> set[str]:
    result = subprocess.run(["git", *arguments], cwd=ROOT, text=True, encoding="utf-8",
                            errors="replace", capture_output=True, check=True)
    return set(result.stdout.splitlines())


class DeploymentComplianceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "configs/stage7_deployment.json").read_text(encoding="utf-8"))
        cls.tracked = git_lines("ls-files")

    def test_single_cloud_dependency_file(self) -> None:
        self.assertTrue((ROOT / "requirements.txt").is_file())
        for forbidden in ["environment.yml", "Pipfile", "uv.lock", "pyproject.toml", "packages.txt"]:
            self.assertFalse((ROOT / forbidden).exists(), forbidden)
        self.assertTrue((ROOT / "conda/environment.yml").is_file())

    def test_required_assets_are_tracked_and_under_limit(self) -> None:
        for relative in (
            self.config["required_models"]
            + self.config["required_reports"]
            + self.config["required_catalogs"]
        ):
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertIn(relative, self.tracked)
            self.assertLess(path.stat().st_size, 100 * 1024 * 1024, relative)
        for forbidden in ["models/stage3/best_ensemble.joblib", "models/stage3/adaboost.joblib",
                          "models/stage4/mlp_validation_best.pt", "models/stage4/validation_preprocessor.joblib"]:
            self.assertNotIn(forbidden, self.tracked)

    def test_sensitive_runtime_files_are_not_tracked(self) -> None:
        forbidden_prefixes = ["data/external/", "data/app/"]
        forbidden_exact = set(self.config["runtime_generated"]) | {".streamlit/secrets.toml"}
        for name in self.tracked:
            self.assertFalse(any(name.startswith(prefix) for prefix in forbidden_prefixes), name)
            self.assertNotIn(name, forbidden_exact)
            self.assertFalse(name.endswith("test_predictions.csv"), name)
            self.assertNotEqual(name, "reports/stage5/unified_test_predictions.csv")
            self.assertFalse(name.endswith(".html") and name.startswith("reports/"), name)

    def test_lfs_and_source_boundaries(self) -> None:
        lfs = git_lines("lfs", "ls-files", "--name-only")
        self.assertIn("models/stage3/random_forest.joblib", lfs)
        source = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for folder in [ROOT / "app_pages", ROOT / "src/movie_rating/web"]
            for path in folder.rglob("*.py")
        )
        self.assertNotIn("C:\\Users\\", source)
        for token in ["optimizer.step", ".backward("]:
            self.assertNotIn(token, source)
        online_source = "\n".join(
            (ROOT / relative).read_text(encoding="utf-8")
            for relative in [
                "src/movie_rating/web/data_service.py",
                "src/movie_rating/web/recommendation.py",
            ]
        )
        self.assertNotIn("test_ratings.csv", online_source)

    def test_manifest_and_five_pages(self) -> None:
        manifest = json.loads((ROOT / "reports/stage7/deployment_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(all(item["purpose"] for item in manifest["files"]))
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        for page in [
            "home.py", "movie_detail.py", "chinese_catalog.py",
            "recommendations.py", "my_movies.py", "model_lab.py",
        ]:
            self.assertIn(page, app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
