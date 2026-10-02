"""复制最小部署包，在无本地数据/数据库的目录中模拟Streamlit Cloud。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "reports" / "stage7" / "clean_deployment_result.json"

COPY_DIRS = ["app_pages", "src", ".streamlit", "configs", "data/catalog"]
COPY_FILES = ["app.py", "requirements.txt"]
DEPLOY_MODELS = [
    "models/global_mean_baseline.joblib", "models/stage3/random_forest.joblib",
    "models/stage3/xgboost.joblib", "models/stage4/final_preprocessor.joblib",
    "models/stage4/mlp_final.pt", "models/stage4/model_metadata.json",
    "models/chinese/random_forest.joblib", "models/chinese/xgboost.joblib",
    "models/chinese/mlp.pt", "models/chinese/model_metadata.json",
]
DEPLOY_REPORTS = [
    "reports/stage5/metrics_comparison.csv", "reports/stage5/efficiency_benchmark.csv",
    "reports/stage5/paired_bootstrap_summary.csv", "reports/stage5/ensemble_group_importance.csv",
    "reports/stage5/mlp_permutation_importance.csv", "reports/stage5/segment_metrics.csv",
    "reports/stage5/residual_summary.csv", "reports/stage5/actual_rating_diagnostics.csv",
    "reports/stage5/model_recommendation.json",
    "reports/chinese_training/dataset_summary.json", "reports/chinese_training/metrics.csv",
    "reports/chinese_training/mlp_training_history.csv", "reports/chinese_training/summary.md",
]


def copy_file(relative: str, destination: Path) -> None:
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)


def build_bundle(destination: Path) -> None:
    for relative in COPY_DIRS:
        shutil.copytree(ROOT / relative, destination / relative, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in COPY_FILES + DEPLOY_MODELS + DEPLOY_REPORTS:
        copy_file(relative, destination)
    copy_file("data/processed/split_manifest.json", destination)
    copy_file("scripts/verify_clean_bundle.py", destination)
    copy_file("scripts/smoke_test_streamlit.py", destination)
    copy_file("tests/test_stage6_app.py", destination)


def run(arguments: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(arguments, cwd=cwd, check=False, text=True, encoding="utf-8",
                          errors="replace", capture_output=True, timeout=timeout, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-venv", action="store_true", help="新建venv并按requirements安装依赖")
    parser.add_argument(
        "--result-path",
        type=Path,
        default=DEFAULT_RESULT,
        help="验证结果JSON路径；相对路径按项目根目录解析",
    )
    args = parser.parse_args()
    result_path = args.result_path if args.result_path.is_absolute() else ROOT / args.result_path
    with tempfile.TemporaryDirectory(prefix="movie-stage7-") as directory:
        bundle = Path(directory) / "repo"
        bundle.mkdir()
        build_bundle(bundle)
        python = Path(sys.executable)
        install_output = f"复用当前Python {sys.version_info.major}.{sys.version_info.minor}环境"
        if args.with_venv:
            # 使用被Git忽略的专用环境，便于失败修复后复验而不反复下载大型CPU依赖。
            environment_dir = ROOT / ".venv-stage7"
            if not environment_dir.exists():
                created = run([str(python), "-m", "venv", str(environment_dir)], bundle, 180)
                if created.returncode:
                    raise RuntimeError(created.stderr)
            python = environment_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            installed = run([str(python), "-m", "pip", "install", "-r", "requirements.txt"], bundle, 1200)
            if installed.returncode:
                raise RuntimeError(f"干净环境依赖安装失败：\n{installed.stdout}\n{installed.stderr}")
            install_output = "新建venv并从requirements.txt安装成功"
        dependency_check = run([str(python), "-m", "pip", "check"], bundle, 120)
        if dependency_check.returncode:
            raise RuntimeError(f"pip check失败：\n{dependency_check.stdout}\n{dependency_check.stderr}")
        verified = run([str(python), "scripts/verify_clean_bundle.py"], bundle, 300)
        if verified.returncode:
            raise RuntimeError(f"干净部署失败：\n{verified.stdout}\n{verified.stderr}")
        payload = {
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "python": str(python),
            "environment_test": install_output,
            "source_data_in_bundle_before_start": False,
            "database_in_bundle_before_start": False,
            "result": "passed",
            "output_tail": verified.stdout[-4000:],
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(verified.stdout)
        print(f"干净部署结果：{result_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"干净部署验证失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
