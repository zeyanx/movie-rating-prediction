"""第7阶段独立验收：发布边界、部署资产、测试证据与论文答辩材料。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def checked(arguments: list[str], label: str, timeout: int = 300) -> None:
    result = subprocess.run(arguments, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
                            capture_output=True, check=False, timeout=timeout)
    if result.returncode:
        raise AssertionError(f"{label}失败：\n{result.stdout}\n{result.stderr}")
    print(f"{label}：通过")


def main() -> int:
    print("=== 第7阶段独立验收 ===")
    require(sys.version_info[:2] == (3, 10), "必须使用Python 3.10")
    config = json.loads((ROOT / "configs/stage7_deployment.json").read_text(encoding="utf-8"))
    require(config["entrypoint"] == "app.py", "云端入口必须为app.py")
    for relative in config["required_models"] + config["required_reports"]:
        require((ROOT / relative).is_file(), f"部署必需文件缺失：{relative}")
    for relative in [
        "docs/thesis_draft.md", "docs/thesis_figures_and_tables.md", "docs/references.md",
        "docs/defense_slides_outline.md", "docs/defense_script.md", "docs/defense_q_and_a.md",
        "docs/live_demo_runbook.md", "docs/final_project_checklist.md",
        "defense/presentation_outline.md", "defense/speech_script.md", "defense/demo_script.md",
        "defense/qa_preparation.md", "defense/final_checklist.md",
        "reports/stage7/deployment_manifest.json", "reports/stage7/clean_deployment_result.json",
        "reports/stage7/python311_compatibility.json", "reports/stage7/online_validation.json",
        "reports/stage7/deployment_summary.md",
    ]:
        require((ROOT / relative).is_file() and (ROOT / relative).stat().st_size > 0, f"缺少材料：{relative}")
    for name in ["01_home.png", "02_movie_detail.png", "03_recommendations.png",
                 "04_my_movies.png", "05_model_lab.png"]:
        screenshot = ROOT / "reports/stage7/screenshots" / name
        require(screenshot.is_file() and screenshot.stat().st_size > 20_000, f"云端截图无效：{name}")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for token in ["data/raw/*.csv", "data/processed/*.csv", "data/app/*.db", ".streamlit/secrets.toml"]:
        require(token in ignore, f"忽略规则缺少：{token}")
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    require("ensure_runtime_assets_cached" in app_source, "入口未配置官方数据首次启动")
    checked([sys.executable, "-m", "pip", "check"], "依赖一致性")
    checked([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_stage7*.py", "-v"], "第7阶段单元测试")
    # 公开仓库按许可与隐私边界排除了逐行测试预测文件，因此干净克隆不能
    # 调用依赖这些历史产物的verify_stage2/verify_stage5链。改用覆盖五页面、
    # 推理、推荐、数据库和数据服务的第6阶段部署安全回归测试。
    checked(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_stage6*.py", "-v"],
        "第6阶段部署安全回归",
    )
    result = json.loads((ROOT / "reports/stage7/clean_deployment_result.json").read_text(encoding="utf-8"))
    require(result["result"] == "passed", "干净部署证据不是passed")
    py311 = json.loads((ROOT / "reports/stage7/python311_compatibility.json").read_text(encoding="utf-8"))
    require(py311["result"] == "passed", "Python 3.11云端兼容验证未通过")
    online = json.loads((ROOT / "reports/stage7/online_validation.json").read_text(encoding="utf-8"))
    require(online["result"] == "passed" and len(online["pages"]) == 5, "线上五页面验收未通过")
    require(config["repository_url"].startswith("https://github.com/"), "GitHub仓库地址缺失")
    require(config["streamlit_cloud_url"].startswith("https://"), "Streamlit在线地址缺失")
    print("GitHub仓库：" + config["repository_url"])
    print("Streamlit应用：" + config["streamlit_cloud_url"])
    print("第7阶段验证通过")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"第7阶段验证失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
