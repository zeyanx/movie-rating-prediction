"""生成部署文件清单、大小与SHA-256，便于答辩和云端复核。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage7_deployment.json"
OUTPUT = ROOT / "reports" / "stage7" / "deployment_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    return set(result.stdout.splitlines()) if result.returncode == 0 else set()


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    required = ["app.py", "requirements.txt", "conda/environment.yml"]
    required += config["required_models"] + config["required_reports"]
    tracked = git_files()
    items = []
    for relative in required:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"部署文件缺失：{relative}")
        items.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "should_be_tracked": True,
            "git_tracked": relative in tracked,
            "git_lfs_required": relative in config["git_lfs_paths"],
        })
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entrypoint": config["entrypoint"],
        "official_dataset_url": config["official_dataset_url"],
        "runtime_generated": config["runtime_generated"],
        "files": items,
        "total_bytes": sum(item["bytes"] for item in items),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"部署清单已生成：{OUTPUT.relative_to(ROOT)}，共{len(items)}个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
