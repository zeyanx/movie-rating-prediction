"""启动临时Streamlit服务并检查健康端点，完成后清理进程。"""

from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    port = find_free_port()
    log_path = Path(tempfile.gettempdir()) / f"movie_streamlit_smoke_{port}.log"
    command = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.headless", "true", "--server.address", "127.0.0.1",
        "--server.port", str(port), "--browser.gatherUsageStats", "false",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 30.0
        url = f"http://127.0.0.1:{port}/_stcore/health"
        last_error = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(url, timeout=2.0) as response:
                    body = response.read().decode("utf-8", errors="replace").strip()
                    if response.status == 200 and "ok" in body.lower():
                        print(f"Streamlit健康检查通过：{url}")
                        return 0
            except Exception as exc:  # 仅记录轮询失败，最终仍会返回非零状态。
                last_error = str(exc)
            time.sleep(0.25)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        raise RuntimeError(f"Streamlit未在30秒内健康启动：{last_error}\n{log_text[-4000:]}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        # Windows上的Streamlit子进程可能短暂持有日志句柄；清理失败不应推翻已通过的健康检查。
        for _ in range(20):
            if not log_path.exists():
                break
            try:
                log_path.unlink()
                break
            except PermissionError:
                time.sleep(0.1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"Streamlit冒烟测试失败：{error}", file=sys.stderr)
        raise SystemExit(1)
