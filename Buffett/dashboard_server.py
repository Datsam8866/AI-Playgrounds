# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import date
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
HOST = "127.0.0.1"
PORT = 8787
_UPDATE_LOCK = threading.Lock()


def run_command(args: list[str], cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.returncode, completed.stdout.strip()


def run_update() -> dict[str, object]:
    if not _UPDATE_LOCK.acquire(blocking=False):
        return {"ok": False, "message": "更新已在執行中，請稍候。", "steps": []}

    steps: list[dict[str, object]] = []
    try:
        commands = [
            ("更新持倉資料", [sys.executable, "-X", "utf8", "update_portfolio.py"], ROOT),
            ("重新產生 Dashboard", [sys.executable, "-X", "utf8", "generate_portfolio_dashboard.py"], ROOT),
            ("加入 dashboard 變更", ["git", "-C", str(REPO_ROOT), "add", "Buffett/portfolio_dashboard.html"], REPO_ROOT),
        ]
        for label, command, cwd in commands:
            code, output = run_command(command, cwd)
            steps.append({"label": label, "code": code, "output": output})
            if code != 0:
                return {"ok": False, "message": f"{label} 失敗。", "steps": steps}

        code, _ = run_command(["git", "-C", str(REPO_ROOT), "diff", "--cached", "--quiet"], REPO_ROOT)
        if code == 0:
            steps.append({"label": "建立 commit", "code": 0, "output": "沒有 dashboard 變更需要提交。"})
            return {"ok": True, "message": "資料已更新，本次沒有 dashboard 變更需要推送。", "steps": steps}

        commit_message = f"每日更新 Dashboard {date.today().isoformat()}"
        code, output = run_command(["git", "-C", str(REPO_ROOT), "commit", "-m", commit_message], REPO_ROOT)
        steps.append({"label": "建立 commit", "code": code, "output": output})
        if code != 0:
            return {"ok": False, "message": "commit 失敗。", "steps": steps}

        code, output = run_command(["git", "-C", str(REPO_ROOT), "push", "origin", "main"], REPO_ROOT)
        steps.append({"label": "推送 GitHub", "code": code, "output": output})
        if code != 0:
            return {"ok": False, "message": "push 失敗。", "steps": steps}

        return {"ok": True, "message": "更新、commit、push 已完成。", "steps": steps}
    finally:
        _UPDATE_LOCK.release()


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.send_json({"ok": True, "message": "Dashboard server is running."})
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/update":
            self.send_error(404)
            return
        if self.client_address[0] not in {"127.0.0.1", "::1"}:
            self.send_error(403)
            return
        self.send_json(run_update())

    def send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Dashboard server: http://{HOST}:{PORT}/portfolio_dashboard.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
