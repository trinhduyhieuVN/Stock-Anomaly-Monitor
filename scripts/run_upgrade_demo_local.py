"""Bounded localhost-only synthetic demo; never starts a provider or another writer."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen
import uuid

from stock_monitor.upgrade.ui_service import load_ui_bundle

ROOT = Path(__file__).resolve().parents[1]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--minutes", type=float, default=45)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    if not 0 < args.minutes <= 60:
        parser.error("Duration must be greater than zero and at most 60 minutes")
    return args


def require_free_port(port):
    with socket.socket() as sock:
        # Exclusive bind on Windows: do not reuse, kill or attach to another app.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Cổng {port} chưa dùng được; chọn --port khác, không dừng phiên khác.") from exc


def launch_command(port):
    return [sys.executable, "-X", "utf8", "-m", "streamlit", "run", "app_upgrade.py",
            "--server.address=127.0.0.1", f"--server.port={port}",
            "--server.headless=true", "--browser.gatherUsageStats=false"]


def stop_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()  # Exact child created by this launcher, never a PID lookup.
            process.wait(timeout=5)


def main(argv=None):
    args = arguments(argv)
    bundle = load_ui_bundle(ROOT)
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Dùng Python của venv project theo hướng dẫn, không sửa môi trường toàn máy.")
    if args.check_only:
        print(json.dumps({"status": "PASS_PREFLIGHT", "manifest_sha256": bundle["key"],
            "sessions": len(bundle["sessions"]), "mode": "SYNTHETIC_NOT_LIVE",
            "provider_calls": 0, "server_started": False, "maximum_minutes": 60}))
        return 0
    require_free_port(args.port)
    name = "local_" + uuid.uuid4().hex[:12]
    directory = ROOT / "reports/demo_completion" / name
    directory.mkdir(parents=True, exist_ok=False)
    log_path = directory / "streamlit.log"
    command = launch_command(args.port)
    env = dict(os.environ)
    env["STOCK_MONITOR_UPGRADE_CONFIG"] = "configs/upgrade_ui.json"
    env.setdefault("STOCK_MONITOR_UPGRADE_JOURNAL", ".local/upgrade_ui/journal.sqlite")
    env["PYTHONUTF8"] = "1"
    process, health, errors, reason = None, None, [], "duration_elapsed"
    url = f"http://127.0.0.1:{args.port}"
    try:
        with log_path.open("x", encoding="utf-8") as log:
            log.write(json.dumps({"command": subprocess.list2cmdline(command)}) + "\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            deadline = monotonic() + 30
            while monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Streamlit đã dừng: exit {process.returncode}; xem {log_path}")
                try:
                    with urlopen(url + "/_stcore/health", timeout=2) as response:
                        health = response.status
                    if health == 200:
                        break
                except (OSError, URLError):
                    sleep(.2)
            if health != 200:
                raise RuntimeError("Localhost chưa sẵn sàng trong 30 giây")
            print(json.dumps({"status": "READY", "url": url, "mode": "SYNTHETIC_NOT_LIVE",
                "minutes": args.minutes, "server_pid": process.pid, "provider_calls": 0,
                "session_directory": directory.relative_to(ROOT).as_posix(),
                "guide": "docs/upgrade/demo_local_hoan_thien.md"}), flush=True)
            deadline = monotonic() + args.minutes * 60
            while monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Streamlit kết thúc sớm: exit {process.returncode}")
                sleep(min(.5, max(0, deadline - monotonic())))
    except KeyboardInterrupt:
        reason = "operator_interrupt"
    except (OSError, RuntimeError) as exc:
        errors.append(str(exc)); reason = "failed"
    finally:
        if process is not None:
            stop_owned(process)
        report = {"schema_version": 2, "criteria_met": False, "status": "STOPPED" if not errors else "FAILED",
            "created_at": datetime.now(timezone.utc).isoformat(), "url": url, "stop_reason": reason,
            "own_child_stopped": process is None or process.poll() is not None, "health_status": health,
            "server_pid": process.pid if process else None, "server_exit_code": process.returncode if process else None,
            "command": subprocess.list2cmdline(command), "log": log_path.relative_to(ROOT).as_posix(),
            "provider_calls": 0, "published": False, "errors": errors}
        with (directory / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
