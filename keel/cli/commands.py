"""`keel` command-line entry points: new / run / build.

This module never imports the rest of keel at the top level so the CLI
starts fast and never depends on optional GL / physics deps.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_RELOAD_DEBOUNCE_SECONDS: float = 0.05

BUILD_NOT_IMPLEMENTED_MSG = (
    "[keel] build: not yet implemented.\nRun your project with: python main.py"
)


def cmd_new(project_name: str) -> int:
    """Scaffold a new project directory."""
    from .templates import MAIN_PY_TEMPLATE, PYPROJECT_TEMPLATE, README_TEMPLATE

    root = Path(project_name)
    if root.exists():
        print(f"[keel] error: {project_name!r} already exists", file=sys.stderr)
        return 1

    root.mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "scenes").mkdir()
    (root / "assets" / ".gitkeep").touch()
    (root / "scenes" / ".gitkeep").touch()
    (root / "main.py").write_text(
        MAIN_PY_TEMPLATE.format(project_name=project_name), encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        PYPROJECT_TEMPLATE.format(project_name=project_name), encoding="utf-8"
    )
    (root / "README.md").write_text(
        README_TEMPLATE.format(project_name=project_name), encoding="utf-8"
    )
    print(f"[keel] created project at {project_name}/")
    return 0


def _spawn(entry: str) -> subprocess.Popen:
    """Start the user's entry script in a fresh Python subprocess."""
    return subprocess.Popen([sys.executable, entry])


def _terminate(proc: subprocess.Popen | None) -> None:
    """Send SIGTERM, wait up to 3s, then SIGKILL — never leave zombies."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def _reload_loop(
    entry: str,
    reload_q: "queue.Queue[str]",
    spawn=_spawn,
    terminate=_terminate,
    *,
    poll_timeout: float = 0.2,
) -> None:
    """Block-then-restart loop. Extracted from cmd_run so tests can drive it directly."""
    proc: subprocess.Popen | None = spawn(entry)
    try:
        while True:
            try:
                reload_q.get(timeout=poll_timeout)
            except queue.Empty:
                if proc is not None and proc.poll() is not None:
                    time.sleep(0.1)
                continue

            # Coalesce a burst of save events into one reload.
            time.sleep(_RELOAD_DEBOUNCE_SECONDS)
            while True:
                try:
                    reload_q.get_nowait()
                except queue.Empty:
                    break

            print("[keel] reloading...")
            terminate(proc)
            proc = spawn(entry)
    except KeyboardInterrupt:
        print("\n[keel] shutting down")
    finally:
        terminate(proc)


def cmd_run(entry: str = "main.py", *, watch_dir: str = ".") -> int:
    """Run `entry` in a subprocess; restart on .py file change; Ctrl+C to exit."""
    if not Path(entry).exists():
        print(f"[keel] error: entry {entry!r} not found", file=sys.stderr)
        return 1

    print(f"[keel] running {entry}")

    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    reload_q: "queue.Queue[str]" = queue.Queue()

    class _Handler(FileSystemEventHandler):
        def _maybe_enqueue(self, event: FileSystemEvent) -> None:
            if event.is_directory:
                return
            path = getattr(event, "dest_path", None) or getattr(event, "src_path", "")
            if path.endswith(".py"):
                reload_q.put(path)

        def on_modified(self, event: FileSystemEvent) -> None:
            self._maybe_enqueue(event)

        def on_created(self, event: FileSystemEvent) -> None:
            self._maybe_enqueue(event)

        def on_moved(self, event: FileSystemEvent) -> None:
            self._maybe_enqueue(event)

    observer = Observer()
    observer.schedule(_Handler(), os.path.abspath(watch_dir), recursive=True)
    observer.start()
    try:
        _reload_loop(entry, reload_q)
    finally:
        try:
            observer.stop()
            observer.join(timeout=2.0)
        except Exception:
            pass
    return 0


def cmd_build() -> int:
    """Build stub for v1 — print a hint and exit 0."""
    print(BUILD_NOT_IMPLEMENTED_MSG)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keel",
        description="Keel developer CLI: scaffold, run, and (later) build projects.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    new_p = sub.add_parser("new", help="Scaffold a new project directory")
    new_p.add_argument("project_name", help="Name of the new project (also the directory name)")

    run_p = sub.add_parser("run", help="Run a project with .py hot reload")
    run_p.add_argument("entry", nargs="?", default="main.py", help="Entry script (default: main.py)")

    sub.add_parser("build", help="Package project for distribution (stub for v1)")

    return parser


def main(argv: list[str] | None = None) -> int:
    """argparse dispatch for `keel <subcommand>`. Returns the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "new":
        return cmd_new(args.project_name)
    if args.command == "run":
        return cmd_run(args.entry)
    if args.command == "build":
        return cmd_build()
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
