#!/usr/bin/env python3
"""One entry point for compact-release integrity, result reproduction and tests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args: str, cwd: Path = ROOT) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONOPTIMIZE", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment.setdefault("MPLCONFIGDIR", str(ROOT / "build/matplotlib"))
    (ROOT / "build").mkdir(exist_ok=True)
    subprocess.run([sys.executable, *args], cwd=cwd, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "results", "figures", "tests", "all"])
    args = parser.parse_args()
    command = args.command
    if command in {"check", "all"}:
        run("tools/release_check.py")
    if command in {"results", "all"}:
        run("reporting/verify_saved_results.py")
    if command in {"figures", "all"}:
        run("reporting/plot_price_results.py", "--output-dir", str(ROOT / "build/figures"))
        run("reporting/plot_spread_concentration.py", "--output-dir", str(ROOT / "build/figures"))
    if command in {"tests", "all"}:
        run("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", cwd=ROOT / "pipeline")
        run("-m", "unittest", "discover", "-s", "reports/tests", "-p", "test_*.py", cwd=ROOT / "pipeline")
        run("-m", "unittest", "discover", "-s", "tools", "-p", "test_*.py")
    if command == "all":
        run("tools/release_check.py")
    print(json.dumps({"status": "passed", "command": command,
                      "scope": "Saved-evidence reproduction and selected computational tests; no raw-feed model refit."}))


if __name__ == "__main__":
    main()
