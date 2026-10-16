"""Verify the compact release without reading a research archive or using a network."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".dylib", ".so", ".dll", ".o", ".aux", ".out", ".log", ".toc"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe manifest path: {name}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes root: {name}")
    return path


def ignored(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (bool(set(relative.parts) & IGNORED_PARTS)
            or relative.parts[0] in {"build", "archive"}
            or (relative.parts[0] == "figures" and path.suffix == ".png")
            or path.name == ".DS_Store" or path.name.endswith(".synctex.gz")
            or path.suffix in IGNORED_SUFFIXES)


def verify() -> dict:
    manifest_path = ROOT / "provenance/release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    failures = []
    expected = manifest["files"]
    for name, item in expected.items():
        path = safe_path(ROOT, name)
        if path.is_symlink() or not path.is_file():
            failures.append(f"Missing or symlinked release file: {name}")
        elif path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            failures.append(f"Release file differs from manifest: {name}")
        if item["bytes"] >= 100 * 1024 * 1024:
            failures.append(f"File reaches regular Git size limit: {name}")
    candidates = [p for p in ROOT.rglob("*") if not ignored(p)]
    failures += [f"Symlink in release tree: {p.relative_to(ROOT)}" for p in candidates if p.is_symlink()]
    actual = {str(p.relative_to(ROOT)) for p in candidates if p.is_file()}
    extra = actual - set(expected) - {str(manifest_path.relative_to(ROOT))}
    failures += [f"Unexpected unmanifested release file: {p}" for p in sorted(extra)]
    links = 0
    for name in expected:
        if not name.endswith(".md"):
            continue
        path = ROOT / name
        for match in LINK.finditer(path.read_text()):
            target = match.group(1).strip().strip("<>")
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            destination = (path.parent / target).resolve()
            if Path(target).is_absolute() or not destination.is_relative_to(ROOT):
                failures.append(f"Nonportable documentation link in {name}")
            elif not destination.exists():
                failures.append(f"Missing documentation link in {name}: {target}")
            links += 1
    result = {"status": "failed" if failures else "passed", "files": len(expected),
              "bytes": sum(item["bytes"] for item in expected.values()),
              "relative_documentation_links": links,
              "scope": "Release inventory, checksums, size limits and local documentation links.",
              "failures": failures}
    if failures:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
