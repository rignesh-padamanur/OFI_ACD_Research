"""Adversarial checks for publication integrity and archive path handling."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import release_check


class ReleaseIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "provenance").mkdir()
        self.patch = patch.object(release_check, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def freeze(self):
        entries = {str(p.relative_to(self.root)): {"bytes": p.stat().st_size,
                   "sha256": release_check.digest(p)}
                   for p in self.root.rglob("*") if p.is_file()}
        (self.root / "provenance/release_manifest.json").write_text(json.dumps({"files": entries}))

    def test_unchanged_release_passes_and_mutation_fails(self):
        path = self.root / "values.csv"
        path.write_text("x\n1\n")
        self.freeze()
        self.assertEqual(release_check.verify()["status"], "passed")
        path.write_text("x\n2\n")
        with self.assertRaisesRegex(RuntimeError, "differs from manifest"):
            release_check.verify()

    def test_unmanifested_directory_symlink_fails(self):
        self.freeze()
        (self.root / "external_directory").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "Symlink in release tree"):
            release_check.verify()

    def test_ignored_virtual_environment_symlink_is_not_release_content(self):
        self.freeze()
        (self.root / ".venv").mkdir()
        (self.root / ".venv/python").symlink_to(self.root.parent, target_is_directory=True)
        self.assertEqual(release_check.verify()["status"], "passed")

    def test_archive_path_traversal_is_rejected(self):
        for path in ("../outside", "/absolute", "data/../../outside", "data\\outside"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                release_check.safe_path(self.root, path)

    def test_broken_and_machine_local_links_are_rejected(self):
        for link in ("[missing](no-file.md)", "[machine](/no-file.md)"):
            path = self.root / "README.md"
            path.write_text(link)
            manifest = self.root / "provenance/release_manifest.json"
            if manifest.exists():
                manifest.unlink()
            self.freeze()
            with self.assertRaises(RuntimeError):
                release_check.verify()


if __name__ == "__main__":
    unittest.main()
