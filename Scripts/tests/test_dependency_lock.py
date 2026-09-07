"""CPU-only checks for the universal Python dependency lock."""
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DependencyLockTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / "uv.lock").open("rb") as lock_file:
            self.lock = tomllib.load(lock_file)
        self.packages = {}
        for package in self.lock["package"]:
            self.packages.setdefault(package["name"], []).append(package)

    def test_python_runtime_floor_is_not_raised(self):
        self.assertEqual(self.lock["requires-python"], ">=3.9")

    def test_legacy_python_resolution_uses_patched_requests(self):
        requests = self.packages["requests"]
        legacy = next(
            package for package in requests
            if any("python_full_version < '3.10'" in marker for marker in package.get("resolution-markers", []))
        )
        self.assertEqual(legacy["version"], "2.32.5")

    def test_vulnerable_legacy_setuptools_is_not_locked(self):
        self.assertNotIn(
            "setuptools",
            self.packages.keys(),
            "Python 3.10+ packaging environments should resolve setuptools 83+, "
            "while Python 3.9 packaging must use an isolated modern interpreter",
        )

    def test_legacy_python_uses_supportable_twine(self):
        twine = self.packages["twine"]
        legacy = next(
            package for package in twine
            if any("python_full_version < '3.10'" in marker for marker in package.get("resolution-markers", []))
        )
        self.assertEqual(legacy["version"], "6.2.0")


if __name__ == "__main__":
    unittest.main()
