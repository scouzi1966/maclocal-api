"""CPU-only checks for Python runtime and packaging dependency policy."""
from pathlib import Path
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 uses the build dependency tomli.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


class DependencyLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "uv.lock").open("rb") as lock_file:
            cls.lock = tomllib.load(lock_file)
        cls.packages = {}
        for package in cls.lock["package"]:
            cls.packages.setdefault(package["name"], []).append(package)

    @staticmethod
    def supports_legacy_python(package):
        return any(
            "python_full_version < '3.10'" in marker
            for marker in package.get("resolution-markers", [])
        )

    def test_python_runtime_floor_is_not_raised(self):
        self.assertEqual(self.lock["requires-python"], ">=3.9")

    def test_unpatchable_requests_twine_graph_is_absent_on_legacy_python(self):
        for name in ("requests", "requests-toolbelt", "twine"):
            with self.subTest(package=name):
                legacy = [
                    package for package in self.packages.get(name, [])
                    if self.supports_legacy_python(package)
                ]
                self.assertEqual(legacy, [])

    def test_modern_python_uses_patched_requests(self):
        versions = {
            package["version"] for package in self.packages.get("requests", [])
        }
        self.assertEqual(versions, {"2.34.2"})

    def test_vulnerable_legacy_setuptools_is_not_locked(self):
        self.assertNotIn("setuptools", self.packages.keys())

    def test_packaging_scripts_select_python_3_12(self):
        scripts = ("build-native-wheel.sh", "build-nightly-wheel.sh")
        for name in scripts:
            with self.subTest(script=name):
                content = (ROOT / "Scripts" / name).read_text()
                self.assertIn("--build-constraints", content)
                self.assertIn("--python 3.12", content)

    def test_packaging_build_constraints_pin_secure_setuptools(self):
        constraints = (ROOT / "Scripts" / "build-constraints.txt").read_text()
        self.assertIn("setuptools==83.0.0", constraints)


if __name__ == "__main__":
    unittest.main()
