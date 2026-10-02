import tempfile
import unittest
from pathlib import Path

from src.utils import load_config


class ConfigLoaderTest(unittest.TestCase):
    def test_relative_recursive_include_and_local_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fragments = root / "fragments"
            fragments.mkdir()

            (fragments / "base.yaml").write_text(
                "model:\n  type: test\nvalue: 1\n",
                encoding="utf-8",
            )
            (root / "config.yaml").write_text(
                "include: fragments/base.yaml\nvalue: 2\n",
                encoding="utf-8",
            )

            config = load_config(root / "config.yaml")

        self.assertEqual(config, {"model": {"type": "test"}, "value": 2})

    def test_include_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.yaml").write_text("include: b.yaml\n", encoding="utf-8")
            (root / "b.yaml").write_text("include: a.yaml\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Cyclic configuration include"):
                load_config(root / "a.yaml")


if __name__ == "__main__":
    unittest.main()

