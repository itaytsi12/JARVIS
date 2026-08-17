import tempfile
import unittest
from pathlib import Path

from training.code_model.context_packer import pack_repository_context


def _write(root: Path, relpath: str, content: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class PackRepositoryContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_seed_file_is_included(self):
        _write(self.root, "calc.py", "def add(a, b):\n    return a - b\n")
        ctx = pack_repository_context(self.root, seed_files=["calc.py"])
        self.assertEqual([f.path for f in ctx.files], ["calc.py"])

    def test_local_import_is_pulled_in(self):
        _write(self.root, "pkg/__init__.py", "")
        _write(self.root, "pkg/main.py", "from pkg import helper\n\ndef run():\n    return helper.do()\n")
        _write(self.root, "pkg/helper.py", "def do():\n    return 1\n")
        ctx = pack_repository_context(self.root, seed_files=["pkg/main.py"])
        paths = {f.path for f in ctx.files}
        self.assertIn("pkg/main.py", paths)
        self.assertIn("pkg/helper.py", paths)

    def test_test_files_are_included(self):
        _write(self.root, "calc.py", "def add(a, b):\n    return a - b\n")
        _write(self.root, "tests/test_calc.py", "def test_add():\n    assert True\n")
        ctx = pack_repository_context(self.root, seed_files=["calc.py"], test_files=["tests/test_calc.py"])
        paths = {f.path for f in ctx.files}
        self.assertIn("tests/test_calc.py", paths)

    def test_keyword_search_finds_relevant_files_with_no_seed(self):
        _write(self.root, "widget.py", "class StaleHandleError(Exception):\n    pass\n")
        _write(self.root, "unrelated.py", "def other():\n    return 42\n")
        ctx = pack_repository_context(self.root, keywords=["StaleHandleError"])
        paths = {f.path for f in ctx.files}
        self.assertIn("widget.py", paths)
        self.assertNotIn("unrelated.py", paths)

    def test_missing_file_is_silently_skipped(self):
        ctx = pack_repository_context(self.root, seed_files=["does/not/exist.py"])
        self.assertEqual(ctx.files, [])

    def test_respects_max_files(self):
        for i in range(10):
            _write(self.root, f"f{i}.py", f"X = {i}\n")
        ctx = pack_repository_context(self.root, keywords=["X ="], max_files=3)
        self.assertLessEqual(len(ctx.files), 3)

    def test_large_file_is_truncated_not_dropped(self):
        _write(self.root, "big.py", "x = 1\n" * 5000)
        ctx = pack_repository_context(self.root, seed_files=["big.py"])
        self.assertEqual(len(ctx.files), 1)
        self.assertIn("truncated", ctx.files[0].content)

    def test_render_produces_readable_text(self):
        _write(self.root, "calc.py", "def add(a, b):\n    return a + b\n")
        ctx = pack_repository_context(self.root, seed_files=["calc.py"])
        rendered = ctx.render()
        self.assertIn("calc.py", rendered)
        self.assertIn("def add", rendered)


if __name__ == "__main__":
    unittest.main()
