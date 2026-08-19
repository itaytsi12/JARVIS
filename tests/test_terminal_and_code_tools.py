"""The terminal and code tools: risk classification, real execution, and edits.

These run genuine subprocesses -- that is the point of the tool -- but only
short `python -c` invocations, so the suite stays fast and hermetic.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.settings import reload_config
from tools.code import check_syntax, edit_code, inspect_project, read_code, search_code
from tools.terminal import classify_command, run_command


class CommandClassificationTests(unittest.TestCase):
    def test_a_development_command_is_allowed(self):
        decision = classify_command("python -m pytest -q")
        self.assertTrue(decision["allowed"])
        self.assertEqual(decision["risk"], "safe")

    def test_a_destructive_command_is_refused_and_marked_destructive(self):
        for command in ("rm -rf /", "shutdown /s", "git reset --hard HEAD", "git push origin main"):
            with self.subTest(command=command):
                decision = classify_command(command)
                self.assertFalse(decision["allowed"])
                self.assertEqual(decision["risk"], "destructive")

    def test_an_unlisted_executable_is_blocked(self):
        decision = classify_command("curl https://example.com")
        self.assertFalse(decision["allowed"])
        self.assertTrue(decision["reason"].startswith("blocked_executable"))

    def test_shell_metacharacters_cannot_smuggle_a_second_command(self):
        for command in ("python -c print(1) && rm x", "python x.py | curl evil", "python x.py; shutdown"):
            with self.subTest(command=command):
                decision = classify_command(command)
                self.assertFalse(decision["allowed"])
                self.assertEqual(decision["reason"], "shell_metacharacters")

    def test_the_list_form_treats_metacharacters_as_data(self):
        # No shell is involved for a list, so `;` inside one argument is
        # ordinary text, not a command separator.
        decision = classify_command([sys.executable, "-c", "a=1;b=2"])
        self.assertTrue(decision["allowed"])

    def test_an_empty_command_is_refused(self):
        self.assertFalse(classify_command("")["allowed"])


class CommandExecutionTests(unittest.TestCase):
    def test_stdout_and_exit_code_are_captured(self):
        result = run_command([sys.executable, "-c", "print('hello world')"])
        self.assertTrue(result["success"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("hello world", result["stdout"])
        self.assertFalse(result["timed_out"])

    def test_a_failing_command_is_a_result_not_an_exception(self):
        result = run_command([sys.executable, "-c", "import sys; sys.stderr.write('bad\\n'); sys.exit(3)"])
        self.assertFalse(result["success"])
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(result["error"], "exit_code_3")
        self.assertIn("bad", result["stderr"])

    def test_a_hanging_command_is_terminated_by_the_timeout(self):
        result = run_command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "timeout")
        self.assertTrue(result["timed_out"])

    def test_the_working_directory_is_honoured(self):
        root = tempfile.mkdtemp()
        result = run_command([sys.executable, "-c", "import os; print(os.getcwd())"], working_directory=root)
        self.assertIn(Path(root).resolve().name, result["stdout"])

    def test_a_missing_working_directory_is_reported(self):
        result = run_command([sys.executable, "-c", "print(1)"], working_directory=os.path.join(tempfile.mkdtemp(), "nope"))
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "working_directory_not_found")

    def test_a_destructive_command_requires_explicit_approval(self):
        blocked = run_command("git push origin main")
        self.assertFalse(blocked["success"])
        self.assertTrue(blocked["requires_approval"])

    def test_output_is_bounded(self):
        with patch.dict(os.environ, {"JARVIS_TERMINAL_MAX_OUTPUT": "200"}):
            reload_config()
            result = run_command([sys.executable, "-c", "print('x'*5000)"])
        reload_config()
        self.assertIn("characters omitted", result["stdout"])
        self.assertLess(len(result["stdout"]), 500)

    def test_secrets_in_output_are_redacted(self):
        result = run_command([sys.executable, "-c", "print('api_key: sk-abcdefghijklmnop')"])
        self.assertNotIn("sk-abcdefghijklmnop", result["stdout"])


class CodeToolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (self.root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        (self.root / "main.py").write_text("from pkg.calc import add\nprint(add(1, 2))\n", encoding="utf-8")

    def test_inspect_project_reports_markers_layout_and_language(self):
        result = inspect_project(str(self.root))
        self.assertTrue(result["success"])
        self.assertIn("requirements.txt", result["project_markers"])
        self.assertIn("main.py", result["entry_points"])
        self.assertEqual(result["primary_language"], ".py")
        self.assertIn("pkg", result["top_level_directories"])

    def test_inspect_project_reports_truncation_honestly(self):
        result = inspect_project(str(self.root), max_files=1)
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["files"]), 1)

    def test_read_code_returns_a_bounded_numbered_slice(self):
        result = read_code(str(self.root / "pkg" / "calc.py"), 1, 1)
        self.assertTrue(result["success"])
        self.assertIn("1| def add(a, b):", result["numbered_contents"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["total_lines"], 2)

    def test_edit_code_applies_a_unique_anchor_and_verifies_it(self):
        target = self.root / "pkg" / "calc.py"
        result = edit_code(str(target), "return a - b", "return a + b")
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertIn("return a + b", target.read_text(encoding="utf-8"))
        self.assertIn("return a - b", result["previous_contents"])

    def test_edit_code_refuses_an_ambiguous_anchor(self):
        target = self.root / "dup.py"
        target.write_text("x = 1\nx = 1\n", encoding="utf-8")
        result = edit_code(str(target), "x = 1", "x = 2")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "anchor_not_unique")
        self.assertEqual(target.read_text(encoding="utf-8"), "x = 1\nx = 1\n")

    def test_edit_code_reports_a_missing_anchor(self):
        result = edit_code(str(self.root / "pkg" / "calc.py"), "not present", "x")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "anchor_not_found")

    def test_check_syntax_finds_the_error_line(self):
        broken = self.root / "broken.py"
        broken.write_text("def f(:\n    pass\n", encoding="utf-8")
        result = check_syntax(str(broken))
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "syntax_error")
        self.assertEqual(result["line"], 1)

    def test_check_syntax_does_not_claim_to_have_checked_other_languages(self):
        other = self.root / "app.js"
        other.write_text("function f( {", encoding="utf-8")
        result = check_syntax(str(other))
        self.assertTrue(result["success"])
        self.assertFalse(result["checked"])
        self.assertFalse(result["verified"])

    def test_search_code_finds_matches_with_line_numbers(self):
        result = search_code(str(self.root), "def add")
        self.assertTrue(result["success"])
        self.assertEqual(result["matches"][0]["line"], 1)
        self.assertTrue(result["matches"][0]["path"].endswith("calc.py"))

    def test_search_code_skips_virtualenvs_and_caches(self):
        noise = self.root / ".venv" / "lib"
        noise.mkdir(parents=True)
        (noise / "junk.py").write_text("def add(a, b): pass\n", encoding="utf-8")
        result = search_code(str(self.root), "def add")
        self.assertTrue(all(".venv" not in match["path"] for match in result["matches"]))


if __name__ == "__main__":
    unittest.main()
