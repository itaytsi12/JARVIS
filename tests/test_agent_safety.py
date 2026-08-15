import unittest

from brain.models import Action, ActionRisk
from brain.session_context import SessionContext
from security.safety import classify_action, may_auto_execute


class SafetyTests(unittest.TestCase):
    def test_destructive_commands_never_auto_execute(self):
        for command in ("shutdown /s", "git reset --hard", "rm -rf data", "diskpart", "taskkill /f /im explorer.exe"):
            action = Action("run_command", {"command": command})
            self.assertEqual(classify_action(action), ActionRisk.HIGH_IMPACT)
            self.assertFalse(may_auto_execute(action))

    def test_sensitive_values_are_redacted(self):
        action = Action("browser_type", {"target": "Password", "text": "secret"}, sensitive_fields={"text"})
        self.assertEqual(action.safe_args()["text"], "<REDACTED>")

    def test_basic_reference_resolution(self):
        context = SessionContext(last_opened_app="notepad", last_opened_file="x.txt")
        self.assertEqual(context.resolve_target("it"), "notepad")
        self.assertEqual(context.resolve_target("the file"), "x.txt")


if __name__ == "__main__":
    unittest.main()
