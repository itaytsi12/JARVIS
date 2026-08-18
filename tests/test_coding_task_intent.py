import unittest

from brain.coding_task_intent import is_coding_task


class CodingTaskEligibilityTests(unittest.TestCase):
    def test_eligible_examples(self):
        eligible = [
            "fix this bug",
            "inspect this repository and fix the failing test",
            "add this feature to the code",
            "find why this class is behaving incorrectly",
            "refactor this implementation",
            "debug the crash in the executor module",
            "please fix the regression in the browser module",
            "self improve this codebase",
        ]
        for text in eligible:
            with self.subTest(text=text):
                self.assertTrue(is_coding_task(text), f"expected eligible: {text!r}")

    def test_ineligible_examples(self):
        ineligible = [
            "open chrome",
            "play music",
            "calculate 2+2",
            "search youtube for cats",
            "volume up",
            "what is the weather today",
            "open notepad",
            "mute",
            "take a screenshot",
            "fix this printer issue",
            "",
        ]
        for text in ineligible:
            with self.subTest(text=text):
                self.assertFalse(is_coding_task(text), f"expected ineligible: {text!r}")


if __name__ == "__main__":
    unittest.main()
