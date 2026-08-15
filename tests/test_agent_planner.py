import unittest

from brain.task_planner import create_task_plan, should_use_task_planner


class PlannerTests(unittest.TestCase):
    def tools(self, goal):
        plan = create_task_plan(goal)
        self.assertIsNotNone(plan, goal)
        return [action.tool for action in plan.actions]

    def test_required_task_sequences(self):
        cases = {
            "Open Notepad and type hello world.": ["open_application", "wait_for_window", "type_text"],
            "Open Notepad, type hello world, and save it to the Desktop as hello.txt.": ["open_application", "wait_for_window", "type_text", "write_text_file", "verify_file"],
            "Open YouTube and search for Jude Law.": ["browser_open_url"],
            "Open YouTube, search for Jude Law, and open the first result.": ["browser_open_url", "browser_click_first_result"],
            "Open YouTube, search for Jude Law, open the first video and make it fullscreen.": ["browser_open_url", "browser_click_first_result", "browser_fullscreen"],
            "Open Chrome, search Google for Python decorators, and open the first result.": ["browser_open_url", "browser_click_first_result"],
            "Open Chrome and maximize it.": ["open_application", "wait_for_window", "focus_application", "maximize_window"],
            "Open VS Code, then open Notepad, then switch back to VS Code.": ["open_application", "wait_for_window", "open_application", "wait_for_window", "focus_application"],
            "Open Notepad, type hello, then close it.": ["open_application", "wait_for_window", "type_text", "close_application"],
            "Create a text file on my Desktop called shopping.txt containing milk, eggs and bread.": ["create_text_file", "verify_file"],
            "Open my Downloads folder.": ["open_folder"],
            "Take a screenshot and open the screenshots folder.": ["take_screenshot", "open_file_explorer"],
            "Open Calculator, then open Notepad and type calculation complete.": ["open_application", "wait_for_window", "open_application", "wait_for_window", "type_text"],
            "Search YouTube for rock and roll.": ["browser_open_url"],
            "Search Google for black and white wallpapers.": ["browser_open_url"],
        }
        for goal, expected in cases.items():
            with self.subTest(goal=goal):
                self.assertEqual(self.tools(goal), expected)

    def test_protected_conjunctions_remain_in_query(self):
        for goal, phrase in (
            ("Search YouTube for rock and roll.", "rock+and+roll"),
            ("Search Google for black and white wallpapers.", "black+and+white+wallpapers"),
        ):
            plan = create_task_plan(goal)
            self.assertIn(phrase, plan.actions[0].args["url"])

    def test_compound_search_uses_local_planner(self):
        self.assertTrue(should_use_task_planner("Open YouTube and search for Jude Law"))
        self.assertTrue(should_use_task_planner("Open Notepad and type hello world"))

    def test_adversarial_phrasings(self):
        cases = {
            "Could you open YouTube and look up Jude Law for me?": ["browser_open_url"],
            "I want to see some Jude Law videos. Open YouTube and search for him.": ["browser_open_url"],
            "Bring up Chrome and look for Python decorators on Google.": ["browser_open_url"],
            "Start Notepad, write hello there, save it as test.txt on my Desktop and close it.": ["open_application", "wait_for_window", "type_text", "write_text_file", "verify_file", "close_application"],
            "Open Chrome, then Notepad, then go back to Chrome.": ["open_application", "wait_for_window", "open_application", "wait_for_window", "focus_application"],
            "Find black and white wallpapers on Google.": ["browser_open_url"],
            "Play rock and roll on YouTube.": ["browser_open_url"],
            "Open YouTube, search Minecraft building tutorials, choose the first result and fullscreen it.": ["browser_open_url", "browser_click_first_result", "browser_fullscreen"],
        }
        for goal, expected in cases.items():
            with self.subTest(goal=goal):
                self.assertEqual(self.tools(goal), expected)
        pronoun_plan = create_task_plan("I want to see some Jude Law videos. Open YouTube and search for him.")
        self.assertIn("Jude+Law", pronoun_plan.actions[0].args["url"])


if __name__ == "__main__":
    unittest.main()
