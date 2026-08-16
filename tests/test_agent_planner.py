import unittest

from brain.task_planner import assess_plan_completeness,create_task_plan,should_use_task_planner,validate_goal_coverage


class PlannerTests(unittest.TestCase):
    def test_question_result_chaining_never_types_unresolved_literal_payload(self):
        for command in ("Ask who created Minecraft, write the answer in Notepad, then open YouTube.","Who created Minecraft? Write the answer in Notepad, then open YouTube."):
            with self.subTest(command=command):
                plan=create_task_plan(command)
                self.assertEqual(plan.actions,[])
                self.assertIn("Ask me the question first",plan.context["clarification"])

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
            "Open Notepad, then click the Save button.": ["open_application","wait_for_window","click_ui_element"],
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

    def test_music_playback_local_plan_is_detected_as_incomplete(self):
        for goal in ("Open Apple Music and play I Love It","Open Spotify and play I Love It"):
            with self.subTest(goal=goal):
                plan=create_task_plan(goal)
                self.assertTrue(should_use_task_planner(goal));self.assertFalse(assess_plan_completeness(goal,plan)["complete"])
                self.assertNotEqual([action.args.get("app_name") for action in plan.actions if action.tool=="open_application"],["Groove Music"])

    def test_chrome_search_remains_complete_and_local(self):
        goal="Open Chrome and search for Minecraft";plan=create_task_plan(goal)
        self.assertEqual([action.tool for action in plan.actions],["open_application","wait_for_window","open_website"])
        self.assertTrue(assess_plan_completeness(goal,plan)["complete"]);self.assertIn("Minecraft",plan.actions[-1].args["url"])

    def test_goal_coverage_rejects_wrong_app_and_partial_playback(self):
        from brain.models import Action
        goal="Open Apple Music and play I Love It"
        self.assertIn("app_identity_mismatch",validate_goal_coverage(goal,[Action("open_application",{"app_name":"Groove Music"}),Action("type_text",{"text":"I Love It"}),Action("press_key",{"key":"enter"})]))
        self.assertIn("missing_playback_clause",validate_goal_coverage(goal,[Action("open_application",{"app_name":"apple music"})]))

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

    def test_semantic_click_reuses_opened_app_and_dependency(self):
        plan=create_task_plan("Open Notepad, then click the Save button.")
        click=plan.actions[-1];self.assertEqual(click.args,{"app_name":"notepad","name":"Save","control_type":"Button"});self.assertEqual(click.depends_on,[1]);self.assertEqual(plan.context["model_calls"],0)


if __name__ == "__main__":
    unittest.main()
