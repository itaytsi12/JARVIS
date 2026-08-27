import unittest
from unittest.mock import patch

from brain import project_registry


class ProjectRegistryTests(unittest.TestCase):
    def test_jarvis_resolves_out_of_the_box_to_the_real_repo_root(self):
        name, path = project_registry.resolve_project("jarvis")
        self.assertEqual(name, "jarvis")
        self.assertEqual(path, project_registry._REPO_ROOT)

    def test_my_x_project_and_the_x_project_phrasing_both_normalize(self):
        for phrase in ("my jarvis project", "the jarvis project", "jarvis project", "JARVIS"):
            with self.subTest(phrase=phrase):
                resolved = project_registry.resolve_project(phrase)
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved[0], "jarvis")

    def test_unknown_project_returns_none(self):
        self.assertIsNone(project_registry.resolve_project("some totally unknown thing"))

    def test_configured_project_is_discoverable(self):
        with patch.dict("os.environ", {"JARVIS_KNOWN_PROJECTS": "website:C:/dev/site, api : C:/dev/api"}):
            self.assertEqual(project_registry.resolve_project("website"), ("website", "C:/dev/site"))
            self.assertEqual(project_registry.resolve_project("my api project"), ("api", "C:/dev/api"))

    def test_empty_name_returns_none(self):
        self.assertIsNone(project_registry.resolve_project(""))
        self.assertIsNone(project_registry.resolve_project("   "))


if __name__ == "__main__":
    unittest.main()
