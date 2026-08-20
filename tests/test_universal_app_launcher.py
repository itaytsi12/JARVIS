"""End-to-end coverage for the universal application launcher feature:

- explicit APP_ALIASES / built-in fast-path entries still win over the
  general resolver (never bypassed).
- unknown apps reach the general resolver and launch via whichever
  launch_type it resolved (uwp / shortcut / exe).
- ambiguous requests fail cleanly with candidates, never guess.
- an explicit, existing, safe direct path launches without going through
  PATH/registry/index lookups at all.
- the deterministic router path never needs a cloud planner/LLM call to
  route "open <anything>", including apps that were never hardcoded.
"""
import unittest
from unittest.mock import Mock, patch

from tools import applications
from tools.app_resolver import AppResolution


class ExplicitAliasPriorityTests(unittest.TestCase):
    def test_builtin_fast_path_wins_even_if_resolver_would_also_match(self):
        process = Mock(pid=111)
        with patch.object(applications.subprocess, "Popen", return_value=process) as popen, \
             patch.object(applications, "_wait_for_visible_window", return_value=42), \
             patch("tools.applications.resolve_app_name") as resolver:
            result = applications.open_application("notepad")
        self.assertTrue(result["success"])
        popen.assert_called_once_with("notepad.exe")
        resolver.assert_not_called()

    def test_vscode_alias_wins_over_resolver(self):
        process = Mock(pid=222)
        with patch.object(applications, "_resolve_vscode_command", return_value=[r"C:\VSCode\Code.exe"]), \
             patch.object(applications.subprocess, "Popen", return_value=process), \
             patch.object(applications, "_wait_for_visible_window", return_value=43), \
             patch("tools.applications.resolve_app_name") as resolver:
            result = applications.open_application("vs code")
        self.assertTrue(result["success"])
        resolver.assert_not_called()


class ResolverFallbackTests(unittest.TestCase):
    def _no_legacy_fallbacks(self):
        return patch.object(applications.shutil, "which", return_value=None), \
               patch.object(applications, "_resolve_start_app_command", return_value=None)

    def test_unknown_app_resolves_via_general_resolver_and_launches(self):
        process = Mock(pid=333)
        resolution = AppResolution(success=True, requested_name="obs studio", resolved_name="OBS Studio", confidence=0.95, resolution_method="exact_display_name", launch_type="shortcut", launch_target=r"C:\StartMenu\OBS Studio.lnk", source="start_menu")
        which, start_apps = self._no_legacy_fallbacks()
        with which, start_apps, \
             patch("tools.applications.resolve_app_name", return_value=resolution), \
             patch.object(applications.subprocess, "Popen", return_value=process) as popen, \
             patch.object(applications, "_wait_for_visible_window", return_value=77) as wait:
            result = applications.open_application("obs studio")
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["resolved_name"], "OBS Studio")
        self.assertEqual(result["resolution_method"], "exact_display_name")
        popen.assert_called_once_with(["explorer.exe", r"C:\StartMenu\OBS Studio.lnk"])
        wait.assert_called_once_with("OBS Studio", None)

    def test_uwp_resolution_launches_via_shell_apps_folder(self):
        process = Mock(pid=444)
        resolution = AppResolution(success=True, requested_name="photos", resolved_name="Photos", confidence=1.0, resolution_method="exact_display_name", launch_type="uwp", app_user_model_id="Microsoft.Windows.Photos_8wekyb3d8bbwe!App", source="start_apps")
        which, start_apps = self._no_legacy_fallbacks()
        with which, start_apps, \
             patch("tools.applications.resolve_app_name", return_value=resolution), \
             patch.object(applications.subprocess, "Popen", return_value=process) as popen, \
             patch.object(applications, "_wait_for_visible_window", return_value=88):
            result = applications.open_application("photos")
        self.assertTrue(result["success"])
        popen.assert_called_once_with(["explorer.exe", "shell:AppsFolder\\Microsoft.Windows.Photos_8wekyb3d8bbwe!App"])

    def test_exe_resolution_launches_the_resolved_target(self):
        process = Mock(pid=555)
        resolution = AppResolution(success=True, requested_name="discord", resolved_name="Discord", confidence=0.97, resolution_method="exact_alias", launch_type="exe", launch_target=r"C:\Users\Test\AppData\Local\Discord\Update.exe", arguments=["--processStart", "Discord.exe"], source="app_paths")
        which, start_apps = self._no_legacy_fallbacks()
        with which, start_apps, \
             patch("tools.applications.resolve_app_name", return_value=resolution), \
             patch.object(applications.subprocess, "Popen", return_value=process) as popen, \
             patch.object(applications, "_wait_for_visible_window", return_value=99):
            result = applications.open_application("discord")
        self.assertTrue(result["success"])
        popen.assert_called_once_with([r"C:\Users\Test\AppData\Local\Discord\Update.exe", "--processStart", "Discord.exe"])

    def test_ambiguous_resolution_fails_cleanly_without_launching(self):
        resolution = AppResolution(success=False, requested_name="studio", resolution_method="ambiguous", candidates=["Visual Studio", "Visual Studio Code"], error="ambiguous_application")
        which, start_apps = self._no_legacy_fallbacks()
        with which, start_apps, \
             patch("tools.applications.resolve_app_name", return_value=resolution), \
             patch.object(applications.subprocess, "Popen") as popen:
            result = applications.open_application("studio")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "ambiguous_application")
        self.assertEqual(result["candidates"], ["Visual Studio", "Visual Studio Code"])
        popen.assert_not_called()

    def test_unresolvable_app_fails_cleanly_without_launching(self):
        resolution = AppResolution(success=False, requested_name="totally unknown thing", resolution_method="not_found", error="unknown_application")
        which, start_apps = self._no_legacy_fallbacks()
        with which, start_apps, \
             patch("tools.applications.resolve_app_name", return_value=resolution), \
             patch.object(applications.subprocess, "Popen") as popen:
            result = applications.open_application("totally unknown thing")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "unknown_application")
        popen.assert_not_called()


class DirectPathTests(unittest.TestCase):
    def test_explicit_existing_exe_path_launches_directly_without_resolver(self):
        process = Mock(pid=666)
        with patch.object(applications.Path, "is_file", return_value=True), \
             patch.object(applications.subprocess, "Popen", return_value=process) as popen, \
             patch.object(applications, "_wait_for_visible_window", return_value=101), \
             patch("tools.applications.resolve_app_name") as resolver:
            result = applications.open_application(r"C:\Tools\portable_app.exe")
        self.assertTrue(result["success"])
        popen.assert_called_once_with([r"C:\Tools\portable_app.exe"])
        resolver.assert_not_called()

    def test_nonexistent_path_falls_through_to_the_resolver(self):
        resolution = AppResolution(success=False, requested_name="c:\\nope\\ghost.exe", resolution_method="not_found", error="unknown_application")
        with patch.object(applications.Path, "is_file", return_value=False), \
             patch.object(applications.shutil, "which", return_value=None), \
             patch.object(applications, "_resolve_start_app_command", return_value=None), \
             patch("tools.applications.resolve_app_name", return_value=resolution) as resolver:
            result = applications.open_application(r"C:\nope\ghost.exe")
        resolver.assert_called_once()
        self.assertFalse(result["success"])


class RouterDoesNotEscalateToCloudTests(unittest.TestCase):
    def test_open_command_for_an_uncatalogued_app_routes_deterministically_without_a_cloud_planner(self):
        from brain.router import route_command
        with patch("brain.planner.create_plan") as create_plan, \
             patch("brain.local_intent_model.route_with_local_model") as local_model, \
             patch("brain.intent_router.classify_intent") as classify_intent:
            route = route_command("open some totally uncatalogued application xyz")
        create_plan.assert_not_called()
        local_model.assert_not_called()
        classify_intent.assert_not_called()
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "open_application")
        self.assertEqual(route["arguments"]["app_name"], "some totally uncatalogued application xyz")


if __name__ == "__main__":
    unittest.main()
