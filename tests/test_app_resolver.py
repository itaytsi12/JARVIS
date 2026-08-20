"""Tests for tools/app_resolver.py: conservative, structured app-name resolution."""
import unittest

from tools.app_index import AppEntry, AppIndex, normalize_app_name
from tools.app_resolver import resolve_app_name


def _index(*entries: AppEntry) -> AppIndex:
    return AppIndex(entries=list(entries))


DISCORD = AppEntry(display_name="Discord", normalized_name="discord", aliases=["discord"], launch_type="shortcut", launch_target="discord.lnk", executable_name="Discord.exe", source="start_menu")
STEAM = AppEntry(display_name="Steam", normalized_name="steam", aliases=["steam"], launch_type="exe", launch_target=r"C:\Steam\steam.exe", executable_name="steam.exe", source="app_paths")
VS_CODE = AppEntry(display_name="Visual Studio Code", normalized_name="visual studio code", aliases=["visual studio code", "code"], launch_type="shortcut", launch_target="vscode.lnk", executable_name="Code.exe", source="start_menu")
VISUAL_STUDIO = AppEntry(display_name="Visual Studio", normalized_name="visual studio", aliases=["visual studio"], launch_type="shortcut", launch_target="vs.lnk", executable_name="devenv.exe", source="start_menu")
CALCULATOR_UWP = AppEntry(display_name="Calculator", normalized_name="calculator", aliases=["calculator"], launch_type="uwp", app_user_model_id="Microsoft.WindowsCalculator_8wekyb3d8bbwe!App", source="start_apps")
OBS_STUDIO = AppEntry(display_name="OBS Studio", normalized_name="obs studio", aliases=["obs studio"], launch_type="shortcut", launch_target="obs.lnk", source="start_menu")


class ExactMatchTests(unittest.TestCase):
    def test_exact_display_name_match(self):
        resolution = resolve_app_name("Discord", index=_index(DISCORD, STEAM))
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.resolved_name, "Discord")
        self.assertEqual(resolution.resolution_method, "exact_display_name")
        self.assertEqual(resolution.confidence, 1.0)

    def test_exact_alias_match(self):
        resolution = resolve_app_name("code", index=_index(VS_CODE, STEAM))
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.resolved_name, "Visual Studio Code")
        self.assertEqual(resolution.resolution_method, "exact_alias")

    def test_exact_executable_name_match(self):
        # Aliases deliberately omit the executable stem so this can only be
        # resolved via the dedicated executable-name step, not exact_alias.
        launcher = AppEntry(display_name="Custom Launcher", normalized_name="custom launcher", aliases=["custom launcher"], launch_type="exe", launch_target=r"C:\Games\game.exe", executable_name="game.exe", source="app_paths")
        resolution = resolve_app_name("game.exe", index=_index(DISCORD, launcher))
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.resolved_name, "Custom Launcher")
        self.assertEqual(resolution.resolution_method, "exact_executable")

    def test_uwp_resolution_carries_app_user_model_id(self):
        resolution = resolve_app_name("calculator", index=_index(CALCULATOR_UWP))
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.launch_type, "uwp")
        self.assertEqual(resolution.app_user_model_id, "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App")


class NormalizationRobustnessTests(unittest.TestCase):
    def test_case_insensitive_match(self):
        self.assertTrue(resolve_app_name("DISCORD", index=_index(DISCORD)).success)

    def test_punctuation_insensitive_match(self):
        resolution = resolve_app_name("OBS-Studio", index=_index(OBS_STUDIO, STEAM))
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.resolved_name, "OBS Studio")

    def test_spoken_full_name_matches_shortcut_display_name(self):
        resolution = resolve_app_name("visual studio code", index=_index(VS_CODE, VISUAL_STUDIO))
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.resolved_name, "Visual Studio Code")


class FuzzyMatchTests(unittest.TestCase):
    def test_high_confidence_fuzzy_match_is_accepted(self):
        typo_entry = AppEntry(display_name="OBS Studio", normalized_name="obs studio", aliases=["obs studio"], launch_type="shortcut", launch_target="obs.lnk", source="start_menu")
        resolution = resolve_app_name("obs studeo", index=_index(typo_entry, STEAM))
        self.assertTrue(resolution.success)
        self.assertEqual(resolution.resolved_name, "OBS Studio")
        self.assertEqual(resolution.resolution_method, "fuzzy_match")

    def test_low_confidence_fuzzy_match_is_rejected(self):
        resolution = resolve_app_name("xyz totally unrelated gibberish", index=_index(DISCORD, STEAM, VS_CODE))
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.resolution_method, "not_found")


class AmbiguityTests(unittest.TestCase):
    def test_word_match_ambiguous_between_visual_studio_and_visual_studio_code(self):
        resolution = resolve_app_name("studio", index=_index(VISUAL_STUDIO, VS_CODE, DISCORD))
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.resolution_method, "ambiguous")
        self.assertEqual(set(resolution.candidates), {"Visual Studio", "Visual Studio Code"})

    def test_ambiguous_result_never_picks_a_winner(self):
        resolution = resolve_app_name("studio", index=_index(VISUAL_STUDIO, VS_CODE))
        self.assertIsNone(resolution.resolved_name)
        self.assertIsNone(resolution.launch_target)


class NotFoundTests(unittest.TestCase):
    def test_unknown_app_fails_cleanly(self):
        resolution = resolve_app_name("some completely unknown application", index=_index(DISCORD, STEAM))
        self.assertFalse(resolution.success)
        self.assertEqual(resolution.resolution_method, "not_found")
        self.assertEqual(resolution.error, "unknown_application")

    def test_empty_index_fails_cleanly(self):
        resolution = resolve_app_name("discord", index=_index())
        self.assertFalse(resolution.success)

    def test_empty_query_fails_cleanly(self):
        resolution = resolve_app_name("   ", index=_index(DISCORD))
        self.assertFalse(resolution.success)


if __name__ == "__main__":
    unittest.main()
