"""Tests for tools/app_index.py: discovery, normalization, merge, and cache.

Every Windows-facing source (filesystem walk, winreg, PowerShell) is mocked
at its "raw fetch" boundary -- these tests never depend on what's actually
installed on the machine running them.
"""
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools import app_index
from tools.app_index import (
    AppEntry,
    AppIndex,
    normalize_app_name,
    discover_start_menu_shortcuts,
    discover_app_paths_registry,
    discover_uwp_apps,
    discover_installed_programs,
    build_index,
    get_index,
    refresh_index,
    search_index,
    _merge_entries,
    _resolve_shortcut_target,
)


class NormalizationTests(unittest.TestCase):
    def test_case_and_whitespace_are_folded(self):
        self.assertEqual(normalize_app_name("Visual Studio Code"), normalize_app_name("visual   studio    code"))

    def test_exe_suffix_is_stripped(self):
        self.assertEqual(normalize_app_name("Code.exe"), normalize_app_name("Code"))

    def test_punctuation_variants_normalize_the_same(self):
        variants = ["VS Code", "vs-code", "vs_code", "vs.code"]
        normalized = {normalize_app_name(v) for v in variants}
        self.assertEqual(len(normalized), 1)

    def test_trailing_filler_word_is_dropped(self):
        self.assertEqual(normalize_app_name("Spotify app"), normalize_app_name("Spotify"))

    def test_trailing_filler_word_is_not_dropped_when_it_is_the_whole_name(self):
        self.assertEqual(normalize_app_name("app"), "app")

    def test_empty_and_whitespace_only_names_normalize_to_empty(self):
        self.assertEqual(normalize_app_name(""), "")
        self.assertEqual(normalize_app_name("   "), "")


class StartMenuDiscoveryTests(unittest.TestCase):
    def test_exact_shortcut_becomes_an_entry(self):
        lnk = Path(r"C:\Users\Test\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Discord.lnk")
        with patch.object(app_index, "_iter_lnk_files", return_value=[lnk]), patch.object(app_index, "_resolve_shortcut_target", return_value={"target": r"C:\Users\Test\AppData\Local\Discord\Update.exe"}):
            entries = discover_start_menu_shortcuts()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.display_name, "Discord")
        self.assertEqual(entry.launch_type, "shortcut")
        self.assertEqual(entry.launch_target, str(lnk))
        self.assertEqual(entry.source, "start_menu")
        self.assertEqual(entry.executable_name, "Update.exe")
        self.assertIn(normalize_app_name("Discord"), entry.aliases)
        self.assertIn(normalize_app_name("Update"), entry.aliases)

    def test_nested_start_menu_shortcut_is_discovered(self):
        lnk = Path(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Microsoft Visual Studio\Visual Studio Code.lnk")
        with patch.object(app_index, "_iter_lnk_files", return_value=[lnk]), patch.object(app_index, "_resolve_shortcut_target", return_value=None):
            entries = discover_start_menu_shortcuts()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].display_name, "Visual Studio Code")
        self.assertEqual(entries[0].source_path, str(lnk))

    def test_a_shortcut_that_fails_to_parse_is_still_indexed_by_filename(self):
        lnk = Path(r"C:\Start Menu\Programs\Weird.lnk")
        with patch.object(app_index, "_iter_lnk_files", return_value=[lnk]), patch.object(app_index, "_resolve_shortcut_target", return_value=None):
            entries = discover_start_menu_shortcuts()
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].executable_name)
        self.assertEqual(entries[0].launch_type, "shortcut")

    def test_one_bad_shortcut_does_not_break_discovery_of_the_rest(self):
        good = Path(r"C:\Start Menu\Programs\Good.lnk")
        bad = Path(r"C:\Start Menu\Programs\Bad.lnk")

        def fake_resolve(path):
            if path == bad:
                raise OSError("boom")
            return None

        with patch.object(app_index, "_iter_lnk_files", return_value=[bad, good]), patch.object(app_index, "_resolve_shortcut_target", side_effect=fake_resolve):
            entries = discover_start_menu_shortcuts()
        self.assertEqual([e.display_name for e in entries], ["Good"])


class ShortcutBinaryParsingTests(unittest.TestCase):
    def _build_minimal_lnk(self, target: str) -> bytes:
        header = bytearray(76)
        header[0:4] = (0x4C).to_bytes(4, "little")
        flags = 0x2  # HasLinkInfo
        header[20:24] = flags.to_bytes(4, "little")

        target_bytes = target.encode("mbcs") + b"\x00"
        local_base_path_offset = 28  # LinkInfoHeaderSize (fixed, no unicode fields)
        link_info_size = local_base_path_offset + len(target_bytes)

        link_info = bytearray(local_base_path_offset)
        link_info[0:4] = link_info_size.to_bytes(4, "little")
        link_info[4:8] = (28).to_bytes(4, "little")  # LinkInfoHeaderSize
        link_info[8:12] = (0x1).to_bytes(4, "little")  # VolumeIDAndLocalBasePath present
        link_info[16:20] = local_base_path_offset.to_bytes(4, "little")
        link_info += target_bytes

        return bytes(header) + bytes(link_info)

    def test_local_base_path_is_extracted(self):
        blob = self._build_minimal_lnk(r"C:\Program Files\Widget\Widget.exe")
        with patch.object(Path, "read_bytes", return_value=blob):
            result = _resolve_shortcut_target(Path("fake.lnk"))
        self.assertEqual(result, {"target": r"C:\Program Files\Widget\Widget.exe"})

    def test_truncated_file_returns_none_without_raising(self):
        with patch.object(Path, "read_bytes", return_value=b"L\x00\x00\x00short"):
            self.assertIsNone(_resolve_shortcut_target(Path("fake.lnk")))

    def test_wrong_magic_returns_none(self):
        with patch.object(Path, "read_bytes", return_value=b"\x00" * 100):
            self.assertIsNone(_resolve_shortcut_target(Path("fake.lnk")))

    def test_unreadable_file_returns_none(self):
        with patch.object(Path, "read_bytes", side_effect=OSError("denied")):
            self.assertIsNone(_resolve_shortcut_target(Path("fake.lnk")))


class AppPathsRegistryTests(unittest.TestCase):
    def test_app_paths_entry_is_built_from_raw_tuples(self):
        raw = [("HKLM", "chrome.exe", r'"C:\Program Files\Google\Chrome\Application\chrome.exe"')]
        entries = discover_app_paths_registry(raw_entries=raw)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.display_name, "chrome")
        self.assertEqual(entry.launch_target, r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        self.assertEqual(entry.executable_name, "chrome.exe")
        self.assertEqual(entry.launch_type, "exe")
        self.assertEqual(entry.source, "app_paths")

    def test_empty_registry_yields_no_entries(self):
        self.assertEqual(discover_app_paths_registry(raw_entries=[]), [])


class UwpDiscoveryTests(unittest.TestCase):
    def test_uwp_entry_uses_app_user_model_id(self):
        raw = [{"Name": "Calculator", "AppID": "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"}]
        entries = discover_uwp_apps(raw_entries=raw)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.launch_type, "uwp")
        self.assertEqual(entry.app_user_model_id, "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App")
        self.assertEqual(entry.launch_target, "shell:AppsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App")
        self.assertEqual(entry.source, "start_apps")

    def test_entries_missing_name_or_appid_are_skipped(self):
        raw = [{"Name": "", "AppID": "x"}, {"Name": "y", "AppID": ""}, {"Name": "Real", "AppID": "real.id"}]
        entries = discover_uwp_apps(raw_entries=raw)
        self.assertEqual([e.display_name for e in entries], ["Real"])


def _fake_dir_entry(name: str, suffix: str) -> MagicMock:
    entry = MagicMock(spec=Path)
    entry.is_file.return_value = True
    entry.suffix = suffix
    entry.name = name
    entry.__str__.return_value = f"C:/Program Files/Widget/{name}"
    return entry


class InstalledProgramsDiscoveryTests(unittest.TestCase):
    def test_single_top_level_exe_is_used_as_launch_target(self):
        with patch.object(app_index.Path, "is_dir", return_value=True), patch.object(app_index.Path, "iterdir", return_value=[
            _fake_dir_entry("Widget.exe", ".exe"),
            _fake_dir_entry("readme.txt", ".txt"),
        ]):
            entries = discover_installed_programs(raw_entries=[("Widget", r"C:\Program Files\Widget", "HKLM\\...\\Widget")])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].launch_type, "exe")
        self.assertEqual(entries[0].executable_name, "Widget.exe")
        self.assertEqual(entries[0].source, "installed_programs")

    def test_ambiguous_install_dir_with_multiple_exes_is_not_used_as_a_launch_target(self):
        with patch.object(app_index.Path, "is_dir", return_value=True), patch.object(app_index.Path, "iterdir", return_value=[
            _fake_dir_entry("Widget.exe", ".exe"),
            _fake_dir_entry("Uninstall.exe", ".exe"),
        ]):
            entries = discover_installed_programs(raw_entries=[("Widget", r"C:\Program Files\Widget", "HKLM\\...\\Widget")])
        self.assertEqual(entries, [])

    def test_missing_install_location_yields_no_entry(self):
        entries = discover_installed_programs(raw_entries=[("Widget", "", "HKLM\\...\\Widget")])
        self.assertEqual(entries, [])


class MergeTests(unittest.TestCase):
    def test_entries_sharing_an_executable_name_are_merged(self):
        shortcut = AppEntry(display_name="Visual Studio Code", normalized_name="visual studio code", aliases=["visual studio code"], launch_type="shortcut", launch_target="a.lnk", executable_name="Code.exe", source="start_menu")
        app_path = AppEntry(display_name="code", normalized_name="code", aliases=["code"], launch_type="exe", launch_target=r"C:\VSCode\Code.exe", executable_name="Code.exe", source="app_paths")
        merged, report = _merge_entries([shortcut, app_path])
        self.assertEqual(len(merged), 1)
        self.assertEqual(set(merged[0].aliases), {"visual studio code", "code"})
        self.assertEqual(len(report), 1)

    def test_distinct_apps_are_not_merged(self):
        a = AppEntry(display_name="Discord", normalized_name="discord", executable_name="Discord.exe", source="start_menu")
        b = AppEntry(display_name="Steam", normalized_name="steam", executable_name="Steam.exe", source="start_menu")
        merged, report = _merge_entries([a, b])
        self.assertEqual(len(merged), 2)
        self.assertEqual(report, [])

    def test_higher_priority_source_wins_the_launch_target(self):
        start_menu = AppEntry(display_name="Chrome", normalized_name="chrome", executable_name="chrome.exe", launch_type="shortcut", launch_target="shortcut.lnk", source="start_menu")
        app_paths = AppEntry(display_name="chrome", normalized_name="chrome", executable_name="chrome.exe", launch_type="exe", launch_target=r"C:\chrome.exe", source="app_paths")
        merged, _ = _merge_entries([app_paths, start_menu])
        self.assertEqual(merged[0].launch_type, "shortcut")
        self.assertEqual(merged[0].launch_target, "shortcut.lnk")


class BuildIndexTests(unittest.TestCase):
    def test_build_index_combines_all_sources_and_survives_one_failing(self):
        with patch.object(app_index, "discover_start_menu_shortcuts", return_value=[AppEntry(display_name="A", normalized_name="a", executable_name="a.exe", source="start_menu")]), \
             patch.object(app_index, "discover_app_paths_registry", side_effect=RuntimeError("registry unavailable")), \
             patch.object(app_index, "discover_uwp_apps", return_value=[AppEntry(display_name="B", normalized_name="b", app_user_model_id="b.id", launch_type="uwp", source="start_apps")]), \
             patch.object(app_index, "discover_installed_programs", return_value=[]):
            index = build_index()
        self.assertEqual({e.display_name for e in index.entries}, {"A", "B"})


class CacheTests(unittest.TestCase):
    def setUp(self):
        app_index._INDEX_CACHE = None

    def tearDown(self):
        app_index._INDEX_CACHE = None

    def test_repeated_get_index_calls_only_scan_once(self):
        with patch.object(app_index, "_load_cache_file", return_value=None), \
             patch.object(app_index, "_save_cache_file"), \
             patch.object(app_index, "build_index", wraps=lambda: AppIndex(entries=[], built_at=1.0)) as build:
            get_index()
            get_index()
            get_index()
        self.assertEqual(build.call_count, 1)

    def test_force_refresh_always_rescans(self):
        with patch.object(app_index, "_load_cache_file", return_value=None), \
             patch.object(app_index, "_save_cache_file"), \
             patch.object(app_index, "build_index", wraps=lambda: AppIndex(entries=[], built_at=1.0)) as build:
            get_index()
            refresh_index()
        self.assertEqual(build.call_count, 2)

    def test_fresh_persistent_cache_is_reused_without_rescanning(self):
        cached = AppIndex(entries=[AppEntry(display_name="Cached", normalized_name="cached")], built_at=app_index.time.time())
        with patch.object(app_index, "_load_cache_file", return_value=cached), \
             patch.object(app_index, "build_index") as build:
            index = get_index()
        self.assertEqual(index.entries[0].display_name, "Cached")
        build.assert_not_called()

    def test_stale_persistent_cache_triggers_a_rebuild(self):
        stale = AppIndex(entries=[AppEntry(display_name="Stale", normalized_name="stale")], built_at=0.0)
        fresh = AppIndex(entries=[AppEntry(display_name="Fresh", normalized_name="fresh")], built_at=app_index.time.time())
        with patch.object(app_index, "_load_cache_file", return_value=stale), \
             patch.object(app_index, "_save_cache_file"), \
             patch.object(app_index, "build_index", return_value=fresh):
            index = get_index()
        self.assertEqual(index.entries[0].display_name, "Fresh")

    def test_corrupt_cache_file_triggers_a_rebuild_instead_of_crashing(self):
        with patch.object(app_index.Path, "read_text", return_value="{not valid json"), \
             patch.object(app_index, "_save_cache_file"), \
             patch.object(app_index, "build_index", return_value=AppIndex(entries=[], built_at=app_index.time.time())) as build:
            get_index()
        build.assert_called_once()

    def test_missing_cache_file_triggers_a_rebuild_instead_of_crashing(self):
        with patch.object(app_index.Path, "read_text", side_effect=FileNotFoundError()), \
             patch.object(app_index, "_save_cache_file"), \
             patch.object(app_index, "build_index", return_value=AppIndex(entries=[], built_at=app_index.time.time())) as build:
            get_index()
        build.assert_called_once()

    def test_round_trip_through_json_preserves_entries(self):
        index = AppIndex(entries=[AppEntry(display_name="X", normalized_name="x", aliases=["x"], executable_name="x.exe")], built_at=123.0)
        restored = AppIndex.from_json(json.loads(json.dumps(index.to_json())))
        self.assertEqual(restored.entries[0].display_name, "X")
        self.assertEqual(restored.built_at, 123.0)


class SearchIndexTests(unittest.TestCase):
    def test_search_matches_by_alias_or_token(self):
        index = AppIndex(entries=[
            AppEntry(display_name="Visual Studio Code", normalized_name="visual studio code", aliases=["visual studio code", "code"]),
            AppEntry(display_name="Discord", normalized_name="discord", aliases=["discord"]),
        ])
        results = search_index("code", index=index)
        self.assertEqual([e.display_name for e in results], ["Visual Studio Code"])


if __name__ == "__main__":
    unittest.main()
