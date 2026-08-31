"""The Windows observation tools added for the agent: clipboard, machine
state, file discovery and desktop scrolling.

Everything is mocked at the OS boundary -- no clipboard is touched, no
process list is read, no wheel event is sent -- so the suite stays safe to
run anywhere. What is asserted is the behaviour these tools were added
for: that they report honestly, that they never raise into the agent loop,
and that a shared OS resource being busy is reported as busy rather than
as a failure of the request.
"""
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from tools import clipboard, machine
from tools.files import file_info, recent_files
from tools.ui import scroll_screen


class _FakeClipboard:
    """A stand-in for `win32clipboard` holding one text value."""

    CF_UNICODETEXT = 13

    def __init__(self, text=None, busy_for=0):
        self.text = text
        self.busy_for = busy_for
        self.opens = 0
        self.closed = 0

    def OpenClipboard(self):
        self.opens += 1
        if self.opens <= self.busy_for:
            raise OSError("clipboard busy")

    def CloseClipboard(self):
        self.closed += 1

    def EmptyClipboard(self):
        self.text = None

    def IsClipboardFormatAvailable(self, fmt):
        return self.text is not None

    def GetClipboardData(self, fmt):
        return self.text

    def SetClipboardData(self, fmt, value):
        self.text = value


class ClipboardTests(unittest.TestCase):
    def _install(self, fake):
        return patch.object(clipboard, "_clipboard_module", return_value=fake)

    def test_write_then_read_round_trips(self):
        fake = _FakeClipboard()
        with self._install(fake):
            written = clipboard.write_clipboard("hello sir")
            read = clipboard.read_clipboard()
        self.assertTrue(written["success"])
        # Verified by reading it back, not by SetClipboardData returning.
        self.assertTrue(written["verified"])
        self.assertEqual(read["text"], "hello sir")

    def test_a_write_that_something_else_overwrote_is_not_reported_verified(self):
        """A clipboard manager can win the race. Reporting success without
        `verified` is the honest outcome; claiming it worked is not."""
        fake = _FakeClipboard()
        with self._install(fake):
            with patch.object(clipboard, "read_clipboard", return_value={"success": True, "text": "something else"}):
                result = clipboard.write_clipboard("hello sir")
        self.assertTrue(result["success"])
        self.assertFalse(result["verified"])

    def test_an_empty_clipboard_is_a_fact_not_a_failure(self):
        with self._install(_FakeClipboard(text=None)):
            result = clipboard.read_clipboard()
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertTrue(result["empty"])

    def test_a_briefly_busy_clipboard_is_retried_rather_than_failed(self):
        """Another process owning the clipboard for a moment is ordinary --
        Chrome and Office both do it -- so it must not surface as an error."""
        fake = _FakeClipboard(text="ok", busy_for=3)
        with self._install(fake):
            result = clipboard.read_clipboard()
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "ok")

    def test_a_permanently_busy_clipboard_reports_that_specifically(self):
        fake = _FakeClipboard(text="ok", busy_for=999)
        with self._install(fake):
            result = clipboard.read_clipboard()
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "clipboard_busy")

    def test_the_clipboard_is_always_closed_even_when_reading_raises(self):
        fake = _FakeClipboard(text="ok")
        with self._install(fake), patch.object(fake, "GetClipboardData", side_effect=RuntimeError("boom")):
            result = clipboard.read_clipboard()
        self.assertFalse(result["success"])
        self.assertEqual(fake.closed, 1)

    def test_oversized_text_is_refused_rather_than_truncated_silently(self):
        with self._install(_FakeClipboard()):
            result = clipboard.write_clipboard("x" * (clipboard.MAX_TEXT + 1))
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "text_too_long")


def _process(pid, name, rss_mb):
    process = MagicMock()
    process.info = {"pid": pid, "name": name, "memory_info": MagicMock(rss=rss_mb * 1024 * 1024)}
    return process


class ProcessTests(unittest.TestCase):
    def _psutil(self, processes):
        fake = MagicMock()
        fake.process_iter.return_value = processes
        return patch.object(machine, "_psutil", return_value=fake)

    def test_processes_are_ranked_by_memory_and_the_total_is_reported(self):
        processes = [_process(1, "chrome.exe", 400), _process(2, "code.exe", 900), _process(3, "notepad.exe", 5)]
        with self._psutil(processes):
            result = machine.list_processes(limit=2)
        self.assertEqual([row["name"] for row in result["processes"]], ["code.exe", "chrome.exe"])
        self.assertEqual(result["total"], 3)
        self.assertTrue(result["truncated"])

    def test_a_process_that_exits_mid_walk_is_skipped_not_fatal(self):
        broken = MagicMock()
        type(broken).info = property(lambda self: (_ for _ in ()).throw(RuntimeError("gone")))
        with self._psutil([broken, _process(2, "code.exe", 10)]):
            result = machine.list_processes()
        self.assertTrue(result["success"])
        self.assertEqual([row["name"] for row in result["processes"]], ["code.exe"])

    def test_process_running_matches_the_way_a_person_names_an_app(self):
        """The agent gets this argument from a spoken request, so "Google
        Chrome", "chrome" and "chrome.exe" all have to resolve."""
        processes = [_process(1, "chrome.exe", 1), _process(2, "chrome.exe", 1)]
        for spoken in ("chrome", "Chrome", "chrome.exe", "google chrome"):
            with self._psutil(processes):
                result = machine.process_running(spoken)
            self.assertTrue(result["running"], spoken)
            self.assertEqual(result["count"], 2, spoken)

    def test_something_not_running_is_reported_plainly(self):
        with self._psutil([_process(1, "chrome.exe", 1)]):
            result = machine.process_running("photoshop")
        self.assertTrue(result["success"])
        self.assertFalse(result["running"])
        self.assertEqual(result["pids"], [])

    def test_a_missing_psutil_is_reported_not_raised(self):
        with patch.object(machine, "_psutil", side_effect=ImportError):
            self.assertEqual(machine.list_processes()["error"], "psutil_unavailable")
            self.assertEqual(machine.process_running("x")["error"], "psutil_unavailable")
            self.assertEqual(machine.system_status()["error"], "psutil_unavailable")


class SystemStatusTests(unittest.TestCase):
    def test_a_machine_with_no_battery_reports_none_rather_than_guessing(self):
        fake = MagicMock()
        fake.cpu_percent.return_value = 12.0
        fake.cpu_count.return_value = 8
        fake.virtual_memory.return_value = MagicMock(percent=50.0, used=8 * 1024**3, total=16 * 1024**3)
        fake.sensors_battery.return_value = None
        with patch.object(machine, "_psutil", return_value=fake):
            result = machine.system_status()
        self.assertTrue(result["success"])
        self.assertIsNone(result["battery_percent"])
        self.assertIn("CPU 12%", result["message"])

    def test_one_failing_probe_does_not_lose_the_others(self):
        fake = MagicMock()
        fake.cpu_percent.side_effect = RuntimeError("no cpu counter")
        fake.virtual_memory.return_value = MagicMock(percent=50.0, used=8 * 1024**3, total=16 * 1024**3)
        fake.sensors_battery.return_value = None
        with patch.object(machine, "_psutil", return_value=fake):
            result = machine.system_status()
        self.assertTrue(result["success"])
        self.assertNotIn("cpu_percent", result)
        self.assertEqual(result["memory_percent"], 50.0)


class NetworkTests(unittest.TestCase):
    def test_a_real_connection_is_what_counts_as_online(self):
        with patch("socket.create_connection"):
            result = machine.network_status()
        self.assertTrue(result["online"])
        self.assertIsNotNone(result["latency_ms"])

    def test_a_refused_connection_is_offline_not_an_error(self):
        with patch("socket.create_connection", side_effect=OSError("refused")):
            result = machine.network_status()
        self.assertTrue(result["success"])
        self.assertFalse(result["online"])


class VolumeTests(unittest.TestCase):
    def test_setting_a_level_is_verified_by_reading_it_back(self):
        with patch.object(machine, "_endpoint_volume", return_value=MagicMock()), patch.object(
            machine, "_read_volume", return_value=(30, False)
        ):
            result = machine.set_volume(30)
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["volume"], 30)

    def test_a_device_that_quantizes_the_level_is_reported_honestly(self):
        """Some endpoints snap to their own steps. Saying "set to 30" when
        the device is at 25 would be a false claim."""
        with patch.object(machine, "_endpoint_volume", return_value=MagicMock()), patch.object(
            machine, "_read_volume", return_value=(25, False)
        ):
            result = machine.set_volume(30)
        self.assertFalse(result["verified"])
        self.assertEqual(result["volume"], 25)

    def test_out_of_range_and_non_numeric_levels_are_refused(self):
        for bad in (-1, 101, "loud", None):
            self.assertFalse(machine.set_volume(bad)["success"], bad)

    def test_an_unavailable_audio_interface_is_reported_not_raised(self):
        with patch.object(machine, "_endpoint_volume", side_effect=OSError("no device")):
            self.assertFalse(machine.get_volume()["success"])
            self.assertFalse(machine.set_volume(50)["success"])


class FileDiscoveryTests(unittest.TestCase):
    def test_file_info_reports_size_and_age_for_a_file(self):
        with TemporaryDirectory() as temp:
            target = Path(temp) / "notes.txt"
            target.write_text("hello", encoding="utf-8")
            result = file_info(str(target))
        self.assertTrue(result["success"])
        self.assertEqual(result["kind"], "file")
        self.assertEqual(result["size_bytes"], 5)
        self.assertLess(result["age_hours"], 1)

    def test_file_info_counts_entries_for_a_directory(self):
        with TemporaryDirectory() as temp:
            (Path(temp) / "a.txt").write_text("a", encoding="utf-8")
            (Path(temp) / "b.txt").write_text("b", encoding="utf-8")
            result = file_info(temp)
        self.assertEqual(result["kind"], "directory")
        self.assertEqual(result["entries"], 2)

    def test_a_missing_path_is_reported_not_raised(self):
        result = file_info(str(Path("no") / "such" / "file.txt"))
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_recent_files_returns_newest_first_within_the_window(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / "old.txt"
            new = root / "new.txt"
            old.write_text("old", encoding="utf-8")
            new.write_text("new", encoding="utf-8")
            long_ago = time.time() - 72 * 3600
            import os

            os.utime(old, (long_ago, long_ago))
            result = recent_files(str(root), within_hours=24)
        names = [item["name"] for item in result["items"]]
        self.assertIn("new.txt", names)
        self.assertNotIn("old.txt", names)

    def test_recent_files_prunes_noise_directories_rather_than_filtering_after(self):
        """Descending into a virtualenv to throw the results away is the
        cost `tools/code.py`'s pruning exists to avoid; this shares it."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".venv" / "lib").mkdir(parents=True)
            (root / ".venv" / "lib" / "junk.py").write_text("x", encoding="utf-8")
            (root / "real.py").write_text("x", encoding="utf-8")
            result = recent_files(str(root), within_hours=24)
        names = [item["name"] for item in result["items"]]
        self.assertEqual(names, ["real.py"])

    def test_recent_files_can_be_limited_to_particular_suffixes(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "keep.docx").write_text("x", encoding="utf-8")
            (root / "skip.py").write_text("x", encoding="utf-8")
            result = recent_files(str(root), within_hours=24, suffixes=["docx"])
        self.assertEqual([item["name"] for item in result["items"]], ["keep.docx"])

    def test_a_folder_that_does_not_exist_is_reported_not_raised(self):
        result = recent_files(str(Path("no") / "such" / "folder"))
        self.assertTrue(result["success"])
        self.assertEqual(result["items"], [])
        self.assertTrue(result["missing_roots"])


class ScrollTests(unittest.TestCase):
    def test_each_direction_sends_wheel_events(self):
        for direction in ("up", "down", "left", "right"):
            with patch("tools.ui.user32") as user32:
                result = scroll_screen(direction, clicks=2)
            self.assertTrue(result["success"], direction)
            self.assertEqual(user32.mouse_event.call_count, 2, direction)

    def test_an_unknown_direction_is_refused_before_anything_is_sent(self):
        with patch("tools.ui.user32") as user32:
            result = scroll_screen("sideways")
        self.assertFalse(result["success"])
        user32.mouse_event.assert_not_called()

    def test_coordinates_move_the_pointer_first_because_windows_scrolls_under_it(self):
        with patch("tools.ui.user32") as user32:
            scroll_screen("down", clicks=1, x=400, y=300)
        user32.SetCursorPos.assert_called_once_with(400, 300)

    def test_the_number_of_clicks_is_bounded(self):
        with patch("tools.ui.user32") as user32:
            scroll_screen("down", clicks=10_000)
        self.assertLessEqual(user32.mouse_event.call_count, 30)

    def test_scrolling_never_claims_the_content_actually_moved(self):
        """Verifying that needs a screenshot, which is the caller's
        decision -- so this reports the action, not an outcome."""
        with patch("tools.ui.user32"):
            self.assertFalse(scroll_screen("down")["verified"])


if __name__ == "__main__":
    unittest.main()
