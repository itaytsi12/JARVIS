"""Tools whose execution and side effects are fully self-contained: they
never READ state written by a sibling action in the same plan/session
(`SessionContext.active_app`/`last_hwnd`/`last_pid`/etc), and no other tool
in this set depends on state THEY write either.

This is the single safety classification reused by two independent
features that both need exactly this property:

- `brain/speculative_execution.py` (Part B/C): safe to start early from an
  unstable partial realtime-STT transcript, before the user finishes
  speaking.
- `brain/agent_runtime.py` (Part H): safe to run CONCURRENTLY with sibling
  actions in the same plan that have no explicit `depends_on` wiring,
  since none of them read or race on shared mutable session state.

Keep this list conservative. Notably absent: `close_application` (may
discard unsaved work and reads `active_app`/`last_hwnd`), `type_text` /
`press_key` / `save_current_document` (read `active_app`/`last_hwnd` --
racing with a concurrent `open_application` would corrupt which window
they type into), `focus_application` / `click_ui_element` / any
`browser_*` tool (read/write shared browser/window session state), and
`send_whatsapp_message` (irreversible communication, never safe to start
speculatively or run unordered).
"""
from __future__ import annotations

CONTEXT_INDEPENDENT_TOOLS = frozenset({
    "open_application",
    "open_website",
    "volume_up",
    "volume_down",
    "mute_volume",
    "take_screenshot",
    "get_time",
    "get_date",
    # Music fast-path controls (see brain/music_intent.py::FAST_PATH_TOOLS,
    # the single source of truth this set mirrors): single-word, reversible,
    # idempotent-ish player commands that never read state a sibling action
    # writes. Deliberately excludes "music_play" and every other
    # entity-resolving music tool -- those need the full, final transcript
    # (Part 7: never guess a search result from a partial "play Starboy...").
    "open_music",
    "music_pause",
    "music_resume",
    "music_stop",
    "music_next",
    "music_previous",
})
