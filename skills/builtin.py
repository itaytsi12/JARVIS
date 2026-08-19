"""The skills JARVIS ships with.

Each one bundles a coherent toolset with the guidance that makes those
tools usable. The guidance is written as operating rules, not as
encouragement -- notably every skill that changes something says how to
VERIFY it changed, because "I edited the file" is not evidence that the
program works.
"""
from __future__ import annotations

import re
from dataclasses import replace

from brain.tool_catalog import (
    AUDIO,
    BROWSER,
    CODE,
    COMPUTER,
    FILESYSTEM,
    INFO,
    MEMORY,
    TERMINAL,
    VISION,
)
from skills.base import Skill, SkillRegistry

CODING = Skill(
    name="coding",
    description="Inspect, run, debug and fix a software project, then prove the fix with a real test run.",
    tool_categories=(CODE, TERMINAL, FILESYSTEM),
    tool_names=("remember_fact",),
    keywords=(
        "code", "bug", "debug", "fix", "error", "crash", "test", "tests", "pytest",
        "project", "repository", "repo", "function", "traceback", "exception",
        "refactor", "implement", "compile", "build", "failing",
    ),
    completion_criteria="The program or its tests run again with a zero exit code, observed after the change.",
    guidance="""When working on code:
1. Orient first. Use inspect_project on the project root, then search_code and read_code
   to read only what is relevant. Do not read whole large files if a slice will do.
2. Reproduce before theorising. Run the program or its tests with run_command and read the
   actual error text and exit code. Never guess at a cause you have not observed.
3. Change one thing at a time. Use edit_code with an anchor that appears exactly once.
4. Re-run after every change. An edit is not a fix: only a fresh run with exit code 0 is
   evidence. If the run still fails, read the NEW error -- it is often a different one.
5. If three attempts do not converge, stop and report what you established, what you tried,
   and the exact remaining error. A precise "I am blocked here" beats a false success.
6. Never claim the code is fixed unless you have seen a passing run in this session.""",
)

COMPUTER_CONTROL = Skill(
    name="computer_control",
    description="Drive Windows applications and windows: open, focus, inspect, click and type.",
    tool_categories=(COMPUTER, AUDIO, VISION),
    keywords=(
        "open", "close", "launch", "start", "app", "application", "window", "click",
        "type", "press", "key", "notepad", "spotify", "volume", "mute", "screenshot",
        "minimize", "maximize", "desktop",
    ),
    completion_criteria="The target application is confirmed open/focused, or the typed text is confirmed present.",
    guidance="""When controlling the desktop:
1. Open or focus the target application first, then confirm it is really there with
   active_window or inspect_window before typing or clicking into it.
2. Prefer click_ui_element with a control name over click_at with coordinates; coordinates
   break the moment a window moves.
3. Use analyze_screen only when the UI state genuinely cannot be read any other way -- it is
   slow and costs a model call.
4. Report what you observed, not what you attempted. If focus could not be verified, say so.""",
)

FILES = Skill(
    name="files",
    description="Find, read, organize and write files and folders.",
    tool_categories=(FILESYSTEM,),
    tool_names=("inspect_project", "search_code"),
    keywords=(
        "file", "files", "folder", "directory", "organize", "organise", "rename", "move",
        "copy", "sort", "downloads", "documents", "desktop", "find",
    ),
    completion_criteria="Every intended file operation is confirmed by re-reading the filesystem.",
    guidance="""When working with files:
1. List and read before you move anything. Decide what a file is from its contents or
   extension, not from an assumption about its name.
2. Move and rename one file at a time, and check the result -- these tools refuse to
   overwrite an existing destination, so a clash is reported, not silently resolved.
3. Create a destination directory with create_directory before moving files into it.
4. You cannot delete files. If a task seems to need deletion, say so and stop.
5. Summarize what actually moved, with counts. Never report a plan as if it were done.""",
)

BROWSING = Skill(
    name="browser",
    description="Open web pages, search, read page content and interact with sites.",
    tool_categories=(BROWSER,),
    keywords=("website", "web", "browser", "url", "search", "google", "youtube", "page", "online", "site"),
    completion_criteria="The intended page is confirmed loaded, or the requested information was read from it.",
    guidance="""When using the browser:
1. open_website is enough to simply show a page to the user. Use the browser_* tools when you
   need to read or interact with the page yourself.
2. After navigating, check the returned page title/URL before acting on the page.
3. If a click does not change the page, treat that as a failure and try a different target
   rather than repeating the same click.""",
)

RESEARCH = Skill(
    name="research",
    description="Gather and summarize information, from the web or from local files, and answer with sources.",
    tool_categories=(BROWSER, INFO),
    tool_names=("read_text_file", "search_text", "recall_memory"),
    keywords=("research", "find out", "look up", "investigate", "summarize", "summarise", "compare", "explain", "what is", "who is"),
    completion_criteria="The question is answered from something actually read, with where it came from.",
    guidance="""When researching:
1. Check memory first with recall_memory -- the answer may already be known.
2. Read before summarizing, and say where each claim came from.
3. If sources disagree or you could not verify something, say that plainly instead of
   presenting a guess as a finding.""",
)

MEMORY_SKILL = Skill(
    name="memory",
    description="Remember durable facts about the user and recall them later.",
    tool_categories=(MEMORY,),
    keywords=("remember", "recall", "forget", "note", "preference", "my project", "from now on"),
    completion_criteria="The fact is stored, or the recalled facts are reported.",
    guidance="""When handling memory:
1. Store only durable facts: preferences, project locations, recurring workflows, corrections.
2. Never store one-off commands, transient state, or secrets of any kind.
3. When recalling, report what you actually found; do not invent a remembered fact.""",
)

# A goal that names a source file or a test command is a coding task even
# when it contains none of the obvious keywords, so the coding skill gets
# a precise extra matcher on top of its keyword list.
_CODING_SIGNALS = re.compile(
    r"(?:\brun (?:the )?(?:tests?|it|my (?:code|project|program))\b|\bpytest\b|\.py\b|"
    r"\bnpm (?:run|test)\b|\bexit code\b|\btraceback\b)",
    re.I,
)
CODING = replace(CODING, matcher=lambda text: bool(_CODING_SIGNALS.search(text)))

ALL_SKILLS = (CODING, COMPUTER_CONTROL, FILES, BROWSING, RESEARCH, MEMORY_SKILL)


def build_default_registry() -> SkillRegistry:
    return SkillRegistry(ALL_SKILLS)
