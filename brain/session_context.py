import time
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class AppEvent:
    """One recorded application lifecycle event, timestamped and turn-tagged.

    `turn` is `SessionContext.turn_counter` at the moment this happened --
    the primary signal `brain.context_resolver` uses to decide whether a
    pronoun like "it" has exactly one plausible target (the previous turn
    introduced only one app) or is genuinely ambiguous (the previous turn
    introduced more than one, e.g. "open Chrome and Spotify"). `at` is a
    `time.monotonic()` timestamp used only for TTL/staleness checks, never
    for ordering -- wall-clock gaps between ordinary sequential commands are
    usually too small to be a reliable ordering signal by themselves.
    """

    name: str
    kind: str  # "opened" | "focused" | "closed"
    at: float
    turn: int


@dataclass
class ResultItem:
    """One item of a `ResultSet`, addressable by ordinal ("the second one")."""

    index: int  # 1-based, matches how people say "the first/second/third one"
    label: str
    value: Any
    kind: str = "item"


@dataclass
class ResultSet:
    """An ordered list of items a follow-up ordinal reference ("the first
    one", "the second result") can resolve against. Replaced wholesale by
    the next tool call that produces a new list -- never merged -- so an
    ordinal reference can never silently mix items from two different
    listings."""

    items: list[ResultItem]
    source: str  # the tool/action that produced this list, e.g. "list_files"
    kind: str  # "file" | "test_failure" | "search_result" | "line" | ...
    query: str | None
    turn: int
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class SessionContext:
    active_app: str | None = None
    last_opened_app: str | None = None
    last_hwnd: int | None = None
    last_pid: int | None = None
    application_windows: dict[str, int] = field(default_factory=dict)
    browser_active: bool = False
    current_url: str | None = None
    last_clicked_element: str | None = None
    last_search_query: str | None = None
    last_search_provider: str | None = None
    last_opened_file: str | None = None
    last_opened_folder: str | None = None
    current_plan: Any = field(default=None, repr=False)
    previous_action: str | None = None
    previous_result: str | None = None
    last_user_message: str | None = None
    last_assistant_response: str | None = None
    last_spoken_response: str | None = None
    interrupted_response: str | None = None
    speech_interrupted: bool = False
    last_messaging_recipient: str | None = None
    pending_messaging_recipient: str | None = None
    pending_messaging_message: str | None = None
    messaging_committed: bool = False
    memory: Any = field(default=None, repr=False)
    session_id: str | None = field(default=None, repr=False)

    # -- generalized short-term conversational context ---------------------
    # A monotonically increasing "turn" counter, bumped once per top-level
    # user request (brain.agent.run_agent). This is the ordering signal
    # brain.context_resolver uses to decide "which app/entity is CURRENT",
    # not wall-clock time -- see AppEvent's docstring.
    turn_counter: int = 0
    # Recency-ordered application lifecycle (opened/focused/closed), capped
    # so it never grows unbounded across a long-running process.
    recent_apps: list[AppEvent] = field(default_factory=list)
    browser_updated_at: float = 0.0
    # Project/workspace the user is currently talking about ("my JARVIS
    # project"). Longer-lived than app/task context -- see
    # brain.context_resolver's TTL constants.
    last_project_name: str | None = None
    last_project_path: str | None = None
    project_updated_at: float = 0.0
    last_opened_file_at: float = 0.0
    # The most recent ordered, addressable list a "the Nth one" reference
    # can resolve against (file listings, test failures, search results...).
    last_result_set: ResultSet | None = None
    last_selected_item: ResultItem | None = None
    # Agent-task / Task Manager follow-up context ("why did that fail?",
    # "fix the first one", "continue it").
    last_task_id: str | None = None
    last_task_goal: str | None = None
    last_task_status: str | None = None
    last_task_result_summary: str | None = None
    last_task_error: str | None = None
    task_context_updated_at: float = 0.0
    # The most recent error observed from ANY source (a failed tool call, a
    # failed task, a non-zero exit code) -- what "why did that fail?" and
    # "what does that mean?" resolve against.
    last_error: str | None = None
    last_error_source: str | None = None
    last_error_at: float = 0.0
    # The most recently executed single command/tool call, for "do it
    # again" and for corrections ("no, I meant Telegram") to replay against.
    last_command_tool: str | None = None
    last_command_args: dict[str, Any] | None = None
    last_command_text: str | None = None
    last_command_time: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        excluded = {"current_plan", "memory"}
        return {item.name: getattr(self, item.name) for item in fields(self) if item.name not in excluded}

    # -- writers -------------------------------------------------------
    def bump_turn(self) -> int:
        self.turn_counter += 1
        return self.turn_counter

    def record_app_event(self, name: str, kind: str) -> None:
        """kind is "opened" | "focused" | "closed"."""
        self.recent_apps.append(AppEvent(name=name, kind=kind, at=time.monotonic(), turn=self.turn_counter))
        if len(self.recent_apps) > 40:
            self.recent_apps = self.recent_apps[-40:]

    def open_app_names(self) -> list[str]:
        """Apps currently believed open, most-recently-touched last."""
        order: dict[str, None] = {}
        for event in self.recent_apps:
            order.pop(event.name, None)
            if event.kind != "closed":
                order[event.name] = None
        return list(order.keys())

    def last_turn_for_app(self, name: str) -> int | None:
        turn = None
        for event in self.recent_apps:
            if event.name == name and event.kind != "closed":
                turn = event.turn
        return turn

    def record_project(self, name: str, path: str | None = None) -> None:
        self.last_project_name = name
        self.last_project_path = path or name
        self.project_updated_at = time.monotonic()

    def record_result_set(self, items: list[tuple[str, Any]], source: str, kind: str, query: str | None = None) -> ResultSet:
        result_items = [ResultItem(index=i, label=label, value=value, kind=kind) for i, (label, value) in enumerate(items, start=1)]
        result_set = ResultSet(items=result_items, source=source, kind=kind, query=query, turn=self.turn_counter)
        self.last_result_set = result_set
        return result_set

    def record_task(self, task_id: str, goal: str, status: str, result_summary: str | None = None, error: str | None = None) -> None:
        self.last_task_id = task_id
        self.last_task_goal = goal
        self.last_task_status = status
        if result_summary is not None:
            self.last_task_result_summary = result_summary
        if error is not None:
            self.last_task_error = error
            self.record_error(error, source=f"task:{task_id}")
        self.task_context_updated_at = time.monotonic()

    def record_error(self, message: str, source: str | None = None) -> None:
        self.last_error = message
        self.last_error_source = source
        self.last_error_at = time.monotonic()

    def record_command(self, tool: str, args: dict[str, Any], text: str | None = None) -> None:
        self.last_command_tool = tool
        self.last_command_args = dict(args or {})
        self.last_command_text = text
        self.last_command_time = time.monotonic()

    # -- readers used by the pre-existing (deliberately small) resolvers ---
    def resolve_target(self, reference: str) -> str | None:
        value = reference.strip().lower()
        if value in {"it", "the app", "this app"}:
            return self.last_opened_app or self.active_app
        if value in {"the browser", "browser", "this window"}:
            return "browser" if self.browser_active else self.active_app
        if value in {"the file", "this file", "the document", "this document"}:
            return self.last_opened_file
        if value in {"the folder", "this folder"}:
            return self.last_opened_folder
        if self.memory:
            resolution = self.memory.resolve(reference, self.session_id, verify_live=True)
            if resolution.status == "resolved":
                entity = resolution.entity
                return entity["metadata"].get("path") or entity["name"]
            if resolution.status == "ambiguous":
                return None
        return reference

    def resolve_reference(self, reference: str):
        if not self.memory:
            return None
        return self.memory.resolve(reference, self.session_id, verify_live=True)

    def resolve_text_reference(self, reference: str) -> str | None:
        """Resolve recent assistant-output references without an API call."""
        value = " ".join(reference.lower().strip(" \t\r\n,.;!?'\"").split())
        value = value.removeprefix("exactly ")
        if value in {
            "what you said",
            "what you just said",
            "what you last said",
            "what you told me",
            "what you just told me",
            "what you last told me",
            "the last thing you said",
        }:
            return self.last_spoken_response or self.last_assistant_response
        if value in {
            "your last answer",
            "your last response",
            "the previous answer",
            "the previous response",
            "your previous answer",
            "your previous response",
            "that",
            "the last thing",
        }:
            return self.last_assistant_response or self.last_spoken_response
        return None
