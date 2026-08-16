from dataclasses import dataclass, field, fields
from typing import Any


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

    def snapshot(self) -> dict[str, Any]:
        excluded = {"current_plan", "memory"}
        return {item.name: getattr(self, item.name) for item in fields(self) if item.name not in excluded}

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
