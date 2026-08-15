from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SessionContext:
    active_app: str | None = None
    last_opened_app: str | None = None
    last_hwnd: int | None = None
    last_pid: int | None = None
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

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("current_plan", None)
        return data

    def resolve_target(self, reference: str) -> str | None:
        value = reference.strip().lower()
        if value in {"it", "the app", "this app"}:
            return self.last_opened_app or self.active_app
        if value in {"the browser", "browser", "this window"}:
            return "browser" if self.browser_active else self.active_app
        if value in {"the file", "this file"}:
            return self.last_opened_file
        if value in {"the folder", "this folder"}:
            return self.last_opened_folder
        return reference
