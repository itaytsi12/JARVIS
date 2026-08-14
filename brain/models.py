from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    success: bool
    tool: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None