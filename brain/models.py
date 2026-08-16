from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PlanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionRisk(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    HIGH_IMPACT = "high_impact"


@dataclass
class Action:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    optional: bool = False
    verify: str | None = None
    stop_condition: str | None = None
    risk: ActionRisk = ActionRisk.SAFE
    sensitive_fields: set[str] = field(default_factory=set, repr=False)
    max_attempts: int | None = None

    def safe_args(self) -> dict[str, Any]:
        return {
            key: "<REDACTED>" if key in self.sensitive_fields else value
            for key, value in self.args.items()
        }


@dataclass
class ToolResult:
    success: bool
    tool: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Plan:
    original_goal: str
    actions: list[Action]
    current_action_index: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    status: PlanStatus = PlanStatus.PENDING
    completed_actions: list[int] = field(default_factory=list)
    failed_action: int | None = None
    retry_count: int = 0
    failure_information: str | None = None

    def safe_goal(self) -> str:
        result = self.original_goal
        for action in self.actions:
            for field in action.sensitive_fields:
                value = action.args.get(field)
                if isinstance(value, str) and value:
                    result = result.replace(value, "<REDACTED>")
        return result
