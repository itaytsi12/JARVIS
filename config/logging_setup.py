"""Observability: one place that decides what gets logged and how.

The goal is that after a request you can reconstruct: what arrived, how
it was routed, which task handled it, which model was called, which tools
ran, whether each succeeded, how long each stage took, and what failed.
Without flooding the terminal to get it.

Two levels:

- default: INFO for JARVIS's own loggers, WARNING for third-party
  chatter (httpx, urllib3, anthropic's transport). Enough to follow a
  request; quiet enough to read.
- `JARVIS_DEBUG=1`: DEBUG for JARVIS, INFO for third parties, and the
  per-step plan trace turns on.

`StageTimer` is the latency instrument. It measures real elapsed time per
stage and reports only stages that actually ran -- a stage a request
never reached is absent, never fabricated as 0.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from config import get_config

JARVIS_LOGGERS = (
    "jarvis",
    "jarvis.agent",
    "jarvis.context",
    "jarvis.memory",
    "jarvis.tasks",
    "jarvis.tools",
    "jarvis.terminal",
    "jarvis.usage",
    "jarvis.providers",
    "jarvis.runtime",
    "jarvis.episodes",
)

NOISY_THIRD_PARTIES = ("httpx", "httpcore", "urllib3", "anthropic", "openai", "asyncio", "PIL")

_CONFIGURED = False


def configure_logging(force: bool = False) -> None:
    """Install JARVIS's logging configuration exactly once.

    Never touches an already-configured root handler set unless `force`
    is passed, so importing JARVIS from another application does not
    hijack that application's logging.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    debug = get_config().debug
    level = logging.DEBUG if debug else logging.INFO

    root = logging.getLogger()
    if not root.handlers or force:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        if force:
            for existing in list(root.handlers):
                root.removeHandler(existing)
        root.addHandler(handler)
    root.setLevel(logging.WARNING)

    for name in JARVIS_LOGGERS:
        logging.getLogger(name).setLevel(level)
    for name in NOISY_THIRD_PARTIES:
        logging.getLogger(name).setLevel(logging.INFO if debug else logging.WARNING)

    _CONFIGURED = True


@dataclass
class StageTimer:
    """Per-request latency instrumentation.

    Usage:
        timer = StageTimer("request")
        with timer.stage("routing"):
            ...
        timer.log(log)
    """

    label: str = "request"
    started: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str, milliseconds: float) -> None:
        self.stages[name] = round(self.stages.get(name, 0.0) + milliseconds, 3)

    def stage(self, name: str):
        timer = self

        class _Stage:
            def __enter__(self):
                self._started = time.perf_counter()
                return self

            def __exit__(self, *exc_info):
                timer.mark(name, (time.perf_counter() - self._started) * 1000)
                return False

        return _Stage()

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 3)

    def summary(self) -> dict[str, Any]:
        return {"label": self.label, "total_ms": self.total_ms, **self.stages}

    def log(self, logger: logging.Logger | None = None, **extra: Any) -> None:
        logger = logger or logging.getLogger("jarvis")
        payload = {**self.summary(), **extra}
        logger.info(
            "%s latency: %s",
            self.label,
            " ".join(f"{key}={value}" for key, value in payload.items() if key != "label"),
        )


def describe_runtime() -> dict[str, Any]:
    """A single log-safe snapshot of how JARVIS is configured right now.

    Printed at startup so a support question ("why isn't Claude being
    used?") is answerable from the log rather than by guessing.
    """
    from providers.registry import provider_status

    config = get_config()
    return {
        "debug": config.debug,
        "data_dir": str(config.data_dir),
        "agent_enabled": config.agent_enabled,
        "agent_escalation": config.agent_escalation_enabled,
        "max_agent_steps": config.max_agent_steps,
        "max_concurrent_tasks": config.max_concurrent_tasks,
        "cost_tracking": config.cost_tracking_enabled,
        "memory_enabled": config.memory_enabled,
        "providers": provider_status(),
    }


def log_startup_status(logger: logging.Logger | None = None) -> dict[str, Any]:
    logger = logger or logging.getLogger("jarvis")
    status = describe_runtime()
    providers = status["providers"]
    active = providers.get("active_provider")
    if active:
        logger.info("Agent provider active: %s (%s)", active, providers.get("active_model"))
    else:
        reasons = [
            f"{entry.get('provider')}={entry.get('unavailable_reason')}"
            for entry in providers.get("providers", [])
            if not entry.get("available")
        ]
        logger.info(
            "No agent provider available; local commands only. %s",
            ", ".join(reasons) or "no providers registered",
        )
    logger.info(
        "Runtime: debug=%s data_dir=%s max_agent_steps=%s max_concurrent_tasks=%s",
        status["debug"],
        status["data_dir"],
        status["max_agent_steps"],
        status["max_concurrent_tasks"],
    )
    return status
