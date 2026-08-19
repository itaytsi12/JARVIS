"""Memory package.

Two layers live here:

- `MemoryManager` (pre-existing): session/entity resolution, artifacts and
  storage budgeting, used by the desktop-control runtime.
- `AgentMemory` (agent runtime): persistent conversation history,
  extracted long-term memory, episodic task memory and relevance
  retrieval. See `memory/agent_memory.py`.

They share the same process but not the same database file, so neither
layer's schema constrains the other.
"""
from .memory_manager import LocalFilesystemArchive, MemoryManager, Resolution, redact

__all__ = [
    "MemoryManager",
    "Resolution",
    "LocalFilesystemArchive",
    "redact",
    "AgentMemory",
    "get_agent_memory",
    "Episode",
    "StepRecord",
]


def __getattr__(name):
    """Expose the agent-memory layer lazily.

    Imported on demand rather than at package import time so that
    `from memory import MemoryManager` -- which the existing desktop
    runtime does on every start -- never pays for the agent schema, and
    so a circular import through `config` can't form during startup.
    """
    if name in {"AgentMemory", "get_agent_memory"}:
        from . import agent_memory

        return getattr(agent_memory, name)
    if name in {"Episode", "StepRecord"}:
        from . import episodic

        return getattr(episodic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
