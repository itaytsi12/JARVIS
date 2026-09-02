"""JARVIS's Obsidian vault: the persistent long-term brain.

The model's context window is temporary working memory. This vault is
permanent memory, and it is plain Markdown on disk -- so JARVIS reads and
writes it directly, Obsidian never has to be running, and when the user
DOES open Obsidian they see and can edit everything JARVIS knows.

    vault/note.py           the note format: frontmatter + Quick Summary
    vault/paths.py          where the vault is and what its folders are
    vault/manager.py        atomic read/write/create/move/archive
    vault/index.py          the cheap metadata map (stage 1 of retrieval)
    vault/retrieval.py      scan -> rank -> deep-read, with a trace
    vault/bootstrap.py      the canonical structure and its seed notes
    vault/jobs.py           Jobs, discovered from notes
    vault/skills.py         Skills, discovered from notes
    vault/missions.py       persistent mission records
    vault/daily.py          Daily Notes
    vault/priming.py        the knowledge boot before each mission
    vault/learning.py       corrections -> updated knowledge
    vault/consolidation.py  refinement instead of contradiction
    vault/projects.py       per-project knowledge
    vault/protected.py      what automation may never change
    vault/policy.py         is this request mission-shaped?
    vault/startup.py        startup memory recovery
    vault/tools.py          the vault tools the agent is offered
    vault/session.py        one mission's whole vault lifecycle

Everything is importable without side effects: nothing here touches the
filesystem at import time.
"""
from __future__ import annotations

from vault.manager import VaultError, VaultManager, get_vault, reset_vault
from vault.note import Note, build_note_text, parse_frontmatter, utc_now

__all__ = [
    "Note",
    "VaultError",
    "VaultManager",
    "build_note_text",
    "get_vault",
    "parse_frontmatter",
    "reset_vault",
    "utc_now",
    # Lazily exposed below.
    "bootstrap_vault",
    "ensure_vault_ready",
    "get_index",
    "get_journal",
    "get_job_registry",
    "get_mission_store",
    "get_primer",
    "get_project_memory",
    "get_retriever",
    "get_skill_library",
    "recover_session",
    "reset_all",
]

_LAZY = {
    "bootstrap_vault": ("vault.bootstrap", "bootstrap_vault"),
    "ensure_vault_ready": ("vault.bootstrap", "ensure_vault_ready"),
    "get_index": ("vault.index", "get_index"),
    "get_journal": ("vault.daily", "get_journal"),
    "get_job_registry": ("vault.jobs", "get_job_registry"),
    "get_mission_store": ("vault.missions", "get_mission_store"),
    "get_primer": ("vault.priming", "get_primer"),
    "get_project_memory": ("vault.projects", "get_project_memory"),
    "get_retriever": ("vault.retrieval", "get_retriever"),
    "get_skill_library": ("vault.skills", "get_skill_library"),
    "recover_session": ("vault.startup", "recover_session"),
}


def __getattr__(name: str):
    """Expose the subsystem entry points without importing them eagerly.

    Importing `vault` must stay cheap: `config/logging_setup.py` and
    `startup/launcher.py` both touch it during startup, and pulling in
    every submodule there would put the whole knowledge layer on the
    critical path of a `--status` call.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'vault' has no attribute {name!r}")
    module_name, attribute = target
    import importlib

    return getattr(importlib.import_module(module_name), attribute)


def reset_all() -> None:
    """Drop every cached singleton. For tests and an explicit reconfiguration."""
    from vault.daily import reset_journal
    from vault.index import reset_index
    from vault.jobs import reset_job_registry
    from vault.learning import reset_correction_learner
    from vault.manager import reset_vault as _reset_vault
    from vault.missions import reset_mission_store
    from vault.priming import reset_primer
    from vault.projects import reset_project_memory
    from vault.retrieval import reset_retriever
    from vault.skills import reset_skill_library

    for reset in (
        reset_journal,
        reset_index,
        reset_job_registry,
        reset_correction_learner,
        reset_mission_store,
        reset_primer,
        reset_project_memory,
        reset_retriever,
        reset_skill_library,
        _reset_vault,
    ):
        reset()
