"""Reusable capabilities composed from tools -- see `skills/base.py`."""
from skills.base import Skill, SkillRegistry
from skills.builtin import ALL_SKILLS, build_default_registry

_REGISTRY: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_default_registry()
    return _REGISTRY


__all__ = ["Skill", "SkillRegistry", "ALL_SKILLS", "build_default_registry", "get_skill_registry"]
