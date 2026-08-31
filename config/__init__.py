"""Centralized JARVIS configuration.

Everything that used to be read with a scattered `os.getenv(...)` call in
the new subsystems is declared once here, as a typed object, so a setting
can be discovered, documented and overridden in one place. Pre-existing
modules keep their own `os.getenv` reads -- this is additive, not a
forced migration.
"""
from config.settings import JarvisConfig, get_config, reload_config
from config.pricing import ModelPricing, estimate_cost, get_pricing_table

__all__ = [
    "JarvisConfig",
    "get_config",
    "reload_config",
    "ModelPricing",
    "estimate_cost",
    "get_pricing_table",
    "configure_logging",
    "configure_file_logging",
    "log_startup_status",
    "StageTimer",
]


def __getattr__(name):
    """Expose the logging helpers lazily.

    `config.logging_setup` imports `providers.registry` for its startup
    report, and `providers` imports `config` -- importing it eagerly here
    would make that a cycle.
    """
    if name in {
        "configure_logging",
        "configure_file_logging",
        "log_startup_status",
        "StageTimer",
        "describe_runtime",
    }:
        from config import logging_setup

        return getattr(logging_setup, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
