"""Model pricing, kept in exactly one place so it can be corrected without
touching any call site.

Prices are USD per one MILLION tokens, matching how providers publish
them. They change over time, so:

- the built-in table is a starting point, not an authority;
- `JARVIS_PRICING_FILE` may point at a JSON file that overrides or adds
  entries;
- an unknown model yields `None` rather than a fabricated number, and
  every consumer must treat `None` as "cost unknown", not "cost zero".
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("jarvis.pricing")


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens for one model."""

    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None


# Prefix-matched (longest prefix wins) so a model id that carries a suffix
# still resolves without a new entry per release. Prices are Anthropic's
# published first-party API rates; cache write/read are the standard
# 1.25x / 0.1x of the input rate for the default 5-minute TTL.
def _claude(input_per_mtok: float, output_per_mtok: float) -> ModelPricing:
    return ModelPricing(
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
        cache_write_per_mtok=round(input_per_mtok * 1.25, 6),
        cache_read_per_mtok=round(input_per_mtok * 0.10, 6),
    )


_BUILTIN: dict[str, ModelPricing] = {
    "claude-fable-5": _claude(10.0, 50.0),
    "claude-mythos-5": _claude(10.0, 50.0),
    "claude-opus-5": _claude(5.0, 25.0),
    "claude-opus-4-8": _claude(5.0, 25.0),
    "claude-opus-4-7": _claude(5.0, 25.0),
    "claude-opus-4-6": _claude(5.0, 25.0),
    "claude-sonnet-5": _claude(3.0, 15.0),
    "claude-sonnet-4-6": _claude(3.0, 15.0),
    "claude-haiku-4-5": _claude(1.0, 5.0),
}

_TABLE: dict[str, ModelPricing] | None = None
_LOCK = threading.Lock()


def _load_overrides() -> dict[str, ModelPricing]:
    path_text = (os.getenv("JARVIS_PRICING_FILE") or "").strip()
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.is_file():
        log.warning("JARVIS_PRICING_FILE points at a missing file; using built-in pricing only")
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.exception("Could not read JARVIS_PRICING_FILE; using built-in pricing only")
        return {}
    overrides: dict[str, ModelPricing] = {}
    for model, values in (raw or {}).items():
        if not isinstance(values, dict):
            continue
        try:
            overrides[str(model)] = ModelPricing(
                input_per_mtok=float(values["input_per_mtok"]),
                output_per_mtok=float(values["output_per_mtok"]),
                cache_write_per_mtok=(
                    float(values["cache_write_per_mtok"]) if values.get("cache_write_per_mtok") is not None else None
                ),
                cache_read_per_mtok=(
                    float(values["cache_read_per_mtok"]) if values.get("cache_read_per_mtok") is not None else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            log.warning("Ignoring malformed pricing entry for %r", model)
    return overrides


def get_pricing_table() -> dict[str, ModelPricing]:
    global _TABLE
    if _TABLE is None:
        with _LOCK:
            if _TABLE is None:
                _TABLE = {**_BUILTIN, **_load_overrides()}
    return _TABLE


def reload_pricing_table() -> dict[str, ModelPricing]:
    global _TABLE
    with _LOCK:
        _TABLE = {**_BUILTIN, **_load_overrides()}
    return _TABLE


def pricing_for(model: str | None) -> ModelPricing | None:
    if not model:
        return None
    table = get_pricing_table()
    exact = table.get(model)
    if exact is not None:
        return exact
    matches = [key for key in table if model.startswith(key)]
    if not matches:
        return None
    return table[max(matches, key=len)]


def estimate_cost(
    model: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float | None:
    """USD estimate, or None when this model's price is genuinely unknown.

    Never guesses: an unpriced model returns None so callers report
    "cost unknown" instead of an invented 0.0.
    """
    pricing = pricing_for(model)
    if pricing is None:
        return None
    total = (input_tokens / 1_000_000) * pricing.input_per_mtok
    total += (output_tokens / 1_000_000) * pricing.output_per_mtok
    if cache_creation_tokens and pricing.cache_write_per_mtok is not None:
        total += (cache_creation_tokens / 1_000_000) * pricing.cache_write_per_mtok
    if cache_read_tokens and pricing.cache_read_per_mtok is not None:
        total += (cache_read_tokens / 1_000_000) * pricing.cache_read_per_mtok
    return round(total, 8)
