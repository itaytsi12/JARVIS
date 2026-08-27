"""Measure the complex-agent path, end to end, on real tasks.

    python scripts/benchmark_agent.py            # dry run: no model call at all
    python scripts/benchmark_agent.py --run      # REAL, PAID Anthropic calls
    python scripts/benchmark_agent.py --run --only 2
    python scripts/benchmark_agent.py --run --tag after --save

The dry run measures everything that does not need the model -- tool latency,
context size, tool-schema count, selected effort -- so the cheap half of a
before/after comparison costs nothing. `--run` adds the real numbers: model
calls, agent steps, tokens, cache hits, time to first tool, time to the first
spoken sentence, total duration and cost.

`--save` writes the result to `data/benchmarks/agent-<tag>.json` so a later
run can be diffed against it rather than against a number in a chat log.

Never run by the automated suite (`pytest.ini` limits discovery to `tests/`).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "benchmarks"

#: The representative read-only tasks. Read-only on purpose: a benchmark must
#: be repeatable, and must never change the repository it measures.
TASKS = [
    (
        "files",
        "Tell me what files are in the JARVIS project folder and briefly explain the important ones. "
        "Do not modify anything.",
    ),
    (
        "git",
        "Run git status in the JARVIS project and tell me what changed. Do not modify anything.",
    ),
    (
        "architecture",
        "Inspect the JARVIS project and explain how the main components are connected. Do not modify anything.",
    ),
]


def show(text: str) -> None:
    """Print without ever failing on the console encoding -- a real answer can
    contain characters cp1252 cannot encode."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def measure_tools() -> dict:
    """Tool latency, warm and cold-ish, with no model involved.

    `inspect_project` used to walk every virtualenv, `.git` and cache in the
    tree because `rglob` cannot prune -- 98 seconds on this repository, more
    than the whole rest of the run put together. This is the number that
    proves it stays fixed.
    """
    from brain.tool_catalog import get_tool_catalog
    from brain.agent_loop import _observation_text

    catalog = get_tool_catalog()
    root = str(PROJECT_ROOT)
    results = {}
    for name, arguments in (
        ("list_files", {"path": root}),
        ("inspect_project", {"path": root}),
        ("read_text_file", {"path": str(PROJECT_ROOT / "main.py")}),
        ("search_code", {"path": root, "query": "def route_command"}),
    ):
        timings = []
        observation = ""
        for _ in range(2):
            started = time.perf_counter()
            result = catalog.execute(name, arguments)
            timings.append((time.perf_counter() - started) * 1000)
            observation = _observation_text(result)
        results[name] = {
            "first_ms": round(timings[0], 1),
            "warm_ms": round(min(timings), 1),
            "observation_chars": len(observation),
        }
    return results


def measure_context() -> dict:
    """Context size, tool-schema count and selected effort per task."""
    from brain.agent_service import select_effort
    from brain.context_builder import ContextBuilder
    from brain.tool_catalog import ToolCatalog
    from memory.agent_memory import get_agent_memory
    from skills import get_skill_registry

    registry = get_skill_registry()
    catalog = ToolCatalog()
    memory = get_agent_memory()
    per_task = {}
    for key, goal in TASKS:
        skills = registry.select(goal)
        wanted = {d.name for skill in skills for d in skill.tools(catalog)} | {"remember_fact", "recall_memory"}
        specs = catalog.specs(names=wanted)
        started = time.perf_counter()
        retrieved = memory.retrieve(goal)
        retrieval_ms = (time.perf_counter() - started) * 1000
        context = ContextBuilder().build(
            goal, retrieved=retrieved, skills=skills, tool_names=[spec.name for spec in specs]
        )
        schema_chars = sum(len(json.dumps(spec.to_dict())) for spec in specs)
        per_task[key] = {
            "skills": [skill.name for skill in skills],
            "effort": select_effort(goal, skills),
            "tool_schemas_selected": len(specs),
            "tool_schemas_available": len(catalog.names()),
            "tool_schema_chars": schema_chars,
            "system_prompt_chars": len(context.system_prompt),
            "memory_retrieval_ms": round(retrieval_ms, 1),
            "first_call_input_tokens_estimate": (
                len(context.system_prompt) + len(context.user_prompt) + schema_chars
            ) // 4,
        }
    return per_task


def run_live(keys: list[str]) -> dict:
    """Execute the tasks against the real API and record what happened."""
    from config import configure_logging, get_config
    from providers.registry import get_agent_provider

    configure_logging()
    if not get_config().has_anthropic_credentials or get_agent_provider() is None:
        show("No agent provider available; skipping the live half.")
        return {}

    from brain.agent import run_agent
    from voice.agent_narration import AgentNarrator

    live = {}
    for key, goal in TASKS:
        if key not in keys:
            continue
        show(f"\n--- {key}: {goal[:70]}...")
        marks: dict[str, float] = {}
        started = time.perf_counter()

        def stamp(name: str) -> None:
            marks.setdefault(name, (time.perf_counter() - started) * 1000)

        spoken: list[str] = []

        def speak(text: str) -> None:
            stamp("first_speech_ms")
            spoken.append(text)

        narrator = AgentNarrator(speak=speak, started_at=time.monotonic())
        stream = narrator.answer_stream()
        # Exactly what the voice runtime does, so the measured
        # time-to-first-speech is the one a user would experience.
        narrator.start_heartbeat()

        def on_progress(stage: str, payload: dict) -> None:
            if stage == "tool_result":
                stamp("first_tool_observed_ms")
            narrator.on_event(stage, payload)

        def on_text(chunk: str) -> None:
            stamp("first_answer_token_ms")
            stream.feed(chunk)

        outcome: dict = {}
        try:
            answer = run_agent(goal, execution_outcome=outcome, progress=on_progress, on_answer_text=on_text)
        finally:
            narrator.stop()
        stream.flush()
        total_ms = (time.perf_counter() - started) * 1000

        live[key] = {
            "total_ms": round(total_ms),
            "model_calls": outcome.get("model_calls"),
            "agent_steps": outcome.get("agent_steps"),
            "tools": outcome.get("agent_tools"),
            "input_tokens": outcome.get("input_tokens"),
            "output_tokens": outcome.get("output_tokens"),
            "cache_creation_tokens": outcome.get("cache_creation_tokens"),
            "cache_read_tokens": outcome.get("cache_read_tokens"),
            "cost_usd": outcome.get("estimated_cost_usd"),
            "effort": outcome.get("effort"),
            "context_chars": outcome.get("context_chars"),
            "selected_tool_count": outcome.get("selected_tool_count"),
            "available_tool_count": outcome.get("available_tool_count"),
            "parallel_tool_batches": outcome.get("parallel_tool_batches"),
            "parallel_saved_ms": round(outcome.get("parallel_saved_ms") or 0.0),
            "time_to_first_tool_ms": round(outcome.get("time_to_first_tool_ms") or 0.0) or None,
            "model_first_event_ms": round(outcome.get("model_first_event_ms") or 0.0) or None,
            "time_to_first_speech_ms": round(marks.get("first_speech_ms", 0)) or None,
            "time_to_first_answer_token_ms": round(marks.get("first_answer_token_ms", 0)) or None,
            "progress_lines": len(narrator.spoken),
            "answer_chunks": len(stream.spoken_chunks),
            "success": outcome.get("success"),
            "verified": outcome.get("verified"),
        }
        for name, value in live[key].items():
            show(f"    {name:32s}: {value}")
        show(f"    answer: {answer[:200]}...")
    return live


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Execute the tasks against the real API (PAID)")
    parser.add_argument("--only", type=int, help="Run just one task, by its number")
    parser.add_argument("--tag", default="run", help="Label for the saved result file")
    parser.add_argument("--save", action="store_true", help="Write the result to data/benchmarks/")
    arguments = parser.parse_args()

    keys = [TASKS[arguments.only - 1][0]] if arguments.only else [key for key, _ in TASKS]

    report = {
        "tag": arguments.tag,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tools": measure_tools(),
        "context": measure_context(),
    }
    show("=== tool latency (no model) ===")
    for name, values in report["tools"].items():
        show(f"  {name:16s} first={values['first_ms']:9.1f} ms  warm={values['warm_ms']:8.1f} ms  observation={values['observation_chars']} chars")
    show("\n=== context and tool schemas (no model) ===")
    for key, values in report["context"].items():
        show(
            f"  {key:13s} tools={values['tool_schemas_selected']}/{values['tool_schemas_available']} "
            f"schema_chars={values['tool_schema_chars']} system={values['system_prompt_chars']} "
            f"effort={values['effort']} est_first_call_tokens={values['first_call_input_tokens_estimate']}"
        )

    if arguments.run:
        show("\n=== live execution (real, paid) ===")
        report["live"] = run_live(keys)
        live = report.get("live") or {}
        if live:
            show("\n=== summary ===")
            for key, values in live.items():
                show(
                    f"  {key:13s} total={values['total_ms']:6d} ms  calls={values['model_calls']}  "
                    f"steps={values['agent_steps']}  in={values['input_tokens']}  out={values['output_tokens']}  "
                    f"cache_read={values['cache_read_tokens']}  first_speech={values['time_to_first_speech_ms']} ms  "
                    f"cost=${values['cost_usd']}"
                )
            totals = [values["total_ms"] for values in live.values()]
            show(f"  median total: {statistics.median(totals):.0f} ms")
    else:
        show("\n(dry run -- pass --run to execute against the real API)")

    if arguments.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        destination = RESULTS_DIR / f"agent-{arguments.tag}.json"
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        show(f"\nsaved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
