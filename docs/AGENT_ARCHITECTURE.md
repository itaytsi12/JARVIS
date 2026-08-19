# JARVIS agent architecture

This document describes the agent runtime added on top of the existing
desktop-control system. The pre-existing subsystems (voice pipeline, music,
self-improvement, continual learning) are documented in `CLAUDE.md` and are
unchanged — everything here is additive.

The organising idea: **Claude is a model used by JARVIS, not the
architecture.** The agent loop, the tool contract, the task manager, memory
and cost accounting all live in JARVIS. A provider only ever answers one
turn at a time.

---

## 1. Request lifecycle

```
input (voice or text)
        |
        v
brain/router.py :: route_command          <- deterministic, no model
        |
        +-- resolved tool / local_plan --> brain/executor.py -> tools/  (FAST PATH)
        |
        +-- unresolvable (plan / ai / coding_task) or explicit "agent: ..."
                |
                v
        brain/agent.py :: _agent_escalation_available()
                |  (False without an API key -> pre-existing path, unchanged)
                v
        brain/agent_service.py :: run_agent_task
                |
                +-- memory/agent_memory.py   retrieve relevant memory
                +-- skills/                  select skills for the goal
                +-- brain/context_builder.py build a budgeted context
                |
                v
        brain/agent_loop.py :: AgentLoop.run
                |
                +--> providers/  (one model turn)
                +--> brain/tool_catalog.py -> brain/executor.py -> tools/
                +--> observation fed back; repeat
                |
                v
        episode + memory written, short answer returned
```

### Fast path

`route_command` resolves "open Spotify", "volume down", "mute", "what time
is it", "calculate 527 * 93", "type hello" and the rest with regular
expressions and lookup tables. No model, no network, sub-millisecond. This
path is untouched by the agent work and is covered by
`tests/test_agent_routing.py::FastPathTests`.

### Agent path

Reached only when the deterministic layer genuinely cannot resolve the
request, or when the user explicitly asks for it (`agent: ...`,
`jarvis, figure out ...`). With no `ANTHROPIC_API_KEY` the escalation
check returns False and behaviour is byte-for-byte what it was before.

---

## 2. Modules

| Area | Module | Responsibility |
|---|---|---|
| Config | `config/settings.py` | One typed `JarvisConfig`, cached, `reload_config()` for tests |
| Config | `config/pricing.py` | Model prices in one place; unknown model -> `None`, never 0 |
| Config | `config/logging_setup.py` | Log levels, `StageTimer`, startup status report |
| Providers | `providers/base.py` | Vendor-neutral `ModelProvider`, messages, tool specs, errors |
| Providers | `providers/anthropic_provider.py` | The only module that imports `anthropic` |
| Providers | `providers/mock_provider.py` | `ScriptedProvider` / `CallableProvider` for tests and demos |
| Providers | `providers/registry.py` | Provider selection and the escalation ladder |
| Providers | `providers/usage.py` | `UsageStore` (SQLite) + `TrackedProvider` wrapper |
| Tools | `brain/tool_catalog.py` | Machine-readable tool specs, schema validation, typed execution |
| Tools | `tools/terminal.py` | Risk-classified command execution with timeouts |
| Tools | `tools/code.py` | Project inspection, bounded reads, anchored edits, syntax check |
| Skills | `skills/base.py`, `skills/builtin.py` | Reusable capabilities above tools |
| Agent | `brain/context_builder.py` | Budgeted context assembly |
| Agent | `brain/agent_loop.py` | The real loop, with every safety limit |
| Agent | `brain/agent_service.py` | Glue: memory + skills + loop + episode + task |
| Tasks | `tasks/models.py` | `Task`, `TaskStatus`, `CancellationToken` |
| Tasks | `tasks/manager.py` | Concurrency, UI exclusivity, cancellation |
| Tasks | `tasks/store.py` | Task persistence and restart reconciliation |
| Memory | `memory/agent_store.py` | Shared SQLite database for the three memory kinds |
| Memory | `memory/conversation.py` | Raw persistent conversation history |
| Memory | `memory/long_term.py` | Extraction rules + durable facts |
| Memory | `memory/episodic.py` | Full structured episodes (also the training record) |
| Memory | `memory/retrieval.py` | Relevance + recency ranking |
| Memory | `memory/agent_memory.py` | The facade the rest of JARVIS uses |

`memory/memory_manager.py` (the pre-existing entity/session store) is
untouched and still used by the desktop runtime. The two layers share a
process but not a database file.

---

## 3. The agent loop

`brain/agent_loop.py`. One turn at a time:

1. check cancellation and the wall-clock deadline;
2. ask the provider for one turn, given the budgeted context and the
   tool specs for the selected skills;
3. if it returned no tool call, that is the final answer — conclude;
4. otherwise execute each tool call through the catalog, turn each
   `ToolResult` into an observation (leading with `FAILED (error_code)`
   when it failed), and feed them all back in one message;
5. repeat.

### Safety limits

| Limit | Env var | Default | Behaviour at the limit |
|---|---|---|---|
| Model turns | `JARVIS_MAX_AGENT_STEPS` | 25 | Stop, report partial progress |
| Identical failing call | `JARVIS_MAX_ACTION_RETRIES` | 2 | Refuse the 3rd, tell the model to change approach |
| Consecutive failures | `JARVIS_MAX_CONSECUTIVE_FAILURES` | 4 | Stop, report honestly |
| Wall clock | `JARVIS_AGENT_TASK_TIMEOUT` | 900s | Stop, report partial progress |

None of these is tight enough to break ordinary multi-step work — an
eight-step plan completes normally (`tests/test_agent_loop.py`).

### Success vs verified

- `success`: the last acting step did not fail.
- `verified`: the last acting step succeeded **and** independently
  confirmed its own outcome (`result.data["verified"]`).

A run with no tool calls at all is never `verified`. An edit alone is
never treated as a fix — only a fresh passing run is.

---

## 4. Tools

Every tool is declared once in `brain/tool_catalog.py::DEFINITIONS` with a
name, a description written for a model, a JSON Schema, a category, a risk
level, its exclusive resource, and whether it is read-only.

Execution always returns the project's existing
`brain.models.ToolResult` and **never raises**: an unknown tool, invalid
arguments and a tool that itself fails are all unsuccessful results with a
specific error, so the loop can read the failure and adapt.

Execution goes through `brain/executor.py` / `brain/agent_runtime.py`,
which already own resource locking and retry — the catalog adds the schema
and description layer, not a second execution path.

Notably absent: any file-deletion tool. Reading is safe; destroying is not.

### Adding a tool

1. Implement it in the right `tools/` module, returning a dict with
   `success`, `message`, and `verified` when the outcome was independently
   confirmed.
2. Dispatch it in `brain/tool_router.py::execute_tool` (the single
   dispatch point).
3. Declare it in `brain/tool_catalog.py::DEFINITIONS` with its schema,
   category and risk.
4. If it drives the keyboard/mouse/browser, add it to the right set in
   `brain/resource_locks.py`.

---

## 5. Skills

A skill (`skills/base.py`) bundles a coherent toolset with the guidance
that makes those tools usable, plus keywords/matcher for selection without
a model call. Shipped: `coding`, `computer_control`, `files`, `browser`,
`research`, `memory`.

Selection is `SkillRegistry.select(goal)`; when nothing clearly applies it
returns nothing and the agent gets the general toolset rather than being
forced into a bad fit.

### Adding a skill

Append a `Skill(...)` to `skills/builtin.py::ALL_SKILLS`. Give it
`tool_categories` and/or `tool_names`, keywords, guidance written as
operating rules, and a `completion_criteria` that says how the work is
verified. No runtime change is needed.

---

## 6. Model providers

`providers/base.py` defines the interface. `complete()` is deliberately
single-turn.

### Adding a provider

```python
class MyLocalProvider:
    name = "local"
    model = "my-model"

    def is_available(self) -> bool: ...
    def unavailable_reason(self) -> str | None: ...
    def describe(self) -> dict: ...
    def complete(self, messages, *, system=None, tools=None, **kwargs) -> ModelResponse: ...

register_provider("local", MyLocalProvider)
```

Then put it earlier in `JARVIS_PROVIDER_ORDER`. That is the whole
escalation ladder: `deterministic local -> (future local model) -> Claude`.

There is no fake provider in the production ladder. When nothing is
available `get_agent_provider()` returns `None` and callers degrade to
local behaviour — they never pretend a model answered.

---

## 7. Memory

Three kinds, one database (`memory/agent_store.py`):

- **Conversation** — every turn, persisted. Raw history.
- **Long-term** — only what is worth keeping. `extract_memories` promotes
  explicit instructions ("remember that…", "from now on…"), stated
  preferences, stated facts and corrections. "open YouTube" is never
  promoted. Storage is idempotent on (kind, text), so repeating a fact
  strengthens it rather than duplicating it.
- **Episodic** — the full structured record of each handled request.

Retrieval (`memory/retrieval.py`) ranks by token overlap, recency and
importance/outcome, and returns a bounded slice. An episode with zero
lexical overlap is dropped outright, however recent it was.

`EMBEDDING_BACKEND` in that module is the single hook a future vector
retriever registers into; nothing else would change.

---

## 8. Episodes and the training dataset

Every handled agent request writes one `Episode` containing the request,
context summary, route, model, plan, every step (tool, arguments,
success, verified, observation, error, duration, attempt, the model's own
stated reasoning for that step), errors, retries, final result, success,
verified, duration, token usage and estimated cost.

Stored twice, deliberately:

- a normalized, indexed SQLite row (queryable);
- an append-only JSONL line at `data/episodes/episodes.jsonl` (the
  untouched raw payload for a future training pipeline).

Nothing is summarized away on write. Processing happens downstream.

Reward-relevant signals are captured directly: `success`, `verified`,
`retries`, `error_count`, `duration_ms`, `stop_reason`, `user_correction`.

This is separate from, and complementary to, the pre-existing
`training_data/` recorder, which continues to capture every route.

---

## 9. Tasks and concurrency

`tasks/manager.py` uses threads, not asyncio — every capability JARVIS
drives (pywinauto, Playwright sync, `subprocess`, Windows message pumps,
audio) is blocking, and the existing runtime is built around a
thread-affine process-wide lock. `run_async()` lets an asyncio caller
await a task without either side blocking.

- `TaskKind.CONCURRENT` — research, file reading, test runs. Run in
  parallel up to `JARVIS_MAX_CONCURRENT_TASKS`.
- `TaskKind.EXCLUSIVE_UI` — anything driving the real keyboard, mouse or
  foreground window. Serialized behind a single semaphore, so two of them
  can never type into each other's windows.

`brain/agent_service.py` classifies a goal automatically from its selected
skills (`computer_control` / `browser` -> `EXCLUSIVE_UI`).

This is *on top of* the per-tool locks in `brain/resource_locks.py`, not
instead of them.

There is a second, finer split inside `brain/tool_catalog.py::_dispatch`.
Desktop and browser tools (`SESSION_AWARE_CATEGORIES`, plus `open_path`)
go through `AgentRuntime`, which owns the window/PID session state *and*
the process-wide `action_plan` lock — so two agent tasks touching the
desktop still serialize, exactly as a voice command does. Everything else
— filesystem, terminal, code, info, memory — runs via
`Executor.execute_action_unlocked_plan`, taking only its own per-tool
resource lock. Without that split, "run the tests while researching
something else" would serialize behind a lock whose purpose is protecting
the keyboard.

### Interruption

Every task carries a `CancellationToken`. `cancel()` returns immediately;
the loop notices before its next model call or tool call. A task queued
behind a UI task can be cancelled without waiting for the queue. In voice
mode an agent run is dispatched off the capture thread, so the microphone
keeps working — and "cancel" still works — while it runs.

---

## 10. Cost and observability

`providers/usage.py` records one row per model call: provider, model,
operation, session, task, every token count the provider reported,
latency, success, and the cost estimate from `config/pricing.py`. An
unpriced model records `NULL`, never `0.0`, and summaries expose
`cost_is_complete` so an incomplete total is never mistaken for a real one.

`TrackedProvider` wraps any provider so cost tracking is not something a
call site can forget.

Prices live only in `config/pricing.py`, overridable at runtime with
`JARVIS_PRICING_FILE` pointing at a JSON file.

Logging: `config/logging_setup.py` — INFO for JARVIS, WARNING for
third-party noise, DEBUG everywhere with `JARVIS_DEBUG=1` (which also
turns on the per-step plan trace). `StageTimer` measures real elapsed
time per stage and reports only the stages a request actually reached.

---

## 11. Running it

```bash
python main.py                     # typed mode (also: 'status', 'tasks')
python main.py --voice             # push-to-talk voice
python main.py --tray              # always-on tray assistant
python main.py --status            # config, provider, memory, task state
python main.py --agent "GOAL"      # one agent run, with step-by-step output
python main.py --agent "GOAL" --background
```

## 12. Testing

```bash
python -m pytest                   # the whole suite; no API key needed, no paid calls
python -m pytest tests/test_agent_loop.py tests/test_agent_integration.py
python scripts/test_claude_agent.py --run    # OPTIONAL, real paid Claude call
```

`tests/conftest.py` clears every external credential and pins the voice
providers to their local paths, so a test can never reach a real API or
depend on the developer's `.env`.

## 13. Enabling Claude

1. Put `ANTHROPIC_API_KEY=sk-ant-...` in `.env`.
2. Optionally set `JARVIS_AGENT_MODEL` (default `claude-opus-5`).
3. `python main.py --status` — it should report `active_provider: anthropic`.
4. `python scripts/test_claude_agent.py --run` for a real end-to-end check.

No code change is required at any point.
