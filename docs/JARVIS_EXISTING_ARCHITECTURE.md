---
title: JARVIS Existing Architecture (pre-Obsidian audit)
type: doc
summary: The architecture that existed before the Obsidian knowledge foundation was added, traced end to end from input to speech, with the exact integration points the vault layer hooks into.
tags: [architecture, audit, milestone-0]
updated: 2026-09-02
---

# JARVIS Existing Architecture

## Quick Summary

- Two input surfaces (**voice** and **typed**) already converge on ONE
  function: `brain/agent.py::run_agent`. There was never a "voice JARVIS"
  and a "text JARVIS" — that convergence is what makes a single shared
  memory possible.
- Routing is a cascade of increasingly expensive layers, and it stops at
  the first one that can satisfy the WHOLE request.
- Genuinely agentic work escalates into `brain/agent_service.py::run_agent_task`,
  which builds a **budgeted context** and runs `brain/agent_loop.py::AgentLoop`.
- Memory before this work was SQLite-only: inspectable by code, not by a
  human. That is the gap the Obsidian vault fills.

## Input → answer, traced

```
voice/background_assistant.py  (wake word, STT, ack)   typed: main.py::typed_mode
                 |                                                  |
                 +--------------------+-----------------------------+
                                      v
                       brain/agent.py::run_agent
                                      |
                       brain/router.py::route_command
                                      |
      +--------------+----------------+-----------------+------------------+
      v              v                v                 v                  v
 deterministic   music route     local_planner     task_planner      agent_task
 tool route      (Apple Music)   (rule-based)      (multi-step)      (escalation)
      |              |                |                 |                  |
      +--------------+----------------+-----------------+                  |
                                      v                                    v
                       brain/agent_runtime.py::AgentRuntime   brain/agent_service.py
                       (plan scheduling, waves, locks)          ::run_agent_task
                                      |                                    |
                       brain/executor.py::Executor              brain/agent_loop.py
                                      |                             ::AgentLoop
                       brain/tool_router.py::execute_tool                  |
                                      |                        providers/*.complete
                                    tools/                     (single-turn only)
                                      |                                    |
                                      +------------------+-----------------+
                                                         v
                                voice/text_to_speech.py / ui/ui_bridge.py
```

### Layer by layer

| Concern | Module | Notes |
| --- | --- | --- |
| Entry points | `main.py` (`--start/--tray/--voice/--agent/--status`), `startup/launcher.py` | `--start` is the full desktop bring-up |
| Voice in | `voice/background_assistant.py`, `voice/wake_word.py`, `voice/speech_to_text.py`, `voice/elevenlabs_realtime_stt.py` | one microphone owner, enforced by a named mutex |
| Voice out | `voice/text_to_speech.py`, `voice/tts/*`, `voice/speech_coordinator.py` | English-only TTS policy |
| Orchestration | `brain/agent.py::run_agent` | the single funnel for BOTH surfaces |
| Routing | `brain/router.py`, `brain/intent_router.py`, `brain/local_intent_model.py`, `brain/request_complexity.py`, `brain/music_intent.py` | deterministic first, escalate only when nothing covers the whole request |
| Planning | `brain/local_planner.py`, `brain/task_planner.py`, `brain/planner.py` | three distinct contracts, not duplicates |
| Plan execution | `brain/agent_runtime.py`, `brain/execution_graph.py`, `brain/executor.py`, `brain/recovery.py` | dependency waves, bounded recovery |
| Agent loop | `brain/agent_loop.py`, `brain/agent_service.py`, `brain/context_builder.py` | the loop lives in JARVIS; providers are single-turn |
| Providers | `providers/registry.py`, `providers/anthropic_provider.py`, `providers/openai_compatible.py` | one vendor import per provider module |
| Tools | `brain/tool_catalog.py` (described) + `brain/tool_router.py` (dispatched) + `tools/*` (implemented) | a tool missing from the catalog is invisible to the model |
| Skills (code) | `skills/base.py`, `skills/builtin.py` | hard-coded Python dataclasses |
| Memory | `memory/*` (SQLite), `brain/session_context.py`, `brain/experience_store.py` | not human-inspectable |
| Tasks | `tasks/manager.py`, `brain/task_supervisor.py` | CONCURRENT vs EXCLUSIVE_UI |
| UI | `ui/app.py`, `ui/ui_bridge.py`, `ui/qml/*`, `config/events.py` | event bus, no simulated animation |
| Config | `config/settings.py` | the ONE `.env` load |

## Where the Obsidian layer plugs in (and where it does not)

Chosen integration points, all additive:

1. **`brain/context_builder.py::ContextBuilder.build(extra=...)`** — already
   accepts named extra sections and already budgets them. Vault priming is
   delivered here, so it is truncated and reported by the SAME budget
   machinery as every other section. No new context path.
2. **`brain/agent_service.py::run_agent_task`** — the single place a real
   reasoning task begins and ends. Priming happens before the loop; mission,
   daily-note and lesson updates happen after it.
3. **`brain/tool_catalog.py` + `brain/tool_router.py`** — vault access is
   exposed to the model as ordinary tools, dispatched at the one existing
   dispatch point.
4. **`brain/agent.py::run_agent`** — correction detection observes the user's
   utterance where BOTH surfaces already funnel through.
5. **`config/events.py`** — new assistant states are published on the existing
   bus; the UI already ignores states it does not draw.

Explicitly NOT changed: the router cascade, the deterministic fast paths, the
provider layer, the voice pipeline, the plan scheduler, and the existing
SQLite memory (which keeps working underneath).

## Baseline test state

Recorded in `docs/JARVIS_FOUNDATION_PROGRESS.md`.
