# JARVIS — project map for coding agents

JARVIS is a Windows desktop voice assistant. This file exists so a fresh
agent doesn't have to rediscover the architecture (or recreate something
that already exists) from scratch.

## Canonical architecture (source of truth)

Command flow, top to bottom:

- `voice/voice_controller.py` / `voice/background_assistant.py` — voice
  input entry points, call into `brain.agent.run_agent`.
- `brain/agent.py` — top-level orchestrator (`run_agent`). Decides between
  the local fast path and the cloud/task-planner path, using
  `brain/agent_runtime.py`'s `AgentRuntime` for session-level state and
  `brain/executor.py`'s `Executor` for running individual actions.
- `brain/router.py` — `route_command`: tries `brain/intent_router.py`
  (`classify_intent`), `brain/local_intent_model.py` (optional local
  MiniLM-classifier HTTP service on `127.0.0.1:5050`, degrades gracefully
  if that service isn't running), and `brain/local_planner.py`
  (`create_local_plan`, rule-based multi-step splitting) before falling
  back to the cloud.
- `brain/planner.py` — cloud LLM planner (`create_plan`, OpenAI). Used by
  `agent.py` when the local paths can't handle a request.
- `brain/task_planner.py` — the multi-step **task** planner
  (`create_task_plan`, `format_plan`, `segment_sequential_commands`, plan
  completeness checks). This is a distinct concern from `planner.py`
  (single-turn cloud plan) and `local_planner.py` (rule-based local
  split) — all three are actively used and independently tested; they are
  not duplicates of each other despite the similar names.
- `brain/tool_router.py` — `execute_tool`: dispatches a resolved tool call
  to the concrete implementation in `tools/`.
- `tools/` — one module per capability domain (`applications.py`,
  `browser.py`, `desktop_agent.py`, `files.py`, `system.py`, `window.py`,
  `whatsapp.py`, etc.). This is the only place OS/browser/app automation
  should live.

Self-improvement pipeline (autonomous bug-fix attempts), all under
`brain/improvement_*.py`:
`improvement_observer.py` (captures real-execution evidence) →
`improvement_triage.py` (decide if a candidate is worth an automated fix)
→ `improvement_worktree.py` (isolated git worktree, never the main tree)
→ `improvement_repro.py` (before/after reproduction) →
`improvement_coding_agent.py` (`ClaudeCodeAdapter`/`FakeCodingAgent`) →
`improvement_diff_analysis.py` → `improvement_evaluator.py` (gate-driven
accept/reject) → `improvement_orchestrator.py` (`run_attempt`, the single
entry point that wires all of the above together). See the module
docstrings in that file for the hard safety invariants — don't relax them.

Training / intent classification: the **current** pipeline is
`training/train_intent_classifier.py` + `training/evaluate_intent_classifier.py`
+ `training/intent_classifier_common.py` + `training/intent_service.py`
(sentence-transformers embeddings + a joblib classifier, served over HTTP
on port 5050 and consumed by `brain/local_intent_model.py`). This replaced
an older SetFit-based implementation (see git history of commit
`c2954f4`, "Replace SetFit intent model with MiniLM classifier") — if you
see a script that imports `setfit` or `datasets`, or a directory called
`models/intent/` (as opposed to `models/intent_classifier/`), that's the
retired approach; don't resurrect it or build a new parallel one.

`training_data/` (separate from `training/`) is the recorder/schema/
sanitizer/exporter for real-execution trajectories used for future model
training — `recorder.py`, `sanitizer.py`, `schema.py`, `validator.py`,
`exporter.py`. Treat anything under `training_data/` and `training/data/`
as real or potentially real user data: never delete it as "noise" without
explicit confirmation it's reproducible/generated.

## Directories that are data/output, not source

Do not treat these as code to refactor; they're generated or local-only:
`data/` (sqlite runtime DB), `logs/`, `screenshots/`, `models/` (binary
model artifacts — some are tracked deliberately, see `.gitignore`),
`.cache/`, any `__pycache__/`, `.venv*/`, `work/` (scratch), and
`.jarvis-improvement-worktrees/` (throwaway worktrees created by the
self-improvement pipeline — never edit the main tree from inside one).

`training/data/*.jsonl` and `training_data/` contents are the exception:
they look like generated output but are actual dataset/trajectory data —
keep them.

## Tests

Run `python -m pytest` (uses `pytest.ini`, `testpaths = tests`) for the
full regression suite. Tests mock at the tool-execution/subprocess
boundary rather than requiring a real browser or real audio hardware, so
the full suite is safe to run without special gating or flags.

`test_chatterbox.py` at the repo root is deliberately excluded from
`testpaths` — it's a heavyweight, explicit GPU model probe (not a
pytest-style test), run manually by path when debugging the Chatterbox
TTS model directly. It's the reference implementation
`voice/chatterbox_service.py` was built from — keep it.

`scripts/test_*.py` (`test_chatterbox_client.py`, `test_service_post.py`,
`test_bilingual_voice.py`, `test_end_to_end_stt.py`, `test_voice_format.py`)
are manual/opt-in smoke tools (several require an explicit `--run` flag
to actually produce audio), not part of automated regression coverage —
that's intentional, not an oversight.

## Multiple requirements files are intentional

`requirements.txt` (main env), `requirements-agent.txt` (`.venv-agent` —
Playwright/pywinauto/wake-word), and `training/requirements-intent.txt`
(`.venv-intent` — the intent classifier's own isolated environment) each
declare a full, self-sufficient dependency set for a separate virtualenv.
The small overlaps between them (e.g. `psutil`, `sounddevice`) are
deliberate, not duplication to be merged — merging them would break the
isolated-venv setup.

## Legacy areas — don't recreate

- Do not add a second SetFit-based intent classifier; the MiniLM/joblib
  one described above is canonical.
- Do not add a new top-level "planner" or "router" module — extend one of
  the four described above; the names look similar but each has a
  distinct, tested contract.
