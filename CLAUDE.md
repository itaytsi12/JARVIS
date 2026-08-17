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

Voice-approved continual learning pipeline (Claude-as-teacher →
voice-approved learning → later batch training), all under
`brain/learning_*.py` + `voice/learning_approval.py`. This is a separate,
downstream concern from the self-improvement pipeline above -- it starts
where that pipeline's `run_attempt` already reached `READY_FOR_REVIEW`, and
never re-implements triage/reproduction/diff-analysis/evaluation:

- `brain/learning_trigger.py` — `evaluate_learning_offer(attempt)`: should
  JARVIS even ask "do you want me to learn how to do that, sir?" Reuses
  `ImprovementAttempt.status`/`.acceptance_gates` rather than re-deriving
  "was this genuinely verified" from scratch, and dedups by a wording-free
  `task_family_fingerprint`.
- `voice/learning_approval.py` — the pure, dependency-injected 30-second
  "yes jarvis"/"no jarvis" state machine (`request_learning_approval`). Its
  real production wiring — `AlwaysOnAssistant.request_learning_approval` in
  `voice/background_assistant.py` — reuses the assistant's single audio
  thread for the approval capture too (a cross-thread `_PendingCapture`
  handoff, never a second microphone consumer).
- `brain/learning_models.py` / `brain/learning_store.py` — the persisted
  `LearningJob` (SQLite, same shape as `brain/improvement_attempt_store.py`).
- `brain/learning_package.py` — deterministic `LearningPackage` extraction
  from a verified attempt (no LLM call, no hidden reasoning — this
  codebase never captures that in the first place).
- `brain/experience_store.py` — immediate, local-retrieval-only experience
  memory, usable before any retraining.
- `brain/learning_variation.py` — bounded, cost-controlled training-family
  variation generation, reusing the SAME `CodingAgent`/worktree machinery
  as the self-improvement pipeline (no second Claude integration).
- `brain/learning_validator.py` — mechanically verifies each variant
  (before-fails/after-passes) before it's trusted; unverified variants
  never enter the dataset.
- `brain/learning_dataset.py` — immutable, versioned dataset manifests.
- `brain/learning_training.py` — `TrainingBackend` protocol,
  `FakeTrainingBackend` (tests/dry-run), `ConfiguredCloudTrainingBackend`
  (never dispatches real training without explicit authorization),
  `ModelRegistry` (the only place `ACTIVE` is ever assigned).
- `brain/learning_evaluation.py` — held-out benchmark evaluation; training
  metrics are never consulted for promotion, only a fresh benchmark run.
- `brain/learning_orchestrator.py` — `handle_verified_teacher_success`
  (approval → job → package → experience) and `start_learning` (the full
  "Hey Jarvis, start learning" batch: dataset build → pre-training checks
  → train → evaluate → promote/reject), both voice-agnostic and fully
  testable with fakes.

A REAL local LoRA/QLoRA training backend and a REAL executable coding
benchmark exist under `training/code_model/` (a separate, dedicated
sub-package — see below). `voice/background_assistant.py::_start_learning_task`
uses them by default in production (`training/code_model/production.py` is
the one place that decision is made) — no `FakeTrainingBackend`/
`FakeBenchmark` in the voice path; those remain test-only.

### `training/code_model/` — the real training/benchmark backend

- `config.py` + `configs/*.yaml` — `CodeModelTrainingConfig` (base model,
  LoRA, quantization, runtime, hyperparameters). Never hardcode a model id
  elsewhere; add/edit a YAML config instead. `small_smoke_test.yaml`
  (`sshleifer/tiny-gpt2`, CPU/GPU-safe, seconds to run) is for tests/dry
  runs only. `qlora_7b.yaml` (`Qwen/Qwen2.5-Coder-7B-Instruct`) is this
  project's recommended real config and production's default.
- `hardware.py` — real GPU/VRAM/RAM/disk/bitsandbytes detection and a real
  parameter-count/VRAM feasibility estimate (fetches the real HF
  `AutoConfig`, no hardcoded per-model table). Verified on this
  deployment's actual hardware (RTX 2060, 6GB VRAM): CUDA and 4-bit
  bitsandbytes quantization both genuinely work here, but `qlora_7b`'s
  ~7.6GB estimated requirement exceeds this card's free VRAM — expect
  `HuggingFaceLoRATrainingBackend.is_available()` to correctly report
  infeasible on this machine until either the config is downsized or
  training is run on a bigger GPU.
- `hf_backend.py` — `HuggingFaceLoRATrainingBackend`, the real
  `TrainingBackend`: genuine `transformers`/`peft`/`accelerate` model load,
  quantization, LoRA injection, tokenization, forward/backward training,
  checkpointing, and a real (weight-level, not bit-exact-optimizer-state)
  resume. Never trains from scratch.
- `dataset_formatting.py` — extends (never modifies) `brain/learning_dataset.py`'s
  JSONL into prompt/response SFT examples.
- `context_packer.py` — compact, evidence-driven multi-file repository
  context (seed files → their local imports → named tests → keyword
  search), used both when formatting training data context and by the
  student adapter at inference time.
- `student_adapter.py` — `LocalCodingModelAdapter`, a real
  `CodingAgent` implementation backed by a loaded HF model (+ optional
  trained LoRA adapter): a bounded inspect → patch → test → revise loop
  against real pytest execution, not chat text.
- `export.py` — real LoRA/QLoRA-adapter → merged-standalone-model export
  (`peft`'s real `merge_and_unload`); GGUF conversion is documented (the
  exact llama.cpp command), not vendored.
- `benchmark/` — `RealCodingBenchmark` (implements
  `brain.learning_evaluation.Benchmark` for real) plus
  `benchmark/fixtures/*/` (five real, executable, hidden-test-verified
  tasks spanning distinct categories — syntax/runtime, logical, cross-file,
  regression, feature-implementation; more categories can be added as new
  fixture directories without any schema change). Reuses
  `brain.improvement_diff_analysis.analyze_diff` and
  `brain.task_supervisor.SafeCommandRunner` — no second diff/subprocess
  implementation.
- `leakage.py` — quarantines any dataset example whose content matches a
  held-out benchmark fixture, via `brain.learning_dataset.build_dataset_version`'s
  `example_filter` hook.
- `production.py` — the ONE place `voice/background_assistant.py` decides
  which backend/benchmark/`TrainingConfig` "Hey Jarvis, start learning"
  actually uses. `JARVIS_CODE_MODEL_CONFIG` (default `"qlora_7b"`) picks
  the config.
- `train.py` / `evaluate.py` / `benchmark/__main__.py` / `export.py` /
  `start_learning.py` — one-command CLIs (`python -m training.code_model.train
  --config ... --dataset ...`, etc.) for debugging without voice/a
  microphone.

Needs `training/requirements-code-model.txt` in its own venv
(`.venv-code-model`, same per-subsystem-venv convention as
`.venv-agent`/`.venv-intent`) — torch/transformers/peft/accelerate/
bitsandbytes are NOT installed in the main JARVIS runtime venv, and tests
that need them (`tests/test_hf_backend.py` and similar) skip themselves
(not fail) when run outside that venv.

## Directories that are data/output, not source

Do not treat these as code to refactor; they're generated or local-only:
`data/` (sqlite runtime DB), `logs/`, `screenshots/`, `models/` (binary
model artifacts — some are tracked deliberately, see `.gitignore`),
`.cache/`, any `__pycache__/`, `.venv*/`, `work/` (scratch), and
`.jarvis-improvement-worktrees/` (throwaway worktrees created by the
self-improvement pipeline — never edit the main tree from inside one).

`training/data/*.jsonl` and `training_data/` contents are the exception:
they look like generated output but are actual dataset/trajectory data —
keep them. The same applies to `data/learning_datasets/` (immutable,
versioned training datasets built by `brain/learning_dataset.py`) and
`data/learning_packages/` (extracted `LearningPackage` JSON, one per
approved learning job) — both are real captured pipeline output, not
disposable cache, even though they live under `data/`.

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

## Voice-approved continual learning: commands

- `"Do you want me to learn how to do that, sir?"` — spoken automatically
  after a Claude teacher fix reaches verified `READY_FOR_REVIEW` (see
  `brain/learning_trigger.py`'s eligibility gate). A 30-second window
  follows; only the exact phrase "yes jarvis" approves and "no jarvis"
  declines (case/punctuation-insensitive) — bare "yes"/"no" do not count,
  and no answer within 30 seconds is treated as a decline.
- `"Hey Jarvis, start learning"` (also "start the learning" / "begin
  learning") — deterministic, matched in `brain/router.py` before any
  planner/intent-model involvement. Gathers every approved job, builds a
  new dataset version, and runs the REAL `HuggingFaceLoRATrainingBackend` +
  REAL `RealCodingBenchmark` (`training/code_model/production.py`). If the
  configured model doesn't fit this machine, it honestly reports that
  rather than pretending to train (`FeasibilityResult.mode ==
  "LOCAL_TRAINING_NOT_FEASIBLE"`, surfaced via `backend.is_available()`).
- `"Hey Jarvis, stop learning"` (also "cancel learning") — cancels an
  in-progress `start_learning` run via the same interactive-task
  cancellation registry `brain/task_supervisor.py` already provides for
  other long voice-triggered actions; the existing plain "cancel"/"stop"
  command cancels it too. Training checkpoints already written are kept
  (never marked `TRAINED`, never promoted).
- `"Hey Jarvis, learning status"` — reports the current run stage (if any),
  how many approved jobs are queued, and the most recent model version's
  status.

CLI/debug equivalents (no microphone needed):
- `python scripts/learning_dry_run.py` — the voice-approval half (real
  `ClaudeCodeAdapter` for exactly one teacher-fix call, fake
  voice/variation/training/benchmark from there), against a disposable
  fixture repo.
- `python scripts/code_model_full_dry_run.py` — the training/benchmark
  half: REAL dataset build → REAL tiny LoRA training
  (`small_smoke_test.yaml`) → real saved adapter → REAL benchmark harness
  (all 5 fixture tasks) → real promotion decision. No fakes for training or
  benchmarking. Needs `.venv-code-model`.
- `python -m training.code_model.train --config <name> --dataset <jsonl>`,
  `.evaluate --model <version>`, `.benchmark --model <version>`,
  `.export --model <version> --output <dir>`,
  `.start_learning --repository-root <path>` (the full pipeline, same as
  the voice command) — see `training/code_model/*.py`.
- Inspect the learning queue: `brain.learning_store.get_learning_job_store().query()` /
  `.query_trainable()`.
- Approve/decline manually: construct a `brain.learning_models.LearningJob`
  and call `LearningJobStore.create`/`.update` directly, or call
  `brain.learning_orchestrator.handle_verified_teacher_success` with a
  fake `request_approval` callable (see `tests/test_learning_orchestrator.py`).
- Start learning manually: call `brain.learning_orchestrator.start_learning(...)`
  directly with whatever `TrainingBackend`/`Benchmark` you want.
- Inspect a job's dataset contribution: `brain.learning_package.load_learning_package(job_id)`.
- Inspect a training run/benchmark result: `brain.learning_training.ModelRegistry.history()`.
- Inspect the active model: `brain.learning_training.get_model_registry().get_active()`.

## Your first REAL model training run

1. Create the environment: `python -m venv .venv-code-model`, then install
   torch from the CUDA index matching your driver (see
   `training/requirements-code-model.txt`'s header), then
   `.venv-code-model\Scripts\python -m pip install -r training/requirements-code-model.txt`.
2. Choose/confirm a base model: `qlora_7b.yaml` (`Qwen/Qwen2.5-Coder-7B-Instruct`)
   is the recommended default; add a new `training/code_model/configs/*.yaml`
   for a different one. `.venv-code-model\Scripts\python -c
   "from training.code_model.hardware import check_feasibility; from
   training.code_model.config import load_config;
   print(check_feasibility(load_config('qlora_7b')))"` tells you honestly
   whether your current machine can run it locally.
3. Use JARVIS normally; approve verified Claude teacher fixes with "yes
   jarvis" as they come up (see the ongoing-use loop below).
4. Say "Hey Jarvis, start learning" (or run
   `python -m training.code_model.start_learning --repository-root <path>`).
   This builds the dataset, and either trains locally (if feasible) or
   tells you it needs external compute plus the exact command/config to
   run there.
5. If local: training checkpoints under `code_model_config.training.output_dir`
   and the final adapter are real, real-time-inspectable via
   `training/code_model/hf_backend.py::load_run_record`.
6. If external/cloud: take the reported config + the built dataset's
   `.jsonl` path to your cloud GPU environment and run the same
   `python -m training.code_model.train --config qlora_7b --dataset <path>`
   command there (install `training/requirements-code-model.txt` there
   too). Copy the resulting adapter directory back.
7. Evaluation and promotion happen automatically at the end of
   `start_learning` (voice or CLI) — the candidate is only promoted if the
   real benchmark shows it's genuinely better; otherwise the active model
   is kept and the candidate is recorded as `REJECTED`, not deleted.
8. Export for local inference: `python -m training.code_model.export
   --model <version> --output <dir>` merges the adapter into a standalone
   model directory; `training.code_model.export.gguf_conversion_command(...)`
   gives the exact llama.cpp command for a GGUF/quantized local build if
   you want one.
9. Configure JARVIS to use the newly ACTIVE model: query
   `brain.learning_training.get_model_registry().get_active()` from
   wherever the real inference call site ends up living (this session ships
   the `CodingAgent`-protocol adapter, `training.code_model.student_adapter.LocalCodingModelAdapter`,
   but does not wire a second, always-on "local student attempts every
   task first" runtime loop into `brain/agent.py` — that's the natural
   next step once a genuinely trained model exists to wire in).

### Ongoing use, going forward

```
Use JARVIS
  -> student/local path fails or has low confidence on a coding task
  -> Claude teaches (existing self-improvement pipeline, unchanged)
  -> independent verification proves the fix works
  -> "Do you want me to learn how to do that, sir?"
  -> "Yes Jarvis"
  -> (repeat over time, across many fixes)
  -> "Hey Jarvis, start learning"
  -> dataset build, training, benchmark evaluation, and promotion
     all happen automatically -- no further code required
```

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
