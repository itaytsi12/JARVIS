---
title: JARVIS Obsidian Foundation - Progress
type: doc
summary: Milestone-by-milestone status of the Obsidian-vault knowledge foundation, with the test baseline, the architecture decisions taken, and the next action.
tags: [progress, foundation, obsidian]
updated: 2026-09-02
---

# JARVIS Obsidian Foundation - Progress

## Quick Summary

- The Obsidian vault is JARVIS's canonical long-term memory. Markdown is
  the source of truth; the JSON index cache is disposable.
- Work is additive: the existing router, planner, voice pipeline, UI and
  SQLite memory all keep working exactly as before.
- If context was compacted: read this file, then `git status` / `git log`,
  then continue from the first milestone not marked DONE.

## Milestone status

| # | Milestone | Status |
| --- | --- | --- |
| 0 | Architecture audit + test baseline | DONE |
| 1 | Build the Obsidian vault + VaultManager | DONE |
| 2 | Enforce the note summary standard | DONE |
| 3 | Vault index + fast two-stage scanner | DONE |
| 4 | Job system | DONE |
| 5 | Skill system | DONE |
| 6 | Mission system | DONE |
| 7 | Priming / knowledge boot | DONE |
| 8 | Automatic learning from corrections | DONE |
| 9 | Memory consolidation | DONE |
| 10 | Daily notes | DONE |
| 11 | Startup memory recovery | DONE |
| 12 | Learn successful methods | DONE |
| 13 | Autonomous execution loop | DONE |
| 14 | Active project memory | DONE |
| 15 | Protected knowledge | DONE |
| 16 | Voice and interface integration | DONE |
| 17 | Obsidian links | DONE |
| 18 | Test suite | DONE |
| - | Critical end-to-end test | DONE |

## Test baseline (Milestone 0)

Recorded before any foundation code was written, in `.venv-agent`.

**The suite could not run at all.** Collection failed for 43 of the test
modules with:

```
brain/context_resolver.py:46: ValueError: could not convert string to float: ''
```

`.env` contains `JARVIS_CONTEXT_APP_TTL_SECONDS=` (present but empty).
`os.getenv(name, default)` returns `""` for a set-but-empty variable, not
the default, so a module-level `float(os.getenv(...))` raised at import
and took `brain.router` -> `brain.agent` -> most of the suite with it.

Three genuine, pre-existing defects were fixed to establish a baseline at
all. All three are general fixes, not per-variable patches:

1. **Empty-tolerant environment readers** (`config/settings.py::env_flag/
   env_int/env_float`), applied to every module-scope numeric env read in
   the repository (13 modules). "Unset" and "set to nothing" now both mean
   "use the default".
2. **`SafeCommandRunner` runs the interpreter JARVIS is running under**
   (`brain/task_supervisor.py::_resolve_interpreter`). Every reproduction,
   regression and benchmark run spelled the command `python -m pytest`,
   which resolved through PATH to an unrelated tool runtime with no pytest
   installed; every such run reported `No module named pytest` and was
   scored as a real test failure. This accounted for roughly 40 of the 51
   failures.
3. **Test isolation now happens at conftest import time**, and clears
   `MEMORY_DB_PATH` instead of overriding it. pytest imports test modules
   during COLLECTION, before any fixture runs, and `brain/agent.py`
   constructs a `MemoryManager` at module scope -- so the fixture was too
   late and the suite had been writing into the user's live
   `data/jarvis_memory.sqlite3`. `memory/memory_manager.py` already had a
   correct pytest-isolation branch; the real `.env` was disabling it.

| Point | Collected | Passed | Failed | Errors |
| --- | --- | --- | --- | --- |
| Before any change | 0 (43 collection errors) | - | - | 43 |
| After the empty-`.env` fix alone | 2016 | 1950 | 51 | 1 |
| After the interpreter + isolation fixes | 2016 | 1986 | 7 | 1 |
| **Final foundation polish** | **2161** | **2139** (+22 skipped) | **0** | **0** |

The 7 that appeared at step three were not regressions: they were tests
that had been making REAL, paid OpenAI calls on the user's own key during
collection, which the new isolation correctly blocked. Each now either
degrades honestly or opts in explicitly through
`tests/conftest.py::allow_cloud_calls`, so "which tests touch a cloud
path" is one grep.

The remaining error is environmental, not a code defect:
`tests/test_multi_model_backend.py` fails with `PermissionError:
[WinError 5]` on this machine's `Temp\pytest-of-Ori` directory. It
predates this work and is unrelated to it.

### One more general fix found along the way

`tests/test_provider_wiring.py` walked the repository with
`PROJECT_ROOT.rglob("*.py")`, which cannot prune, so it descended into
every virtualenv, `.git`, the model artifacts and the caches. One
assertion took **205 seconds** -- longer than the rest of the suite put
together -- and was what made a full run appear to hang. It now uses the
project's own pruning walker (`tools/code.py::walk_source_files`, which
exists for exactly this reason): same answer, **0.37 seconds**.

## Architecture decisions

1. **The vault is a new top-level `vault/` package**, not a subpackage of
   `memory/` or `brain/`. It is imported by both and must not create a
   cycle, and it is a distinct concern from the pre-existing SQLite
   memory (which is untouched and still runs).
2. **Priming is delivered through `ContextBuilder.build(extra=...)`**, the
   existing budgeted-context mechanism, rather than a second context path.
   Vault sections are truncated and reported by the same machinery as
   memories, episodes and conversation.
3. **`brain/agent_service.py::run_agent_task` is the single hook.** It is
   already the one entry point for every request that needs real
   reasoning (verified: the only callers are `brain/agent.py`, `main.py`
   and `scripts/test_claude_agent.py`). Priming happens before the loop;
   the mission, daily note and knowledge updates happen after it.
4. **Deterministic fast paths are never routed through the vault.**
   "Volume down" costs no vault work at all -- `vault/policy.py` decides,
   offline, whether a request is mission-shaped.
5. **Markdown is canonical, the cache is disposable.** The JSON index
   cache lives OUTSIDE the vault so Obsidian never sees it, and is keyed
   on each file's `(mtime, size)`; deleting it loses nothing.
6. **Nothing is ever deleted.** Superseded knowledge is archived with its
   provenance, because the history of a correction is what makes the
   correction trustworthy.
7. **The vault holds the user's personal knowledge, so `data/vault/` is
   git-ignored.** The seed notes live in `vault/bootstrap.py` and a fresh
   clone regenerates the whole structure deterministically.

## Files added

```
vault/__init__.py          public API
vault/note.py              note format, frontmatter, quick summary, sections
vault/paths.py             vault location and folder names
vault/manager.py           atomic read/write/create/move/archive
vault/index.py             NoteSummary, VaultIndex, VAULT_INDEX.md generation
vault/retrieval.py         two-stage scan -> rank -> deep read, with a trace
vault/bootstrap.py         canonical structure + seed notes (idempotent)
vault/jobs.py              Job discovery and selection from notes
vault/skills.py            Skill discovery, loading and method recording
vault/missions.py          persistent Mission records and resume
vault/daily.py             Daily Note creation and incremental append
vault/priming.py           the knowledge boot: mission -> bounded context
vault/learning.py          persistent-vs-one-time correction handling
vault/consolidation.py     duplicate/contradiction refinement
vault/protected.py         protected-rule guard
vault/startup.py           startup memory recovery
vault/policy.py            is this request mission-shaped?
vault/tools.py             the vault tools offered to the agent
vault/cli.py               `python -m vault` diagnostics
```

## Files modified

```
config/settings.py         env_flag/env_int/env_float, vault settings
brain/task_supervisor.py   _resolve_interpreter
brain/agent_service.py     vault priming, mission lifecycle, learning hooks
brain/agent.py             correction observation on the shared funnel
brain/context_builder.py   (unchanged API; used via extra=)
brain/tool_catalog.py      the vault tool definitions
brain/tool_router.py       the vault tool dispatch
config/events.py           the new assistant states
config/logging_setup.py    vault status at startup
voice/background_assistant.py  new states published
startup/launcher.py        vault readiness + startup recovery
tests/conftest.py          import-time isolation, vault redirected
+ 13 modules               empty-tolerant env reads
```

## Measured results

| What | Measurement |
| --- | --- |
| Full suite | 2139 passed, 22 skipped, 0 failed (11m04s) |
| Vault tests | 145 passed (2m09s) |
| Deterministic routing | 0.01-0.10ms per command (unchanged) |
| Priming, 418-note vault, cold | 1,660ms |
| Priming, 418-note vault, warm | ~350ms (one filesystem scan) |
| Scan vs full read | 57 summaries scanned, 5 notes read, 2,332 of 6,000 budgeted chars |
| Vault content vs context | 1,544 of 505,360 characters reached the model |

## Foundation polish (2026-09-02)

- Preferences are split into `preferences/global.md` and one excluded,
  explicitly linked note under `preferences/jobs/` for every Job. A Job's
  rules override conflicting global rules.
- Superseded preferences and notes move to `archive/`, which is excluded
  from ordinary scanning and priming. Existing `user/preferences.md`
  installs migrate automatically and are archived intact.
- Meaningful mission types with no matching Job author a reusable draft
  Job before work begins; trivial commands still create no Job.
- Explicit project names load their project notes, while Daily Notes are
  deep-read only for requests that refer to earlier work.
- Personality/initiative guidance and Git/GitHub workflow knowledge are
  now seeded in the vault.
- Focused vault regression result: **145 passed**.

## Known issues

1. pytest cannot write `.pytest_cache` on this checkout (`WinError 5`), so
   it emits one cache warning. Test execution and temporary test storage
   are unaffected.
2. Ranking is lexical, not semantic. A note describing the same thing in
   entirely different words will not rank. The note format compensates
   (every note carries a hand-written summary and tags), and an embedding
   stage could be added behind the same `VaultRetriever.scan` interface
   without changing anything above it.
3. `memory/memory_manager.py::export_obsidian` is a pre-existing,
   separate, off-by-default one-way export of SQLite entities to
   Markdown. It is unrelated to this vault and was left alone; do not
   confuse the two.

## Next action

Foundation complete. The natural next step is the Clipping Job's Skills;
the Job note, the mission machinery, the long-running loop, the restart
recovery and the morning report are already in place.

## Voice foundation polish (2026-09-02)

- Local speech recognition now stays warm and produces incremental in-memory
  partials from the existing single microphone stream. Stable partials reuse
  the existing deterministic safe-action ledger; the final full transcript is
  authoritative, and partials are never written to training data or the vault.
- Safe stable commands can endpoint after 450 ms of silence. Other requests use
  the configured 750 ms local threshold, preserving natural pauses and the
  existing barge-in state machine.
- `WHISPER_MODEL=` now correctly selects a language-aware default. Controlled
  offline Windows SAPI measurements on the target machine (RTX 2060 6 GB,
  6-core/12-thread AMD CPU, 16 GB RAM) selected `base.en`, int8, beam size 1:

| Cached model | Warm latency range | Controlled-corpus result |
| --- | ---: | --- |
| `tiny.en` | 0.63-0.73 s | Perfect short commands; errors in a location and one long-query word |
| `base.en` | 0.95-1.08 s | Perfect short commands and long query; same synthetic location-name ambiguity |
| `small.en` | 3.85-4.72 s | Perfect short commands and long query; same synthetic location-name ambiguity |

- NVIDIA Parakeet/Nemotron streaming models were investigated but not installed:
  this environment has neither PyTorch nor NeMo, their Python path adds a large
  runtime/model footprint, and available RAM was constrained. Faster-Whisper
  `base.en` is the measured low-risk fit; Whisper remains the fallback when a
  configured ElevenLabs realtime session fails.
- The shared agent system prompt now defines one concise, capable personality
  for every reasoning provider, preserves exact payloads, avoids repetitive
  filler, and uses “sir” occasionally rather than mechanically. TTS provider
  selection remains centralized and independent from reasoning-provider choice.

## Live routing repair (2026-09-03)

- Deterministic routing now canonicalizes sentence-final STT punctuation before
  matching. Volume, site navigation, direct Google/YouTube searches, and simple
  compound open/navigation commands remain entirely local and bypass vault
  priming, Jobs, missions, and reasoning providers.
- Search parsing accepts provider-first and natural provider-last forms and uses
  URL-encoded direct navigation instead of browser UI clicking.
- Job selection is gated by request shape before lexical retrieval. General
  questions and one-shot actions cannot acquire a Job; reusable execution and
  project-modification missions still can.
- Tool schemas are now a hard model capability requirement. Known classifier,
  safety, audio, and provider-orchestration models are excluded from tool use;
  OpenRouter discovery consumes declared supported parameters, modalities, and
  context lengths. A provider-confirmed unsupported-tools response removes and
  caches that capability before falling through to the next compatible route.
- Tests clear every live cloud credential and disable Anthropic fallback before
  collection, so routing regressions cannot spend provider credit.

### Completing the repair: two over-narrow gates and a poisoned temp root

The request-shape gate above was correct in direction and too aggressive in
two places. Both were found by the full suite, and both fixed at the level
they occur rather than by loosening the gate as a whole.

- **The execution-verb list only named CHANGE verbs.** A request that
  produces a deliverable or starts a long-running job matched none of them,
  so "Research the options and write me a summary" -- and, far worse, "run
  the clipping job tonight" -- were forced down to LIGHT and could not
  select a Job at all. That is the flagship overnight mission this whole
  architecture exists for. The list now also covers deliverable verbs
  (research, write, draft, summarise, generate, design, analyse, review,
  ...) and work-starting verbs (run, execute, schedule, investigate, ...).
  This can only ever PREVENT a forced downgrade: the pre-existing scoring
  still decides FULL vs LIGHT, so an extra verb costs nothing while a
  missing one silently loses a Job.
- **A multi-clause request is not a one-shot action.** "Open Apple Music
  and play the album, then verify it is playing" is a mission whichever
  verbs it uses, and the ACTION branch downgraded it purely because "play"
  was not on the verb list. The branch now also consults the existing
  `_MULTI_STEP` detector, so a sequenced request is never treated as
  one-shot.
- **`tmp_path` had never worked on this machine.** pytest reuses
  `<temproot>/pytest-of-<user>`, and that directory exists here with broken
  ACLs -- `os.scandir` and `mkdir` both raise `PermissionError: [WinError 5]`.
  Every test using `tmp_path` errored before its body ran, which silently
  included two of the new model-capability regression tests: the behaviour
  they cover was never actually being verified. `tests/conftest.py` now sets
  pytest's own `PYTEST_DEBUG_TEMPROOT` to a directory it creates, which
  fixes the suite without deleting anything outside the repository and makes
  it robust on any machine with a poisoned temp root.
