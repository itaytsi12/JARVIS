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
  back to the cloud — or, when an agent provider is configured, to the
  agent runtime instead (see the escalation invariant below).
- `brain/request_complexity.py` — the whole-request complexity/coverage
  guard the router consults before every free-form pattern. Not a router
  and not a planner: a pure, offline assessment of whether ANY single
  deterministic action could satisfy the entire request.
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

### Agent runtime (Claude as a provider, not as the architecture)

A second, additive layer sits above the deterministic router for requests
the local layer genuinely cannot resolve. Full detail is in
`docs/AGENT_ARCHITECTURE.md`; the invariants that matter here:

- **Claude is optional.** `brain/agent.py::_agent_escalation_available()`
  returns False whenever no provider is configured, and every
  deterministic route then behaves exactly as it did before. Do not add
  code that assumes a provider exists.
- **The loop lives in JARVIS.** `brain/agent_loop.py::AgentLoop` owns
  plan/act/observe/retry/verify. `providers/base.py::ModelProvider.complete`
  is deliberately SINGLE-TURN. Do not move the loop into a vendor SDK
  helper (the Anthropic SDK's `tool_runner` is deliberately unused).
- **One vendor import.** `providers/anthropic_provider.py` is the only
  module allowed to import `anthropic`. Everything else uses the neutral
  types in `providers/base.py`.
- **One tool dispatch point.** New tools are implemented in `tools/`,
  dispatched in `brain/tool_router.py::execute_tool`, and DESCRIBED in
  `brain/tool_catalog.py::DEFINITIONS`. The catalog adds schemas and
  descriptions; it never becomes a second executor.
- **One `ToolResult`.** The catalog returns the pre-existing
  `brain/models.py::ToolResult`. Never invent a second result type.
- **Success is not verification.** `AgentRun.verified` requires the final
  acting step to have independently confirmed its own outcome. An edit is
  never a fix; only a fresh passing run is.
- **Memory is two things.** `memory/memory_manager.py` (entities/sessions,
  pre-existing, untouched) and `memory/agent_memory.py` (conversation,
  long-term, episodic). They share a process, not a database file.
  `memory/long_term.py::extract_memories` decides what is worth keeping --
  "open YouTube" never becomes a memory; "remember that ..." always does.
- **UI work is serialized.** `tasks/manager.py` runs `TaskKind.CONCURRENT`
  tasks in parallel but only ever one `TaskKind.EXCLUSIVE_UI` task, on
  top of (not instead of) `brain/resource_locks.py`.
- **Escalation is decided once, in the router, before any planner.**
  `brain/request_complexity.py::assess_complexity` is a pure, offline
  whole-request assessment (reasoning required, operation count, software
  domain, local-machine reference, scoping constraint) plus
  `looks_like_simple_target`, the coverage guard for the router's
  open-ended `(.+)` captures. `brain/router.py` consults it before every
  free-form pattern and returns `{"type": "agent_task", "route_source":
  ...}` for a request no single deterministic action can satisfy. Three
  inferred sources exist: `complexity_guard`, `local_context_question`
  (a QUESTION that needs filesystem/terminal tools -- the web-answer
  service cannot see this machine), and `no_deterministic_route` (nothing
  matched; the agent supersedes the gpt-5-mini intent classifier, so that
  cloud call is skipped entirely). `brain/agent.py::_run_agent_impl`
  ALWAYS resolves the route before deciding whether to use the task
  planner -- `should_use_task_planner(command)` is a text heuristic that
  knows nothing about the route, and evaluating it first meant the typed
  entry point (`main.py`, `run_agent(command)` with no route) skipped
  `route_command` altogether. That is what sent "Tell me what files are in
  the JARVIS project folder. Do not modify anything." into the legacy
  planner and back with "I couldn't create a safe local plan for that
  task." Confirmed live.
- **The agent runtime supersedes the legacy cloud planner, never the
  other way round.** Both points in `brain/agent.py` where an incomplete
  or missing local plan used to fall back to `brain/planner.py::create_plan`
  now escalate to the agent runtime first when a provider is available
  (`escalated_from=incomplete_local_plan` / `missing_local_plan` in the
  performance log). The cloud-planner fallback is intact and unchanged
  for the no-provider case.
- **`providers/registry.py::agent_escalation_available()` is the single
  answer to "is the agent reachable".** `brain/router.py` needs it and
  cannot import `brain/agent.py` (circular);
  `brain.agent._agent_escalation_available` delegates to it so the router
  and the runtime can never disagree.
- **Sync Playwright gets its own thread, per session.**
  `tools/playwright_runtime.py`. `sync_playwright().start()` creates an
  asyncio loop and leaves it RUNNING on the calling thread for the life of
  that instance, so a second `start()` on that thread raises "It looks like
  you are using Playwright Sync API inside the asyncio loop" -- the live
  failure on "Open Music." / "Play Israeli playlist.", since JARVIS has two
  independent sync sessions (`tools/browser_agent.py`'s ephemeral browser
  and `tools/browser_authenticated.py`'s CDP attachment) started lazily on
  whichever pooled thread called first. Each session now gets its own
  worker thread (`BROWSER`, `AUTHENTICATED`), keyed off the resource names
  `brain/resource_locks.py::resource_for_tool` already assigns so the two
  do not serialize behind each other. The hop happens exactly ONCE, at the
  dispatch boundary (`brain/tool_router.py::execute_tool` and
  `brain/agent_runtime.py::_browser_action`) -- the session classes must
  never hop themselves, because several of their methods hold a per-session
  RLock across calls to each other and RLock reentrancy is per-thread.
  Exceptions are re-raised on the calling thread unchanged; nothing is
  suppressed.
- **`VOICE_LANGUAGE=he` is bilingual, not "force Hebrew".**
  `voice/voice_language.py::EXPECTED_INPUT_LANGUAGES` maps each mode to the
  languages STT must RECOGNIZE, and `stt_language_code()` forces a language
  code only when a mode expects exactly one (`"en"`). Both providers ask
  that one function -- ElevenLabs sends `language_code` or
  `include_language_detection=true`, Whisper passes the code or `None` --
  so the fallback can never be stricter than the primary. Forcing
  `language="he"` on the Whisper fallback turned ordinary English commands
  into Hebrew transliterations no route could match; JARVIS's own command
  vocabulary is English (the wake phrase, the hard English-only TTS policy,
  `brain/router.py`'s grammar), so a Hebrew-mode user issues English
  commands routinely. `resolve_utterance_language` follows the same rule.
- **Every `AssistantState` must render in the tray.**
  `voice/tray_app.py::state_color` returns a default (and logs) for an
  unmapped state instead of raising: a missing colour is cosmetic and must
  never kill the icon thread. `INTERRUPTED_LISTENING` (barge-in) and
  `WAITING_FOR_LEARNING_APPROVAL` were missing from `STATE_COLORS` and
  raised `KeyError` live while the runtime itself was working correctly.
- **The runtime venv must carry the agent's own dependencies, and a
  missing one must be loud.** `python main.py --tray` runs in
  `.venv-agent`, which had a valid `ANTHROPIC_API_KEY` and
  `JARVIS_AGENT_MODEL` but no `anthropic` package installed. That is not
  an error anywhere: `providers/anthropic_provider.py::is_available()`
  correctly returned False with
  `unavailable_reason="anthropic_sdk_not_installed"`, `get_agent_provider()`
  correctly returned None, and the live tray correctly degraded -- to the
  legacy cloud planner, reporting only "no agent provider is configured"
  while a perfectly good key was loaded. `requirements-agent.txt` now
  declares `anthropic`, and `config/logging_setup.py::log_startup_status`
  (called by BOTH `main.py::main()` and `voice/tray_app.py::run_tray()`,
  once per process) reports `Agent provider configured: yes/no; provider=
  model= api_key_present=` and escalates to **ERROR** whenever a key is
  present but no provider could be initialized, naming the reason and the
  interpreter. `providers/registry.py::agent_unavailable_reason()` gives
  the same reason to `brain/agent.py` so a request-time fallback says WHY.
  Neither ever prints the key.
- **`.env` is loaded once, from the project root, by `config/settings.py`.**
  A bare `load_dotenv()` searches upward from the CURRENT WORKING
  DIRECTORY, so configuration depended on where the process was started:
  launched from anywhere else, `get_config()` cached an empty key and the
  DEFAULT `agent_model`, and `voice/tray_app.py`'s later explicit
  `load_dotenv(PROJECT_ROOT / ".env")` could not undo it (the config is
  cached, and `load_dotenv` never overrides an already-set variable).
  `config/settings.py` now loads `PROJECT_ROOT / ".env"` explicitly, and
  no other module calls `load_dotenv` at all -- `brain/agent.py`,
  `brain/planner.py`, `brain/intent_router.py`, `brain/web_answer.py` and
  `vision/screen_analyzer.py` import `config` for the side effect instead.
  `tests/test_provider_wiring.py` enforces both rules.
### Plan execution: dependency graph, parallelism, result passing

`brain/models.py::Action` has always had `depends_on`, but for a long time
nothing scheduled from it. `AgentRuntime` either ran a plan strictly one
action at a time or -- only when EVERY action was simultaneously
dependency-free and context-independent -- ran the whole plan at once. That
all-or-nothing choice lost on the requests people actually make: "open Chrome
and Spotify, then lower the volume" has one ordering edge at the end, so both
independent launches were serialized for no reason.

- **`brain/execution_graph.py` is the scheduler, and it is a strict
  generalization -- not a replacement.** `build_waves` levels the dependency
  graph (Kahn); `partition_wave` splits one wave into the actions that may
  genuinely run together and those that must not. A pure chain (what
  `task_planner` emits) levels into N single-action waves and takes the
  untouched sequential path in `_execute_plan`, so existing plans behave
  exactly as before. An all-independent plan levels into one all-parallel
  wave -- the old `_execute_plan_parallel` case, which now DELEGATES to
  `_execute_plan_scheduled` rather than being a second engine that could
  drift. Mixed plans, previously forced fully sequential, now overlap what is
  safe to overlap. A cycle raises `CyclicPlanError` rather than silently
  executing an arbitrary subset.
- **Parallel safety is never re-derived.** `partition_wave` reuses
  `brain/safe_tools.py::CONTEXT_INDEPENDENT_TOOLS` and
  `brain/resource_locks.py::resource_for_tool`, and additionally refuses to
  batch an action with a repeat of itself or with a sibling claiming the same
  exclusive resource. `ActionRisk` above SAFE and `optional` actions are never
  parallel candidates. Concurrency is bounded by `JARVIS_MAX_PARALLEL_TOOLS`.
- **`plan_lock_held` is per-thread, and getting it wrong is subtle.**
  `AgentRuntime.execute()` holds the process-wide `action_plan` lock for the
  whole plan on the CALLING thread. Only actions dispatched to worker threads
  pass `plan_lock_held=True`; actions run sequentially within a wave run on
  the calling thread and must keep `plan_lock_held=False` -- both because the
  RLock is already held reentrantly there, and because that flag changes which
  arity `_execute_with_retry` uses to call `_execute_action`, which test
  subclasses override with the historical two-argument signature.
- **Failure semantics are unchanged by scheduling.** A dependency that never
  completed yields `dependency_failure` without executing; an `optional`
  failure is skipped; anything else stops the plan -- but only AFTER the wave
  it happened in finishes, so a sibling already running still returns its own
  result. An exception escaping one action becomes a failed `ToolResult`,
  never a swallowed error and never an aborted sibling.
- **`brain/action_results.py` lets one action consume another's result.**
  `{"__from_result__": {"action": 0, "field": "data.failures"}}` in an
  action's arguments is replaced at execution time by that field of action 0's
  result. It is validated data, never code -- the only thing a reference can
  do is read a field of a result that already exists. `ToolResult` stays the
  SINGLE result type; status/summary/text/artifacts/metadata/`data.*` are a
  VIEW over its existing fields, so any tool that already populates `data`
  gains result passing for free. A reference is itself an ordering
  constraint: `with_reference_dependencies` adds every referenced index to
  `depends_on` before scheduling, so referencing a result orders the two
  actions even if the planner forgot to say so. An unresolvable reference
  fails the action that MADE it (`error="unresolved_reference: ..."`), never
  silently passes nothing.
- **`brain/recovery.py` is bounded by construction, not by a counter.**
  Consulted once per failed action; a strategy may only PROPOSE actions, and
  those run with `allow_recovery=False`, so a failing recovery can never
  generate more recovery. At most `MAX_RECOVERY_ACTIONS` (2), SAFE risk only,
  and `cancelled`/`human_confirmation_required`/`dependency_failure`/
  `resource_timeout`/`unresolved_reference` are never recovered -- those are
  decisions, or they point at a different action. A failed recovery leaves the
  ORIGINAL error standing; it never invents a success. This sits ABOVE the
  tool layer's own fallbacks (`tools/applications.py` already tries VS Code
  aliases, a direct path, `shutil.which`, the start-app command and the app
  index) -- do not duplicate those here.

**Adding a tool that should work with the planner/executor:** implement it in
`tools/`, dispatch it in `brain/tool_router.py::execute_tool`, describe it in
`brain/tool_catalog.py::DEFINITIONS`, and return the existing
`brain/models.py::ToolResult`. Then decide two things: whether it belongs in
`CONTEXT_INDEPENDENT_TOOLS` (only if it neither reads state a sibling writes
nor writes state a sibling reads -- when in doubt, leave it out and it stays
sequential), and whether it needs an entry in
`brain/resource_locks.py::resource_for_tool`. Populate `data` with a
`summary` and any structured payload so later actions can reference it. That
is the whole integration -- the scheduler, result passing, recovery and
dataset capture then apply automatically.

### Multi-target commands and connector-aware dependencies

`brain/task_planner.py` used to chain every clause to its predecessor
(`depends_on=[len(actions)-1]`) whether or not anything required that order,
so no request could ever be scheduled concurrently. Worse, "open Spotify and
VS Code" planned only Spotify: the clause splitter only splits before a
command VERB, "vs code" is not one, and the single-clause path then dropped
every target after the first.

- `segment_with_connectors` reports HOW each clause was joined;
  `segment_sequential_commands` is now written in terms of it and keeps its
  exact previous behavior (including the quoted-payload and
  connector-not-before-a-verb cases).
- `states_an_order` distinguishes explicit sequencing ("then", "and then",
  "after that", "next") from a bare "and"/comma. A clause depends on the
  previous one when the user sequenced it explicitly, OR when its tool is
  outside `CONTEXT_INDEPENDENT_TOOLS` and therefore needs shared desktop state
  (typing, clicking and saving all need the right window in front).
- `split_coordinated_targets` treats a coordinated phrase as several targets
  ONLY when every piece independently resolves to a known app or website, so a
  document called "my report and notes" is never torn in half.

Result: "open chrome and spotify" plans four actions in two waves with both
launches parallel; "open notepad and type hello" still runs strictly in order.

### What the dataset captures about a plan

`PLAN_CREATED` records `index`/`tool`/`arguments`/`depends_on`/`optional`/
`risk` per action (`brain/agent.py::_plan_action_records`) plus the wave and
concurrency breakdown (`_plan_schedule_record`) -- the dependency structure is
exactly what a future model would need and used to be thrown away. Redaction
is unchanged: arguments still go through `_safe_action_arguments` and the
payload still through `training_data/sanitizer.py::privacy_safe_event`.
`plan.context["execution_metrics"]` carries measured `waves`,
`parallel_actions`, `scheduled_ms` and `parallel_saved_ms` (a wave's summed
action durations minus its slowest member -- measured, not modelled), and each
action's own duration lands on its result as `action_ms`. A pure chain records
no scheduler metrics at all, so overhead is zero where there is nothing to
schedule.

### Complex-agent performance (measured, not assumed)

`scripts/benchmark_agent.py` is the harness: a free dry run (tool latency,
context size, tool-schema count, effort) plus `--run` for the real numbers,
saved to `data/benchmarks/agent-<tag>.json` so before/after is a diff rather
than a memory. What it found, and what was done:

- **`tools/code.py`'s tree walk was the single largest cost in a live run.**
  `root.rglob("*")` cannot prune, so `inspect_project` descended into four
  virtualenvs, `.git`, `models/` and the caches, then discarded them: **98.5
  seconds** on this repository, more than every model call put together.
  `walk_source_files` uses `os.walk` with in-place `dirnames` pruning
  (~200ms warm) and `inspect_project` / `search_code` both use it. Same
  answer, one four-hundredth of the work.
- **Prompt caching.** `providers/anthropic_provider.py` sends `system` as a
  content-block list carrying `cache_control: {"type": "ephemeral"}` (which
  caches the tool schemas too -- the prefix renders `tools` -> `system` ->
  `messages`), plus one ROLLING breakpoint on the last message so the
  conversation prefix is cached as the run grows. Measured effect: a task
  that billed 19,000-27,000 input tokens at full price now bills ~110
  uncached with 20,000-94,000 served from cache. Cost per task fell roughly
  40%.
- **Parallel tool execution.** `brain/agent_loop.py::_parallel_safe` runs a
  turn's tool calls concurrently ONLY when every one is read-only, needs no
  exclusive resource, is not a session-aware desktop/browser tool, and is
  not a repeat. Anything else stays strictly sequential.
  `JARVIS_MAX_PARALLEL_TOOLS` bounds it. Honest result: now that the tools
  themselves are fast, the wall-clock saving is small (`parallel_saved_ms`
  in the log says exactly how small); the round-trip saving from batching
  several reads into one model turn is the real win.
- **Observation compaction.** `_compact_list` renders a long list as a count
  plus the first `MAX_LIST_ITEMS`, and says so. `inspect_project`'s
  observation went from 21,222 characters to 1,670 -- and that text is
  re-sent on every subsequent step, so it compounds. The wording deliberately
  tells the model that repeating the same call returns the same summary: an
  earlier version invited a retry and measurably caused redundant steps.
- **Effort.** `brain/agent_service.py::select_effort` sends
  `output_config: {"effort": ...}` -- the configured interactive default
  (`medium`) for read-only inspection, `high` for anything naming work that
  changes something or has to be reasoned out (`_DEMANDING`). It never
  lowers effort for the hard cases.
- **Perceived latency.** `voice/agent_narration.py` speaks rate-limited,
  deduplicated progress derived from real `tool_started`/`tool_result`
  events, with a heartbeat that says the TOOL is slow while one is in
  flight and the MODEL is once they have all returned -- never a guess about
  either. `voice/sentence_stream.py` releases the streamed final answer one
  whole speakable sentence at a time (markdown stripped, code blocks
  dropped, nothing released until enough text has arrived that it cannot be
  a tool preamble). Measured: first speech at 5.6-12.1s instead of nothing
  until the whole answer at 20-93s.
- **Never stream reasoning.** The provider forwards `text` events only, and
  stops the moment a `tool_use` content block starts, so a tool payload or a
  tool-turn preamble can never reach speech. Thinking blocks are not
  subscribed to anywhere.
- **What was NOT optimized, deliberately:** total wall clock is dominated by
  Sonnet's own generation time and varies widely run to run; nothing here
  tries to cut it by skipping verification, dropping evidence the model
  needs, or preventing follow-up steps.

- **Cost is never guessed.** `config/pricing.py` returns `None` for an
  unpriced model and `providers/usage.py` stores `NULL`, so "unknown" and
  "free" stay distinguishable.
- **Tests never spend money.** `tests/conftest.py` clears every external
  credential for the whole suite; the only real Claude call in the repo is
  `scripts/test_claude_agent.py --run`, which is never auto-discovered.

### Overlapping realtime voice pipeline (ElevenLabs)

JARVIS's always-on voice loop (`voice/background_assistant.py`'s
`AlwaysOnAssistant`, still the single owner of the one real microphone
stream -- see `_audio_session`) can use ElevenLabs for BOTH speech-to-text
and text-to-speech, with speech/planning/execution deliberately overlapping
instead of forming one long blocking pipeline. Configured via `.env`:
`STT_PROVIDER=elevenlabs` + `ELEVENLABS_STT_MODEL` (Scribe realtime),
`TTS_PROVIDER=elevenlabs` + `ELEVENLABS_VOICE_ID` + `ELEVENLABS_TTS_MODEL`
(`eleven_flash_v2_5`). Everything here degrades gracefully to the
pre-existing Whisper/pyttsx3-or-cloud TTS chain if ElevenLabs isn't
configured or fails -- neither fallback was removed.

- `voice/elevenlabs_realtime_stt.py` — `ElevenLabsRealtimeSTT`, a
  synchronous websocket client (via `websocket-client`) for ElevenLabs'
  realtime speech-to-text. Never opens a microphone itself -- it's FED PCM
  frames by whichever caller already owns the one real audio stream.
  Delivers `partial_transcript`/`committed_transcript` events through an
  `on_event` callback; `commit()` blocks (bounded) for the final transcript.
- `voice/realtime_capture.py` — `RealtimeSTTController`, the bridge between
  `AlwaysOnAssistant`'s audio-owning thread and one `ElevenLabsRealtimeSTT`
  session per wake/listen interaction: connects on a background thread (so
  a slow network never blocks mic reads), buffers frames captured before
  the connection is ready, and feeds every partial transcript through
  `brain/speculative_execution.py`'s `PartialActionLedger`.
- `brain/speculative_execution.py` — safe speculative execution from
  unstable partial transcripts. `classify_partial_route` only ever
  resolves through the real `brain.router.route_command`, restricted to
  `brain.safe_tools.CONTEXT_INDEPENDENT_TOOLS` (open app/website, volume,
  mute, screenshot -- nothing destructive, communicative, or irreversible).
  `PartialActionLedger` requires 2 consecutive identical partials before
  firing, and `reconcile_final_route`/`reconcile_local_plan_actions`
  prevent the eventual final command from re-running an action a partial
  transcript already started.
- `brain/safe_tools.py` — the single `CONTEXT_INDEPENDENT_TOOLS` allowlist
  shared by `speculative_execution.py` (safe to start early) AND
  `brain/agent_runtime.py`'s Part-H parallel-independent-action execution
  (safe to run concurrently with siblings) -- same underlying safety
  property (self-contained, no shared-state race), one list.
- `brain/agent_runtime.py` — `AgentRuntime._execute_plan` now takes a
  concurrent path (`_execute_plan_parallel`) when EVERY action in a plan
  has empty `depends_on` AND is in `CONTEXT_INDEPENDENT_TOOLS`; any plan
  with real ordering dependencies (e.g. open browser -> navigate -> click)
  is untouched and stays fully sequential. Uses
  `Executor.execute_action_unlocked_plan` to avoid re-acquiring the
  process-wide `action_plan` resource lock from a worker thread (that lock
  is already held for the whole plan by the calling thread; RLock
  reentrancy is per-thread, not per-plan).
- `voice/tts/elevenlabs_tts.py` — streaming ElevenLabs TTS: requests raw
  PCM over `POST /v1/text-to-speech/{voice_id}/stream` and writes each
  chunk straight into a live `sounddevice.OutputStream` as it arrives (no
  temp file, playback starts before the response finishes). Wired into
  `voice/text_to_speech.py`'s existing provider-order/fallback chain
  (`_provider_order`/`_speak_unlocked`/`stop`) alongside the pre-existing
  `openai_tts.py`/`chatterbox_tts.py` providers -- never replaces them.
- `voice/voice_perf.py` — `VoiceInteractionTimer`: per-interaction,
  honestly-measured latency stages (`wake_detected`, `first_partial_transcript`,
  `first_stable_intent`, `acknowledgement_tts_request`, `committed_transcript`,
  `planner_started`/`finished`, `final_tts_started`, `interaction_finished`,
  etc.) with a compact "VOICE PERF" summary logged once per interaction. A
  stage that a given interaction never reached is simply absent, never
  fabricated.
- `voice/startup_validation.py` — logs which STT/TTS providers are active
  at startup (`run_tray()` and `voice_controller.run_voice_loop()`) without
  making a real (paid) API call just to check health.
- `voice/voice_language.py` — the `VOICE_LANGUAGE` mode: `"auto"`
  (recommended default -- English OR Hebrew, detected per utterance),
  `"en"` (forced English input), `"he"` (forced Hebrew input); anything
  else raises `UnsupportedVoiceLanguage` rather than guessing. This is
  STT INPUT only. `get_tts_language()` is a separate, hard-enforced policy
  that ALWAYS returns `"en"` regardless of `VOICE_LANGUAGE` -- `TTS_LANGUAGE`
  is read only so a non-`"en"` value can be reported/logged, it can never
  change actual behavior (Hebrew TTS is out of scope). `detect_input_language(text)`
  is the local, script-based (no LLM), Unicode-range (`֐`-`׿`)
  per-utterance detector; `resolve_utterance_language(text)` is what
  callers actually use -- the forced language in `en`/`he` mode, or
  `detect_input_language(text)` in `auto` mode. Both
  `voice/elevenlabs_realtime_stt.py`'s realtime Scribe session and
  `voice/speech_to_text.py`'s local Whisper fallback read `get_voice_language()`
  for INPUT configuration:
  - Forced `en`/`he`: ElevenLabs sends an explicit `language_code` query
    param (a real, documented ElevenLabs realtime parameter); Whisper
    passes that code to `.transcribe(language=...)`.
  - `auto`: ElevenLabs OMITS `language_code` and instead sends
    `include_language_detection=true` (also a real, documented parameter
    -- confirmed against the live API reference, not guessed) so Scribe
    detects and preserves whichever language was actually spoken. Whisper
    passes `language=None` (its own per-segment auto-detect), never the
    literal string `"auto"` (not a real Whisper language code).
  Whisper's DEFAULT model size is `"small.en"` (English-only, historical
  default) only for forced `en`; `"small"` (multilingual -- the `.en`
  variants literally cannot transcribe Hebrew) for `he` AND `auto`; the
  English `initial_prompt` hint is skipped for anything but forced `en`
  so it can't bias output back toward English; `WHISPER_MODEL` still
  overrides either way. `brain/music_intent.py::classify_music_intent`
  detects Hebrew text itself (any `֐`-`׿` character) and routes
  through a separate, self-contained Hebrew pattern set
  (`_classify_hebrew_intent`) covering the same intents as the English
  cascade -- entities are extracted with the same span-recovery helper
  used for English, so Hebrew Unicode is preserved exactly (never
  translated or transliterated) all the way through to the Apple Music
  search query. `tools/music/apple_music_provider.py::_norm` was fixed
  from an ASCII-only `[^a-z0-9 ]` filter (which silently stripped Hebrew
  titles to nothing before fuzzy-scoring them) to a Unicode-aware `\w`
  filter. `_classify_hebrew_intent`'s song/query extraction is a single
  merged pattern (`(?:נגן|תנגן|שים) (?:לי )?(?:את ה(?:שיר|סינגל) )?(.+)`)
  covering every combination of verb + optional dative filler ("לי") +
  optional "את השיר"/"את הסינגל" wrapper -- the remaining substring is
  captured WHOLE via `(.+)` and never tokenized, so an arbitrary
  multi-word title ("שני משוגעים", "יום חדש", "דרך השלום") always comes
  through intact (confirmed live: an earlier, narrower set of separate
  patterns -- one requiring exactly "נגן"/"תנגן" for the "את השיר" wrapper,
  none stripping "לי" -- meant "שים לי X" incorrectly kept "לי" as part of
  the song, and "שים את השיר X" didn't strip the wrapper at all). Optional
  song+artist qualification ("X של Y" / "X מאת Y") is split by
  `_split_hebrew_song_artist` on the LAST such separator (own dedicated
  known limitation: a song title that itself legitimately contains a
  standalone "של"/"מאת" word would be mis-split -- not distinguishable
  from a real qualifier by text alone). `tools/music/apple_music_provider.py`'s
  `_PLAY_DISPATCH["PLAY_QUERY"]` reuses `_play_song` (not the ambiguous
  `_play_query`) whenever an artist is present, so the artist is used for
  both a better catalog search AND post-playback verification instead of
  being silently dropped. Known limitation: an artist whose Apple Music
  catalog metadata is itself in Latin script (e.g. "Omer Adam") can still
  be mis-ranked against a Hebrew-titled SEARCH query, since local
  fuzzy-text scoring can't bridge a script mismatch the way Apple's own
  search backend does -- Hebrew-titled content (the common case) is
  unaffected and confirmed live to work correctly; separately, the same
  cross-script mismatch on VERIFICATION (comparing the spoken Hebrew
  artist name against the observed Latin-script player-bar text) is
  handled in `_play_and_record`: once the SONG itself is a confirmed exact
  match, an unbridgeable Hebrew-vs-Latin artist mismatch no longer blocks
  `verified` (transliterating to force a match would violate the "never
  transliterate Hebrew" rule; a same-script but genuinely wrong artist
  still correctly fails verification).

  **False-success rule (confirmed live and fixed):** `_play_and_record`
  used to return `success=True` unconditionally -- a search hit, row
  click, or Play-button click succeeding was never itself treated as
  proof the RIGHT track is playing, but the reported `success` field said
  otherwise regardless of the independently-computed `verified` flag,
  meaning the honest "I started the request, but I couldn't confirm the
  track, sir." hedge was already being composed but never actually
  reached the user (background_assistant.py's ack/response architecture
  stays silent on `success=True`). `success` is now `verified` for this
  path specifically -- the same convention every other tool in this
  codebase already follows (`tools/applications.py`, `tools/system.py`'s
  window-state tools) when a claimed outcome can be independently
  confirmed. A short, bounded settle-retry (up to 3 tries, 300ms apart)
  covers a live-observed timing race where the player-bar's marquee
  label can very briefly report stale/inconsistent artist text
  immediately after a fresh play trigger (song text was correct every
  time this was observed; not reliably reproduced on repeated fresh
  triggers, so this is defensive, not a confirmed root cause).
  `music_now_playing` no longer falls back to the locally-remembered
  last-requested song when the live DOM can't be read -- it now honestly
  reports `"I can't tell what's currently playing, sir."` instead,
  since local/requested state can silently disagree with what a human
  changed via another route entirely.

  **Preview-vs-full-playback: root-caused and fixed live (storefront
  mismatch, NOT DRM/CDP).** An earlier investigation pass wrongly
  suspected DRM/EME behaving differently under CDP attachment -- fully
  superseded once the user reported previews happen even on a fully
  manual click inside the SAME dedicated JARVIS Chrome (ruling out
  Playwright/automation as a factor entirely). The REAL, confirmed-live
  root cause: `search()`, `get_recently_played()` (`/listen-now`), and
  `list_library_playlists()` (`/library/all-playlists`) all navigated to
  storefront-LESS URLs (e.g. `music.apple.com/search?term=...`), which
  Apple silently redirects to the `us` storefront regardless of the
  signed-in account's actual region -- while the bare root `/` correctly
  redirects to the account's real storefront (confirmed live: `/il/home`
  for this account). Every search result inherited a `/us/...` href,
  pointing at catalog content the account's subscription has no
  full-playback entitlement for; Apple Music Web serves only a short
  instant-preview clip for that (standard, expected behavior for
  cross-region catalog access) while the player-bar UI still reports
  completely normal "now playing" state -- so song/artist metadata
  matching alone was never sufficient proof this was a full play, which
  is what `playback_type()` below exists to catch independently of the
  root cause. Live evidence ruling out DRM/CDP specifically:
  `navigator.requestMediaKeySystemAccess('com.widevine.alpha', ...)` is
  called by MusicKit JS and succeeds every time (`result: 'granted'`);
  zero DRM/license network traffic occurs for a `/us/` play at all (the
  site chooses a plain unencrypted preview file up front, never even
  attempting a DRM handshake); the account's MusicKit `musicUserToken`/
  `isAuthorized` state is valid. Playing the SAME song via its `/il/...`
  href was confirmed live to switch to a real MSE-backed streaming
  `<audio>` element (`duration: Infinity`, no `AudioPreview` CDN path in
  its `currentSrc`) instead of the static preview file.
  `AppleMusicWebController._resolve_storefront()` now resolves the real
  storefront once (via the root-redirect trick above, cached for the
  controller's lifetime, falling back to `"us"` -- the previous, if
  wrong, hardcoded behavior -- on any resolution failure) and every
  storefront-less URL in this module uses it.

  `AppleMusicWebController.playback_type()` still independently detects
  whether current audio is a short instant-PREVIEW clip (a plain
  `<audio>` element on Apple's `AudioPreview<N>` CDN path, ~90s or a
  `duration` this low) rather than real full-track streaming, kept as a
  defense-in-depth signal (e.g. against a future storefront-resolution
  edge case, or an account genuinely without full-catalog entitlement for
  a specific track) even now that the storefront bug itself is fixed --
  `_play_and_record` downgrades `verified`/`success` to `False` with the
  message `"I could only start a short preview, not the full track,
  sir."` whenever one is detected, rather than silently accepting a
  preview as a successful play.
- `voice/background_assistant.py::_process_capture` accepts an optional
  ElevenLabs committed transcript (skips Whisper entirely when present),
  an optional `PartialActionLedger` (dedupes the final route/plan against
  anything already fired speculatively), and speaks a brief
  "I'll check that, sir." CONCURRENTLY with cloud/task-planner work
  starting (`_command_needs_planning`, reusing `brain.task_planner.should_use_task_planner`
  -- no separate heuristic). `brain/agent.py::run_agent`'s new
  `speculative_ledger` kwarg (threaded through `_execute_recorded_plan`)
  extends the same deduplication to plans built later inside `run_agent`
  itself (local task-plan and cloud-plan branches), not just the router's
  own `local_plan`/`tool` routes.
- `scripts/test_elevenlabs_voice.py` — optional, explicit-`--run`-only
  manual smoke test making REAL, paid ElevenLabs calls (records a short
  phrase, transcribes it, synthesizes and plays one phrase). Never run by
  the automated test suite.

**Language UX rule — TTS speaks English only, always; every command gets
an immediate pre-action acknowledgement that overlaps with execution.**
`voice/background_assistant.py::_process_capture`'s main command path
(`tool`/`local_plan`/`plan`/`ai` routes -- QUESTION-type routes via
`_start_question_task` and WhatsApp send/message/tell/`analyze_screen`
via `_start_cancellable_action_task` are separate dispatch paths, out of
scope here) resolves `input_language = resolve_utterance_language(transcript)`
per utterance, then immediately (BEFORE the blocking `run_agent(...)`
call even starts -- action and acknowledgement genuinely overlap, never
sequential) fires exactly one acknowledgement via `_start_speech_task`:
- English utterance: `voice/response_formatter.py::compose_contextual_ack(route)`
  -- deterministic (no LLM), derived only from the resolved route's
  tool/arguments ("Opening YouTube, sir." / "Playing Starboy, sir." /
  "Okay, playing your Gym playlist, sir." / "I'll check that, sir." for
  `plan`/`ai` routes / "On it, sir." fallback for an unmapped tool). Never
  claims completion ("Opening YouTube, sir.", never "YouTube is open,
  sir.").
- Hebrew utterance: `generic_acknowledgement()` -- a random pick from a
  fixed 4-phrase, always-English, entity-free tuple ("Okay, on it, sir."
  / "Certainly, sir." / "Right away, sir." / "On it, sir."). The action
  itself still receives and uses the exact Hebrew text (e.g. the Apple
  Music search query) -- only the TTS output is restricted; Hebrew is
  never spoken, translated, or transliterated.

After `run_agent` returns (checked via `execution_outcome["success"]`,
which `brain/agent.py::run_agent` always populates): on success, NOTHING
further is spoken for either language (the immediate ack already covered
it -- never speaks twice for one outcome). On failure, exactly one more
message: English reuses `format_spoken_response`'s existing specific,
user-friendly tool failure text (e.g. "I couldn't confirm playback,
sir."); Hebrew always uses the fixed, entity-free `generic_failure_message()`
("I couldn't complete that action, sir.") instead, since the tool's own
failure text could itself contain the recognized Hebrew entity.

**Planner must never override an already-resolved deterministic route.**
`brain/agent.py::_run_agent_impl` used to decide whether to invoke the
task planner from `should_use_task_planner(command)` ALONE, independent
of whether `route_command` had already resolved a deterministic route --
confirmed live that a Hebrew music command could this way reach the
generic (cloud) planner and have it invent its own action (observed:
opening `music.youtube.com` instead of using the already-correct Apple
Music route). `_is_deterministic_music_route(route)` now guards this:
when `route` is already a `{"type": "tool", "tool": "open_music" |
"music_*"}` route (everything `brain/music_intent.py::route_music_command`
returns), the planner is never even consulted, for either language.
`brain/router.py::route_command` also gained a small Hebrew
website-name lookup (`יוטיוב`/`גוגל`/`רדיט`/`גיטהאב`/`טיקטוק`) checked
before its generic `open_application` fallback -- Hebrew "פתח
יוטיוב"/"תפתח יוטיוב" previously fell through to (and failed) trying to
launch a nonexistent desktop app named after the Hebrew site word, since
the existing English website-alias branch only ever matched an ASCII
`"open "`/`"go to "` prefix.

`python -m brain.music_intent "<text>"` is a no-voice-needed diagnostic
(`brain/music_intent.py::_diagnose`) printing detected language,
normalized text, classified intent, extracted entities, resolved
provider, route type, and tool for one input string -- the fastest way to
trace which layer a live Hebrew (or English) routing failure broke at.

**ElevenLabs realtime STT: on_open is not proof of an authenticated
session.** Confirmed live against the real API: the WebSocket handshake
can succeed (`on_open` fires) and a business-logic `auth_error` message
then arrives shortly AFTER, followed by the server closing the
connection -- this reproduced the exact observed symptom of a silent,
unlogged fallback to Whisper with no error anywhere. `connect()` now
waits a brief additional grace window (`ELEVENLABS_STT_AUTH_GRACE_MS`,
default 300ms) after `on_open` for an error/close before declaring the
session genuinely ready. Safe diagnostic logging was added throughout
the STT path (`[STT] configured_provider=...`, `voice_language_mode=...`,
`active_provider=...`, `provider_fallback_reason=...`,
`committed_language=...`, `committed_text=...`) -- never keys/tokens.

### Conversational context, task-vs-media priority, and provider health

A live conversational-context test surfaced six integration gaps between
the deterministic router and real session state. `brain/router.py::route_command`
now takes an optional second `context` (a `SessionContext`) parameter --
the real production callers (`brain/agent.py`, `voice/background_assistant.py`,
`voice/voice_controller.py`) all pass `agent_runtime.context`; every test
caller and `brain/speculative_execution.py` (context-independent tools
only) still call it with one argument, unaffected.

- **A generic explanatory follow-up resolves against real recent
  context, never blind.** `brain/conversational_context.py::resolve_explanatory_followup`
  matches a fixed, narrow set of context-only phrasings ("What does that
  mean?", "Why?", "Explain that.", "What happened?", "Tell me more about
  that.") -- deliberately NOT a bare keyword match, so "why is Chrome using
  so much memory" (its own subject) is untouched -- and resolves them
  against `SessionContext.last_assistant_response` (already the one field
  every route's final answer/result/error text funnels through, per
  `brain/agent.py::run_agent`). Checked in `route_command` before the
  QUESTION classifier ever runs, so it never reaches
  `brain/web_answer.py` blind. Returns a `{"type": "contextual_question",
  "context_text": ...}` route; with no real referent it returns `None` and
  ordinary QUESTION handling continues exactly as before.
  `brain/agent.py::_run_agent_impl` folds the referent into the goal
  explicitly and escalates to the agent runtime (`route_type` ends
  "agent_task" like every other escalation, with `escalated_from=
  "conversational_context"` recording provenance) when a provider is
  configured; with none, it falls back to `brain/web_answer.py` with the
  referent folded into the query -- still not blind, just without the
  agent's own reasoning. `_is_context_resolved_route` stops
  `should_use_task_planner`'s text-only heuristic (which independently
  matches "tell", among others) from re-capturing an already-resolved
  contextual route and inventing a new plan. `voice/background_assistant.py::_is_agent_route`
  treats `contextual_question` like `plan`/`ai` (dispatched off-thread,
  cancellable, narrated, exactly like every other agent turn) when a
  provider is available; when one isn't, a dedicated dispatch branch sends
  it through the same cancellable `_start_question_task` path an ordinary
  `question` route gets, instead of the synchronous default path (which
  would otherwise block the microphone for the whole web-answer call).
- **A browser search correction resolves locally, with zero model
  calls, whenever there is a real search to correct.**
  `brain/conversational_context.py::resolve_browser_search_correction`
  matches "X instead.", "search for X instead.", "try X instead.", "change
  that/it to X." against `SessionContext.browser_active` /
  `last_search_provider` / `last_search_query` (already populated by
  `brain/agent_runtime.py::AgentRuntime._update_context` on every
  `browser_open_url` action, deterministic or agent-driven alike), reusing
  the exact same provider -> search-URL templates `brain/router.py` and
  `brain/local_planner.py` already use for "search youtube for X" so a
  correction always lands on the identical URL shape. Resolves to a single
  `{"type": "local_plan", "actions": [Action("browser_open_url", ...)]}`
  route -- `AgentRuntime.execute` already verifies a `browser_open_url`
  action's URL actually changed, so this correction is verified exactly
  like every other browser action, not just fired-and-forgotten. Excludes
  "send it to X instead" explicitly (that's `revise_whatsapp_recipient`'s
  own pattern) so a coincidentally-active browser session can never
  hijack a WhatsApp recipient correction. An unresolvable provider or a
  genuinely contentless correction ("search for something else") returns
  `None` and falls through to the pre-existing routing, eventually the
  agent runtime for real ambiguity -- never guessed locally.
- **Task stop/pause/cancel outranks an ambiguous media command, but an
  explicit media phrase always wins.** `brain/music_intent.py`'s bare
  `pause` pattern (`pause`/`pause it`/`pause music`... all map to
  `music_pause`) collided with a JARVIS task the user wanted to
  stop -- confirmed live as Whisper transcribing "stop" as the literal
  standalone word "pause". `brain/router.py::route_command` now normalizes
  the utterance through `brain/control_words.py::normalize_control_word`
  (a small, fixed, non-LLM multilingual equivalents table -- Russian
  "Стоп"/"отмена", Hebrew "עצור"/"בטל" -- for exactly the STT
  mis-detections observed live, never a general translation system) before
  the existing unconditional `cancel`/`stop`/`never mind` set (now also
  including `forget it`), so a mis-detected-language "stop" still
  cancels. Bare `pause`/`pause that` is a SEPARATE, narrower check: it
  only resolves to `cancel_read_only_task` (`route_source=
  "task_priority_over_media"`) when `brain/task_supervisor.py::any_active_interactive_work`
  is true (an active read-only/interactive task, an active `tasks/manager.py`
  agent task, or JARVIS is currently speaking -- the last one via
  `brain/activity_state.py`, a narrow one-directional flag
  `voice/background_assistant.py::_set_state` writes on every SPEAKING
  transition and only `brain/task_supervisor.py` reads, since the brain
  layer must never import `voice/*`). An EXPLICIT media phrase ("pause the
  music", "pause Spotify") never matches the bare-word check at all, so it
  always reaches the music route regardless of task state -- confirmed by
  keeping the check on the literal strings `"pause"`/`"pause that"` only,
  never a substring/keyword match. Zero model calls either way: this is a
  dict lookup plus the existing task-status bookkeeping.
- **The speaker resource is preempted, not waited out.**
  `voice/speech_coordinator.py::SpeechCoordinator` sits between every
  speech dispatch site (`voice/background_assistant.py::_start_speech_task`,
  the read-only question path, `voice/agent_narration.py::AgentNarrator`'s
  progress/heartbeat AND its separate `speak_final` for the streamed
  answer) and `voice/text_to_speech.py`'s pre-existing `speak_response`
  resource lock. A live follow-up used to wait out that lock's own 30s
  timeout (`TimeoutError: resource_timeout:speaker`) behind a stale
  progress phrase, most often because a hung ElevenLabs call (see
  provider health below) was still holding it. `SpeechCoordinator.speak()`
  tracks in-flight priorities (`PRIORITY_STATUS` for acks/progress/
  heartbeat, `PRIORITY_FINAL` for a narrated task's real final answer) and
  calls `text_to_speech.stop()` to interrupt a STRICTLY lower-priority
  in-flight utterance before it starts its own -- `>`, never `>=`, so two
  same-priority utterances (an ordinary ack immediately followed by its
  own ordinary result) simply queue for the lock and both play in full,
  exactly as before this existed; only a genuinely stale PROGRESS phrase
  is cut off by the narrated FINAL answer that supersedes it. Deliberately
  does NOT cache `speak`/`stop` as bound callables -- it holds the
  `voice.text_to_speech` MODULE and resolves both as attribute lookups on
  every call, so `unittest.mock.patch.object(text_to_speech, "speak", ...)`
  is honored no matter when it's applied; caching them once (an earlier
  version of this fix) silently kept calling whatever was live the FIRST
  time the process-wide singleton's `speak()` ran, including a `Mock` left
  over from an unrelated, already-finished test -- confirmed live as a
  cross-test failure (`tests/test_barge_in.py`) with no relation to the
  test that actually failed once diagnosed back to its real cause.
- **A definite ElevenLabs quota/funds failure degrades the provider
  once per process, not on every request.** `voice/provider_health.py::ProviderHealth`
  is a per-provider-name flag: `note_result(exc)` marks a provider
  unavailable ONLY for a known non-transient error signature
  (`quota_exceeded`, `insufficient_quota`/`_funds`, `payment_required` --
  matched against the stringified exception, provider-agnostic) and logs
  the reason exactly once; an ordinary transient failure (timeout, 5xx,
  dropped connection) never marks anything, so a real temporary outage
  still retries next time exactly as before this existed.
  `voice/tts/elevenlabs_tts.py::is_available()` and
  `voice/elevenlabs_realtime_stt.py::is_configured()` both check
  `get_provider_health(...).available` before any network attempt --
  every later request in the session then skips straight to the
  configured fallback (Whisper / OpenAI / chatterbox / pyttsx3) with zero
  wasted connect attempts. `voice.provider_health.reset(name)` is the
  manual refresh (e.g. after topping up credits) for the rare case a
  process needs to retry before restarting.
- **Whisper's unconstrained auto-detect is sanity-checked against what
  JARVIS is actually configured to recognize.** Confirmed live: plain
  English commands committed as Dutch, "stop" committed as Russian --
  `language=None` (used whenever more than one language is expected) picks
  from faster-whisper's full detection vocabulary, not just en/he.
  `voice/speech_to_text.py::transcribe_audio` now checks the returned
  `info.language` against `voice_language.py::expected_input_languages`;
  when the detected language is a real, non-empty string outside that set
  (a `MagicMock`-shaped or missing value is deliberately never treated as
  implausible -- that guard is what keeps this inert for a real
  info object with a genuinely unavailable language field, and is also
  what the existing `TranscribeLanguageParamTests` fixture relies on), it
  re-transcribes ONCE, constrained to whichever of the configured
  candidates the mistranscribed text's own script suggests
  (`voice_language.py::detect_input_language`'s existing Hebrew-block
  check -- never a guess toward English by default, so genuine Hebrew
  recognition is not destroyed). A forced single-language mode
  (`VOICE_LANGUAGE=en`) is never second-guessed -- there is no
  unconstrained detection to sanity-check there at all. Still local,
  still free: no cloud LLM call.

All six are covered offline (`tests/test_conversational_context.py`,
`tests/test_task_priority_routing.py`, `tests/test_speech_coordinator.py`,
`tests/test_provider_health.py`, `tests/test_stt_language_sanity.py`) plus
end-to-end sequence tests driving the real `brain.agent.run_agent` against
a scripted fake provider (`tests/test_conversational_sequences.py`, same
pattern as `tests/test_agent_runtime_integration.py`) -- asserting the
provider was actually called with the resolved context embedded, not just
that a route carries the right label. None of this has been re-verified
against real ElevenLabs quota exhaustion or a live Whisper mis-detection
recording; the fixes are built from the confirmed-live symptoms as
reported, and the offline tests reproduce those exact symptoms with fakes.

### Music (Alexa-like Apple Music Web control)

First-class, deterministic music capability -- no LLM for ordinary
requests. Apple Music Web (`https://music.apple.com`) is the default
provider; explicit "on Spotify"/"on YouTube" reuses the existing generic
browser-search-and-click-first-result flow instead of a second bespoke
integration. Apple Music desktop is intentionally NOT used (unreliable
sign-in on this machine).

- `brain/music_intent.py` — `classify_music_intent`/`route_music_command`,
  a local classifier module in the same family as `local_planner.py`/
  `intent_router.py` (called BY `brain/router.py`, never a competing
  router). Regex/keyword based, extracts song/artist/album/playlist/mood/
  provider entities; ambiguous titles ("play Starboy") are deliberately
  left unresolved here and disambiguated by real search-result scoring at
  execution time, not guessed from text. `FAST_PATH_TOOLS` is the single
  source of truth `brain/safe_tools.py::CONTEXT_INDEPENDENT_TOOLS` mirrors
  for which music tools are safe to fire from a partial realtime-STT
  transcript or run concurrently.
- `tools/browser_authenticated.py` — `AuthenticatedBrowserSession`: the
  shared, reusable session manager for any capability that needs a
  browser already signed into the user's REAL accounts. Several earlier
  approaches were tried and rejected, each confirmed LIVE on this machine
  (not just theorized), in order:
  1. Attaching to the user's already-running everyday Chrome directly --
     not technically available; Chrome's profile-singleton lock blocks a
     second automated process from getting a distinct, debuggable handle
     to it.
  2. A JARVIS-owned dedicated persistent Playwright profile
     (`launch_persistent_context`) -- the profile mechanism worked, but
     Apple's interactive sign-in (idmsa.apple.com/appleid.apple.com) hung
     indefinitely after the password step specifically when the
     signing-in browser was Playwright-driven (`navigator.webdriver`,
     `--enable-automation`, a live CDP connection) -- confirmed live with
     the SAME account signing in fine in an ordinary Chrome window using
     that exact profile.
  3. CDP-attaching to the user's REGULAR Chrome profile via
     `--user-data-dir=<the real default %LOCALAPPDATA%\Google\Chrome\User Data>`
     plus `--remote-debugging-port` -- confirmed live that Chrome does
     NOT actually enable the debugger there while the user's normal
     Chrome is running (virtually always, including a background-only
     process many installs keep alive after all windows close): the new
     process just prints "Opening in existing browser session." and exits
     immediately (code 0) without ever applying the debug flags.
  4. A RELATIVE `--user-data-dir` path (even for a brand-new, never-before
     -used directory) -- confirmed live to trigger the exact same
     "Opening in existing browser session." + immediate-exit behavior as
     #3, for reasons not fully understood (Chrome/Windows resolving it
     inconsistently) but reliably reproduced and reliably fixed by always
     resolving to an absolute path (`Path.resolve()`) first.

  What actually works, confirmed live end-to-end: `launch_chrome_for_jarvis`
  (`python -m tools.browser_authenticated --launch`) launches a genuinely
  ordinary, human-driven Chrome window -- no `--enable-automation`, no
  Playwright -- against a DEDICATED, non-default, ABSOLUTE-path profile
  directory (`DEFAULT_AUTH_PROFILE_DIR`, default
  `data/browser_profiles/authenticated_chrome/`, override with
  `JARVIS_AUTH_CHROME_PROFILE_DIR`). That profile starts out signed out of
  everything; sign in manually once, directly in the window `--launch`
  opens (safe to do there -- this launch adds no automation fingerprint).
  JARVIS then attaches over CDP (`chromium.connect_over_cdp(...)`) and
  reuses that session for every future run. `--launch`:
  - Binds the debugger to `127.0.0.1` ONLY, refuses any other host.
  - Refuses to target the user's true default profile directory
    (`default_user_data_dir()`) even if explicitly requested, since #3
    above proved that never works while the user's regular Chrome runs.
  - Refuses if a chrome.exe is already running against that SAME resolved
    profile directory specifically (`_chrome_running_with_profile`,
    inspects each process's own command line -- NOT "is any chrome.exe
    running anywhere," which would wrongly block launches while the
    user's unrelated, different-profile daily browsing is open).
  - Never claims success from `Popen` not raising alone: detects an
    almost-immediate process exit (the "forwarded to an existing session"
    signature) and polls `/json/version` for up to `verify_timeout`
    seconds, reporting a real failure honestly in either case rather than
    a false "launched successfully."
  - Prints the exact, non-secret command line before launching.

  Exposes `ensure_page`/`find_page`/`list_pages` (tab reuse -- never opens
  a duplicate; opens a new tab in the SAME context only when none
  matches), `diagnose()` (`python -m tools.browser_authenticated
  --diagnose`: CDP reachability, context/page counts, each page's
  redacted hostname/title -- no voice/JARVIS needed), and
  `cookie_counts(urls=...)` (safe diagnostic: per-domain COUNTS only,
  never values, and callers should scope `urls` to their own domains
  rather than enumerating the whole real profile). Opt-in
  (`JARVIS_AUTH_BROWSER_DEBUG=1`) `attach_diagnostics` logs URL
  transitions/popups/console/page-errors/failed-request status codes --
  never headers, cookies, tokens, or request/response bodies. Any future
  authenticated-session consumer (WhatsApp Web, Gmail, Calendar, ...)
  should reuse `get_authenticated_browser_session()` and add its tool
  names to `brain/resource_locks.py::AUTHENTICATED_BROWSER_TOOLS` rather
  than inventing a second session/profile mechanism.
- `tools/music/apple_music_browser.py` — `AppleMusicWebController`: Apple
  Music-specific DOM control (search, transport, playlists, queue,
  now-playing parsing) layered on top of whatever page
  `AuthenticatedBrowserSession.ensure_page("music.apple.com", ...)` hands
  back. Owns NO browser lifecycle/profile of its own -- no dedicated
  `user_data_dir`, no `--setup` step; if Apple Music itself isn't signed
  into inside the user's session, `is_signed_in()` reports that honestly
  (`AppleMusicSignInRequired`) rather than JARVIS ever touching the sign-in
  form. Uses semantic ARIA-role/name locators first (mirrors
  `tools/browser_agent.py`'s locator strategy), and exposes
  `diagnose()` (`python -m tools.music.apple_music_browser --diagnose`)
  and `diagnose_auth_state()` (delegates to `cookie_counts`, scoped to
  Apple's own domains only) for tightening a selector or investigating a
  stuck sign-in against the real site, without voice/JARVIS involved.
- `tools/music/apple_music_provider.py` — turns a classified intent into
  real controller calls: search-result scoring (Part 8: never blindly
  click the first result), playlist fuzzy-matching via
  `tools/music/playlist_cache.py` (locally cached playlist names, TTL
  refresh, never credentials), local playback history/resume via
  `brain/music_state.py` (`MusicStateStore`, same SQLite-store shape as
  `brain/experience_store.py`), and honest verification -- a `PLAY_*`
  result is only worded as confirmed when the observed song/artist
  actually match; otherwise the response hedges instead of claiming false
  success. Also records local history from OBSERVED playback (not just
  playback JARVIS itself started) via `_maybe_record_observed_playback`,
  called from the pause/resume/next/previous fast path and
  `music_now_playing` -- so "play the last song I listened to" still
  works even after a manually-started track, deduped so the same
  continuing track is never recorded twice.

  **Real Apple Music Web DOM, confirmed live** (spot-checked against the
  actual site -- re-verify if Apple changes their markup):
  - The page-level hero "PLAY" button is UNRELIABLE: observed live
    `disabled` right after navigating to a playlist despite being fully
    visible, and even when clickable, a row-button click can leave it
    needing a *second*, separate click to actually start playback (see
    below). Every track row instead exposes an accessible button named
    exactly `"Play <title> by <artist>"` (`_ROW_PLAY_NAME` in
    `apple_music_browser.py`) -- confirmed to work in Hebrew-titled rows
    too (the aria-label stays in English). This is the reliable target,
    used by `play_from_current_page` (first row) and
    `play_specific_track(title, artist)` (exact row, for a known song).
  - A row-button click sometimes only SELECTS/LOADS the track (the
    player-bar metadata updates, the hero PLAY button becomes enabled)
    WITHOUT starting playback -- confirmed live on a playlist immediately
    after navigation, while the identical click pattern started playback
    immediately elsewhere (a song's own detail page). Never assume the
    row click alone is enough:
    `AppleMusicWebController._confirm_playing_or_click_hero` waits
    briefly for the Pause state and clicks the (by then enabled) hero
    PLAY button as a follow-up if it never arrives.
  - `document.title` does NOT update to reflect now-playing (confirmed
    live -- it stays the static page title). The real now-playing source
    is the player bar: `[data-testid="marquee-text-item"]` (song, a
    `<span>`) and `[data-testid="marquee-text-item-button"]` (artist, a
    `<button>`) -- `current_track_info`'s primary source; title-parsing
    is kept only as a last-resort fallback.
  - Real catalog song titles routinely carry a `"(feat. X)"` suffix the
    user's spoken request never mentions (Apple's actual title for
    "Starboy" is "Starboy (feat. Daft Punk)") -- a plain fuzzy ratio
    against the bare request scores this only ~0.48, well under any sane
    threshold, and an exact-title ALBUM of the same name used to
    outrank the correct SONG entirely. `_title_matches` (prefix-match
    aware) and `_best_search_match`'s decisive type-preference bonus
    (Part 8) both exist specifically to fix this -- used for BOTH
    result-selection scoring and post-playback verification so the two
    can never disagree.
  - The user's real library playlists live at `/library/all-playlists`,
    NOT `/library/playlists` (which renders only the sidebar's own "All
    Playlists" nav link and nothing else -- an easy, silent trap).
    Apple's own Recently Played shelf lives on `/listen-now` (redirects
    to `/us/home`), NOT the bare root `/`, under a heading literally
    named "Recently Played"; some of its entries are playlist/station
    contexts with `href="#"` (JS-driven, not a real link) and are
    skipped since there's nothing safe to navigate to for them.
  - This is a heavy client-rendered SPA: search results, the library
    playlist list, and the recently-played shelf all render well after
    `domcontentloaded` -- every navigation-driven read waits for a real
    element (`locator(...).wait_for(state="visible")`) rather than a
    fixed sleep, which was confirmed live to sometimes need ~1.5-2s.
- `tools/music/media_keys.py` — Windows SMTC play/pause/next/previous key
  presses (extends `tools/audio.py`'s existing volume-key mechanism). The
  fast path for pause/resume/next/previous/stop: press the media key
  first (near-instant, no browser round-trip), then verify via the
  Playwright page state if a tracked session exists, falling back to a
  direct control click only if the key press isn't confirmed.
- `tools/music/provider.py` — a minimal `MusicProvider` `Protocol` (Part
  24's provider abstraction). Only `apple_music_provider.py` is a full
  implementation; Spotify/YouTube don't get a class here since
  `music_intent.py` already handles them via the generic browser flow.
- `tools/music/diagnose_cli.py` — no-voice-needed diagnostics:
  `python -m tools.music.diagnose_cli <now-playing|playlists|history|recently-played>`,
  `... search "QUERY"`, `... play "SONG" ["ARTIST"]` (the last one
  genuinely starts playback and prints every step -- result selected,
  action used, observed state, verification -- the fastest way to debug
  a live playback issue without going through voice).
- Registered tool names (`music_pause`, `music_resume`, `music_next`,
  `music_previous`, `music_stop`, `open_music`, `music_play`,
  `music_now_playing`, `music_queue_add`, `music_queue_next`,
  `music_shuffle_on`/`_off`, `music_repeat_on`/`_off`,
  `music_add_to_library`, `music_add_to_favorites`, `music_restart_track`,
  `music_artist_more`) live in `brain/tool_router.py`'s dispatch table like
  every other tool, and share a dedicated `"authenticated_browser"`
  resource lock in `brain/resource_locks.py`
  (`AUTHENTICATED_BROWSER_TOOLS`, separate from the generic
  `browser_session` lock `tools/browser_agent.py`'s ephemeral,
  unauthenticated browser uses) so ordinary web browsing and authenticated
  playback never serialize behind each other. Any future authenticated-
  session tool (WhatsApp Web, Gmail, ...) is added to that same set.
- Known limitation: a compound multi-clause command like "open Apple Music
  and play X" still goes through the pre-existing cloud/task-planner path
  (`brain/task_planner.py`, desktop `open_application`), not this module --
  that path is separately tested (`tests/test_plan_validator.py`) and out
  of scope here. Every single-clause Alexa-style request ("play X", "open
  music", "pause", "play my gym playlist", ...) uses the deterministic
  music router instead.

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

### Desktop startup and the graphical interface

JARVIS has a windowed front end and one controlled way to bring the whole
desktop up. Neither replaces anything: `main.py --tray`, `--voice`, the
typed mode and every headless path still work exactly as before.

- **`main.py --start` is the entry point**, and `startup/launcher.py` is
  the sequence. Order: single-instance mutex -> logging -> the Qt window
  (main thread) -> JARVIS's own Chrome and the backend/voice/tray (worker
  threads, dispatched from Qt's `on_started`). The window is created
  BEFORE the slow stages so the core is on screen while the wake-word
  model and Chrome are still loading, not after them. Every stage is
  individually survivable: Chrome failing, the tray failing or the whole
  voice stack failing is logged and the rest still comes up. Only the
  single-instance check stops the sequence, and it exits 0 on purpose.
- **The CLI flags are per-run overrides of settings, not a second source
  of truth.** `config/settings.py` owns `ui_enabled`, `ui_fullscreen`,
  `auto_open_chrome`, `auto_start_voice` and `tray_enabled`;
  `--no-ui`/`--ui`, `--fullscreen`/`--windowed`, `--no-chrome`,
  `--no-voice` and `--no-tray` each override exactly one for one run, and
  a flag that was not typed is `None` and changes nothing. Those flags use
  their own `start_*` argparse dests: `--no-voice` and `--no-tray`
  originally reused the dests of the PRE-EXISTING `--voice` and `--tray`
  flags, argparse allows that silently, and a bare `--start` then
  inherited `--voice`'s `store_true` default of False -- JARVIS came up
  with no voice and no tray while the log reported it as configured.
  Confirmed live; `tests/test_startup_launcher.py::CommandLineTests` is
  the regression.
- **One mutex, one microphone.** `voice/single_instance.py`'s Windows
  named mutex, under the name `startup/launcher.py::MUTEX_NAME` shares
  with `voice/tray_app.py::run_tray`. If those names ever diverged,
  `--start` and `--tray` would each think it was the only instance and two
  processes would fight over the one real audio stream. A second launch
  prints, logs, and exits 0 having opened no window, started no backend
  and touched no browser.
- **One assistant, and the tray owns its lifecycle.** `TrayApplication`
  takes `assistant=` and `on_exit=` so the window and the tray observe the
  SAME `AlwaysOnAssistant`. `TrayApplication.run()` starts and stops it in
  its own `finally`, so the launcher must NOT also start it -- that would
  start the single microphone owner twice. With `--no-tray` the launcher
  starts it directly instead. The tray's Exit closes the window through
  `UiBridge.run_on_gui_thread` (Qt may only be touched on the GUI thread,
  and Exit runs on the tray's).
- **JARVIS's Chrome is identified specifically, never as "chrome.exe is
  running".** `startup/chrome.py` reuses `tools/browser_authenticated.py`
  -- `resolved_auth_profile_dir` (one resolver shared by detection AND
  launching, so the directory inspected can never drift from the one a
  launch would use), a reachable CDP endpoint on `127.0.0.1:9222`, and
  `jarvis_chrome_is_running`, which matches each chrome.exe's own
  `--user-data-dir` argument. Confirmed live with 13 of the user's
  personal chrome.exe processes running: both JARVIS-specific indicators
  correctly read false, JARVIS's own Chrome was launched, and a later
  `ensure_jarvis_chrome` then reported `already_running_debuggable` and
  launched nothing.
- **Windows auto-start is one per-user scheduled task.**
  `scripts/autostart.py` registers `main.py --start` under `pythonw.exe`
  from `.venv-agent`, at logon, `LeastPrivilege` + `InteractiveToken` (no
  administrator rights), with a 10-second logon delay that is a real Task
  Scheduler trigger delay and NOT a sleep inside Python. Every path --
  interpreter, project directory, user name -- is resolved at install
  time. Install with `python scripts/install_jarvis_autostart.py`, remove
  with `python scripts/remove_jarvis_autostart.py`; the tray's "Start with
  Windows" item toggles the same task.
- **A windowed run has no console, so the log file goes in FIRST.**
  `config/logging_setup.py::configure_file_logging` (the implementation
  `voice/tray_app.py::configure_logging` used to own, moved so both use
  one file) installs `logs/jarvis_background.log` before
  `configure_logging()` runs -- which also stops it attaching a
  `StreamHandler` to a `sys.stderr` that is `None` under `pythonw.exe`.

#### `config/events.py` -- how backend state reaches the window

The bus lives in `config` because every layer already depends on it and
publishing from `providers`/`brain`/`voice` must not create an import
cycle -- the same one-directional reasoning as `brain/activity_state.py`,
generalized from one boolean to named events. A subscriber can never break
a publisher (every callback runs in its own try/except) and publishing
with nothing subscribed is a dict lookup, so the CLI, the tests and
`--no-ui` pay nothing.

Real publishers, all at genuine call sites: `providers/anthropic_provider.py`
(which publishes its TRANSLATED error type, so the UI keys the amber
rate-limit state on `ProviderRateLimited` rather than on whichever vendor
class happened to raise), `brain/planner.py`, `brain/intent_router.py`,
`brain/web_answer.py` and `vision/screen_analyzer.py` via
`events.model_activity(...)`, `brain/local_intent_model.py`, and
`voice/background_assistant.py` for assistant state, transcripts and
replies. There are no simulated animations anywhere in `ui/`.

- `ui/model_status.py` answers "which model modules does this install
  actually have" from the same sources the runtime uses
  (`providers/registry.py`, the OpenAI credential, the local intent
  service, the promoted local model registry). **Gemini is deliberately
  never reported available** -- nothing in this repository implements it;
  the node exists and says `not_implemented`. The header count is derived
  from this, never a hard-coded "5 MODELS ACTIVE", and it reads
  "DETECTING MODULES" until the first probe lands rather than claiming
  zero.
- `ui/ui_bridge.py` is the ONE object QML binds to. Every public setter
  marshals onto the GUI thread through a `Signal(object)` queued
  connection, so the audio thread, an agent worker and a provider call can
  all publish without touching Qt. It also understands the vendor-neutral
  `model_thinking`/`model_active`/`model_error`/`model_rate_limited`
  family declared in `config/events.py` for the multi-provider router
  work, resolving the node id from several plausible payload keys and
  ignoring quietly (debug, never a warning) anything naming a capability
  or provider this window does not draw.
- An unavailable module never lights up, not even red: the optional local
  intent service reporting "not running" on every command is its normal
  state, not an alarm. `rate_limited` is amber and deliberately distinct
  from the red `error` -- a throttled module is configured and working.
- `ui/app.py` owns the Qt application and nothing else, and
  `is_available()` reports a missing PySide6 as a normal state --
  `startup/launcher.py` then logs it and brings JARVIS up without a window
  rather than failing. `ui/qml/main.qml` plus `components/` (`CoreRing`,
  `ModelNode`, `ConnectionLine`) draw the core; Escape always leaves
  fullscreen and F11 toggles it, so you are never trapped. Every HUD
  detail is painted into a Canvas once on resize and animated with
  transforms/opacity on the render thread, so the window costs one texture
  upload at startup and no repaints afterwards.

### The tool catalog is the agent's entire view of JARVIS

`brain/tool_catalog.py::DEFINITIONS` is not documentation -- it is the
list of tools the model is actually given. A tool that
`brain/tool_router.py` can dispatch but the catalog does not describe is
**invisible**: it can never be called, however well it works.

That was not hypothetical. The whole Apple Music family -- 18 implemented,
tested, dispatchable tools -- had no catalog entry, so a request like
"open Apple Music and make me a playlist" reached a model holding no music
tools at all and could not possibly succeed. Describing them was the
entire fix. The catalog went from 49 tools to 86.

- **`tests/test_tool_catalog_coverage.py` is the invariant.** It parses
  the router's own dispatch table and fails if anything is dispatchable
  but undescribed, or described but duplicated. Two tools are excluded on
  purpose and listed in `INTENTIONALLY_UNDESCRIBED`: `lock_computer`
  (locking the machine is something the user asks for explicitly, never a
  step an agent should choose) and `send_whatsapp_message` (irreversible
  and outward-facing; it keeps the voice path that asks first). Both are
  still reachable by their own routes -- the catalog controls what the
  AGENT is offered, not what JARVIS can do.
- **`ToolDefinition` now carries `retry_safe` and `timeout_seconds`.**
  `retry_safe` answers "is running this twice the same as running it
  once": every read-only tool is (normalised in `__post_init__`, so no
  call site has to remember), and so are the idempotent writers
  (`create_directory`, `set_volume`, `write_clipboard`, `scroll_screen`).
  `append_text_file`, `volume_up` and `volume_down` are deliberately NOT
  -- a retry duplicates the text or moves the volume twice.
  `timeout_seconds` is advisory: it tells the model a slow tool is
  expected rather than hung, and it is deliberately not a hard kill,
  because interrupting a half-finished desktop action is worse than
  waiting for it.

### Windows tools the agent was missing

New modules, all dispatched in `brain/tool_router.py` and described in the
catalog like everything else:

- **`tools/clipboard.py`** -- `read_clipboard` / `write_clipboard`. The
  cheapest bridge to an application JARVIS cannot script, and the
  read-back half of a verification. The clipboard is a single-owner OS
  resource, so every operation retries briefly: another process holding
  it for a moment is ordinary (Chrome and Office both do it) and must not
  surface as a failure. A write is verified by reading it back, so a
  clipboard manager winning the race is reported honestly rather than as
  success.
- **`tools/machine.py`** -- `system_status` (CPU, memory, disk and battery
  in ONE call, so "how is this machine doing" costs one model turn, not
  four), `network_status` (a real TCP connect, because an adapter can be
  "up" behind a captive portal), `list_processes`, `process_running`
  (matches the way a person names an app: "chrome", "Chrome",
  "chrome.exe" and "Google Chrome" all resolve, because the agent gets
  this argument from a spoken request), `get_volume` and `set_volume`.
  Volume goes through the real Windows `IAudioEndpointVolume` COM
  interface, declared inline via `comtypes` (already installed by
  `pywinauto`; `pycaw` is not, and adding a package for two methods was
  not worth it). Media keys cannot express "set it to 30%", move in
  device-defined steps, and cannot report the current level at all --
  `tools/audio.py`'s relative controls remain untouched as the fallback.
  `set_volume` reads the level back, so a device that quantizes is
  reported honestly instead of being claimed.
- **`tools/files.py`** gained `file_info` (size, kind, age in one call)
  and `recent_files` -- what actually answers "find the file I worked on
  yesterday". It walks with `os.walk` and prunes using `tools/code.py`'s
  existing ignore vocabulary rather than a second one, for the same
  reason `walk_source_files` exists: descending into a virtualenv to
  discard the results costs orders of magnitude more than the answer.
- **`tools/ui.py`** gained `scroll_screen`, the desktop counterpart to
  `browser_scroll` (which only ever worked inside a Playwright page).
  Windows delivers wheel input to whatever is under the CURSOR, not to
  the focused window, which is why it takes optional coordinates. It
  reports the action and explicitly does NOT claim the content moved --
  verifying that needs a screenshot, which is the caller's decision.

### JARVIS's Chrome starts itself

`tools/browser_authenticated.py::AuthenticatedBrowserSession._autostart_chrome`
starts JARVIS's own Chrome when nothing is listening on the debug port,
once per session object, delegating to
`startup/chrome.py::ensure_jarvis_chrome` so the "is it already running"
decision stays in the one place that owns it. Every authenticated-session
tool used to fail with "Start the JARVIS browser session first" -- a
correct diagnosis and a useless one, since JARVIS is perfectly able to do
it. `JARVIS_BROWSER_AUTOSTART=0` restores the old behaviour. It still only
ever attaches to a real debuggable Chrome and still never touches the
user's personal profile.

### Apple Music: playlists, and two real search bugs

`music_create_playlist`, `music_add_to_playlist` and
`music_list_playlists` complete the Apple Music surface. All three were
built against the REAL signed-in account and the real DOM, and three
separate defects were found by doing so:

- **The row context menu is `role="button"`, not `role="menuitem"`**, and
  its Add-to-Playlist submenu is in the DOM but not clickable until the
  parent entry is activated. Querying `menuitem` found nothing and timed
  out. `_menu_entry` tries button first and keeps menuitem as a fallback.
- **"New Playlist" does not navigate.** It reveals an inline field
  (`data-testid="playlist-title-input"`) on the same page, and committing
  that creates the playlist with the track already in it -- so the name is
  set AT creation, not by a rename afterwards, which is both fewer steps
  and impossible to leave half-done as a stray "New Playlist".
- **`search()` returned the PREVIOUS query's results.** This is one
  long-lived tab, so the old query's links are still in the DOM the
  instant the new URL commits; waiting for "a result link is visible"
  returned immediately and the caller scored the old page. A search for
  "Save Your Tears" returned Khalid and twenty one pilots, and
  `_best_search_match` duly picked "Stressed Out" -- a wrong song chosen
  with complete confidence. `_wait_for_search_results` now settles on the
  new query in two observed steps: the SPA's own search box reflecting the
  new term, then the result count going stable.
- **The type bonus could carry an unrelated title over the threshold.**
  Even with correct results, `_fuzzy("Save Your Tears", "Stressed Out")`
  is 0.44, and the 0.35 "is a song" bonus clears `min_score`. `TITLE_FLOOR`
  requires the title itself to be plausible, bypassed whenever
  `_title_matches` already accepts it (so "Starboy" ->
  "Starboy (feat. Daft Punk)" still works) and disabled for genuinely
  open-ended requests (`_play_query`, `_play_mood`), which name no title.

Playlist authoring applies one gate playback does not: `_locate_song`
refuses a match that fails `_title_matches`. Playing the wrong song is
audible and correctable; adding the wrong song to a playlist is a silent,
persistent edit to the user's library. Verification is a real re-read of
the playlist, and it checks for the song the USER asked for -- checking
for the matched title would be trivially true and prove nothing.
`_playlist_exists` retries briefly, because Apple's library listing is
eventually consistent and a single eager read reported a genuinely
successful creation as `verification_failed` (confirmed live).

`is_signed_in` now waits for the app shell to hydrate. Apple renders a
"Sign In" control in its initial HTML and swaps it for the account
affordance once the session loads, so checking immediately after opening a
fresh tab reported "not signed in" for a perfectly signed-in account.

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
