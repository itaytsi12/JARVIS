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
