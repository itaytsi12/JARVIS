---
title: The Obsidian Vault Architecture
type: doc
summary: How JARVIS uses an Obsidian vault as its persistent long-term brain - the note standard, two-stage retrieval, Jobs, Skills, Missions, learning from corrections, and Daily Notes.
tags: [architecture, obsidian, vault, memory]
updated: 2026-09-02
---

# The Obsidian Vault Architecture

## Quick Summary

- **The vault is the brain.** Markdown notes on disk are JARVIS's
  persistent memory; the model's context window is temporary working
  memory. The model never has to remember anything itself.
- **Obsidian does not need to be running.** JARVIS reads and writes the
  files directly. Obsidian is the human interface to the same files.
- **Nothing is loaded blindly.** Every note states its purpose in one
  sentence, so JARVIS triages hundreds of notes without reading them and
  deep-reads only the handful that matter.
- **It is additive.** The router, the planners, the voice pipeline, the
  UI and the pre-existing SQLite memory all work exactly as before. A
  vault failure degrades memory, never the assistant.

## Where it is

```
data/vault/                     (override with JARVIS_VAULT_PATH)
```

Open that folder in Obsidian ("Open folder as vault") and everything
below is browsable and editable. The path may be absolute, so pointing
JARVIS at an Obsidian vault you already have is one setting.

```
data/vault/
  VAULT_INDEX.md            generated map of every note
  identity/
    jarvis.md               who JARVIS is
    core_rules.md           the standing operating rules
  user/
    profile.md              durable facts about the user
    preferences.md          standing instructions
  jobs/                     recurring KINDS of mission
    INDEX.md
    fix-software-bug.md
    answer-about-this-machine.md
    clipping.md             placeholder, proves the shape
  skills/                   reusable knowledge about HOW
    INDEX.md
    code-inspection.md
    python-debugging.md
    test-verification.md
    windows-desktop-control.md
  projects/                 knowledge about one body of work
    INDEX.md
    jarvis.md
  lessons/                  things discovered by experience
    INDEX.md
  missions/
    active/                 written before work starts
    completed/              kept forever, inspectable
  daily/
    2026-09-02.md           chronological record of the day
  state/
    current.md              what JARVIS is working on now
  system/
    protected_rules.md      never edited automatically
```

A JSON index cache lives at `data/vault_index_cache.json` -- OUTSIDE the
vault, so Obsidian never sees it. Deleting it loses nothing; Markdown is
the source of truth.

## The note standard

Every substantial note begins with machine-readable metadata and a human
Quick Summary, so its purpose is knowable without reading it:

```markdown
---
title: Apple Music Control
type: skill
summary: How JARVIS opens, reuses, searches and controls Apple Music.
tags:
  - music
  - apple-music
updated: 2026-09-02T14:03:11+00:00
---

# Apple Music Control

## Quick Summary

- Reuse the existing window if one is already open.
- Launch only when no instance exists.

## Procedure
...
```

`vault/note.py` owns this format. Parsing is tolerant -- a note a human
broke degrades to "a note with no metadata" and never stops a scan --
while writing always goes through one serializer, so a note JARVIS wrote
round-trips byte for byte and produces small diffs. `build_note_text` is
the only creation path and requires a summary, which is why "every note
has a summary" is a guarantee rather than a hope.

## Two-stage retrieval

```
   Mission: "Fix the Apple Music playlist problem."

   STAGE 1 -- SCAN (offline, no model call, no bodies read)
     Fix Apple Music Playlist   27.80  title, summary, tag, quick-summary
     Apple Music Control        16.00  title, summary, tag, quick-summary
     Fix Software Bug            7.00  title, summary
     Video Editing                  -  no signal
     ...414 more summaries seen and rejected

   STAGE 2 -- DEEP READ (bounded by a character budget)
     identity/core_rules.md
     jobs/fix-apple-music-playlist.md
     skills/apple-music-control.md
     user/preferences.md
```

Measured on a 418-note vault: 1,544 characters entered the model's
context out of 505,360 characters of note content. Ranking is
deterministic lexical scoring over the fields the format guarantees --
no embedding model, no network, no cost.

**Structural bonuses may only reorder, never qualify.** A note of the
requested type, or one touched recently, gets a small boost only if it
already matched the request topically. Before that rule existed, "is a
skill" (0.75) plus "touched today" (0.4) cleared a 1.0 threshold on its
own and loaded a video-editing Skill into an Apple Music mission.

**Missions and daily notes are excluded from the knowledge scan.** They
are records of past work, not knowledge about how to do it, and a
completed mission note is nearly a verbatim copy of the request -- it
scored 48.6 against the correct Job's 22.0 on a repeat. They stay fully
searchable through `vault_search`.

Every selection is traced: `RetrievalTrace.explain()` prints what was
considered, what was chosen and why. `python -m vault scan "..."` and
`python -m vault prime "..."` print it for any request.

## Jobs

A Job is a recurring KIND of mission, stored as a note. There is no list
of Jobs anywhere in the Python source, deliberately -- dropping

```
data/vault/jobs/write-sales-email.md
```

into the vault makes that Job selectable on the next request. Its
`summary` and `When To Use` section are what decide whether it applies.
A Job declares its Skills as wikilinks, which is how they get loaded:

```markdown
## Required Skills

- [[Python Debugging]]
- [[Test Verification]]
```

A Job with `status: placeholder` is visible but never selected
automatically -- that is how `jobs/clipping.md` describes work that is
not built yet without JARVIS confidently attempting it.

## Skills

A Skill is reusable knowledge about HOW to do something. One Skill serves
many Jobs. Its most valuable section is `Known Working Method`, which is
where "approach A failed, approach C works" is recorded:

```markdown
## Known Working Method

- **For this kind of work, `music_now_playing` produced the verified result.**
  - Confirmed working 2026-09-02 (mission 97c7fb469e36).
  - Does NOT work: `click_ui_element` failed here first
```

That section is rendered FIRST in the model's prompt, before the general
procedure: it is the expensive-to-rediscover part, and putting it last is
how it ends up being what a budget truncation drops.

This is a knowledge layer, separate from the pre-existing code-level
`skills/` package, which selects TOOLSETS. Both are used and neither
replaces the other: the code skill decides which tools the model is
offered, the vault Skill decides what it is told about using them. Only
the vault Skill is editable at runtime by a correction.

## Missions

A Mission is one concrete piece of work, written to `missions/active/`
BEFORE the work starts and appended to as it happens -- never batched to
the end, because a crash must not erase it. It records the request, the
knowledge that was loaded, the plan, progress, failures, discoveries,
artifacts and the outcome, then moves to `missions/completed/`.

Any mission still sitting in `active/` when a new process starts is by
definition one whose owner is gone, so startup marks it `interrupted`
rather than guessing an outcome for it, and `resumable()` offers it back.

Only mission-shaped requests get one. `vault/policy.py` decides, offline,
whether a request is substantial; "volume down" gets light priming
(identity and preferences, ~1.5KB) and no mission at all.

## Learning from corrections

```
User: "No. When Apple Music is already open, don't open another one.
       Use the existing window."
```

`vault/learning.py` classifies this as PERSISTENT (it states the
situation the rule applies in), finds the note that governed the
behaviour, rewrites it as a clean imperative rule -- never pasting the
user's sentence -- writes it into the right section, updates the Quick
Summary, refreshes `updated`, and records it in the Daily Note.

```diff
  ## Procedure

- 1. Launch Apple Music.
- 2. Search for the track or album.
+ 1. Check whether Apple Music is already running.
+ 2. If running, focus the existing instance.
+ 3. Launch a new instance only when none exists.
```

"Make this answer shorter" writes nothing. The distinction is made from
scope markers, and immediacy wins ties on purpose: wrongly writing a
standing rule is worse than wrongly treating one request as local.

**Consolidation** (`vault/consolidation.py`) stops the vault becoming a
pile of contradictions. A new rule is compared against what is there and
either duplicated (nothing happens), refined ("keep responses short" plus
"when coding, give detail" becomes one scoped rule), superseded (the old
rule moves to `## Superseded`, dated), or added. Nothing is ever deleted.

**Protected knowledge** (`vault/protected.py`) is what automation may
never touch: everything under `system/`, plus any correction that would
weaken a confirmation step, a credential rule, a financial guard or a
destructive-action guard -- wherever it appears. A refusal is recorded
and told to the user, never silent.

## Daily Notes

One note per day, appended to throughout it. Timeline entries carry what
was asked, what JARVIS did, what came of it, which files changed and what
was learned; separate sections collect decisions, corrections, problems,
working methods, projects touched, artifacts, unfinished work and
suggested next actions. The Quick Summary is regenerated from the note's
own content after every event, because it is what a LATER session reads
first and it has to be true.

Credentials are redacted before any write.

## Startup memory recovery

`startup/launcher.py` calls `vault/startup.py::recover_session` before
anything else starts. It reads SUMMARIES -- identity, preferences, the
active project, unfinished missions, today's and the most recent previous
day's note -- so a first request like "carry on with what we were doing
yesterday" resolves against something real. "The most recent previous day
with a note" rather than literally yesterday, because JARVIS is not used
every day.

## How it plugs into the existing JARVIS

Two hooks, both on paths the voice and typed surfaces already share:

| Where | What happens |
| --- | --- |
| `brain/agent_service.py::run_agent_task` | Priming before the loop; mission, learning and Daily Note after it |
| `brain/agent.py::run_agent` | Corrections observed on the single funnel both surfaces use |
| `brain/context_builder.py` | Vault knowledge is delivered via the EXISTING `extra=` sections, so it is budgeted and reported like everything else |
| `brain/tool_catalog.py` + `tool_router.py` | Eight vault tools, described and dispatched at the one existing point |
| `config/events.py` + `ui/ui_bridge.py` | `SCANNING_MEMORY`, `READING_CONTEXT`, `LEARNING` states |
| `startup/launcher.py` | Startup memory recovery |

There is no "voice JARVIS" and "text JARVIS": both reach the same vault,
the same Jobs, the same Skills, the same missions and the same learning.

## Diagnostics

```
python -m vault status                     where it is, what is in it
python -m vault scan "fix the music bug"   stage 1 only, with scores
python -m vault prime "fix the music bug"  the full knowledge boot
python -m vault jobs / skills / missions
python -m vault daily [YYYY-MM-DD]
python -m vault recover                    what a new session recovers
python -m vault learn "from now on ..."    what a correction would change
python -m vault index                      regenerate VAULT_INDEX.md
```

## Ready for Job: Clipping

`jobs/clipping.md` already exists as a placeholder. Everything the real
Job needs is in place: summary-based discovery, Skill loading, a
persistent Mission that survives a restart, progress appended as it goes,
learning, and the Daily Note that becomes the morning report. What is
missing is only the Skills themselves (Campaign Discovery, Video
Transcription, FFmpeg Video Editing, and so on) -- writing them, and
filling in the Job's Procedure, needs no change to any of this.
