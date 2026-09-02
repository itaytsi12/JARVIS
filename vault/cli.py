"""`python -m vault` -- inspect and exercise the knowledge vault, no voice needed.

    python -m vault status                      where it is, what is in it
    python -m vault bootstrap                   create the structure and seeds
    python -m vault index                       regenerate VAULT_INDEX.md
    python -m vault scan "fix the music bug"    stage 1 only: what would rank
    python -m vault prime "fix the music bug"   the full knowledge boot
    python -m vault jobs                        every Job and its summary
    python -m vault skills                      every Skill and its summary
    python -m vault missions                    active and recent missions
    python -m vault daily [YYYY-MM-DD]          one day's note
    python -m vault recover                     what a new session recovers
    python -m vault learn "from now on ..."     what a correction would change

`scan` and `prime` are the two worth knowing: they print exactly which
notes were considered, which were selected, and why -- which is how a
"JARVIS did not use the note I wrote" report gets diagnosed in one
command instead of by re-running a whole voice session.
"""
from __future__ import annotations

import argparse
import json
import sys


def _vault():
    from vault.manager import get_vault

    return get_vault()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vault", description="Inspect JARVIS's Obsidian knowledge vault")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Where the vault is and what it holds")
    sub.add_parser("bootstrap", help="Create the folder structure and any missing seed note")
    sub.add_parser("index", help="Regenerate VAULT_INDEX.md and the per-folder indexes")
    scan = sub.add_parser("scan", help="Stage 1 only: rank note summaries, read nothing")
    scan.add_argument("query")
    scan.add_argument("--type", dest="note_type", default="")
    prime = sub.add_parser("prime", help="The full knowledge boot for one request")
    prime.add_argument("query")
    prime.add_argument("--budget", type=int, default=6000)
    sub.add_parser("jobs", help="Every Job and its summary")
    sub.add_parser("skills", help="Every Skill and its summary")
    sub.add_parser("missions", help="Active and recently completed missions")
    daily = sub.add_parser("daily", help="One day's note")
    daily.add_argument("date", nargs="?", default="")
    sub.add_parser("recover", help="What a new session recovers at startup")
    learn = sub.add_parser("learn", help="What a correction would change (does not write)")
    learn.add_argument("text")
    learn.add_argument("--apply", action="store_true", help="Actually write the change")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "status":
        from vault.index import get_index

        payload = _vault().describe()
        payload.update(get_index().statistics())
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "bootstrap":
        from vault.bootstrap import bootstrap_vault

        print(json.dumps(bootstrap_vault(), indent=2))
        return 0

    if args.command == "index":
        from vault.index import get_index

        index = get_index()
        index.invalidate()
        index.refresh(force=True)
        print(f"Wrote {index.write_markdown_index()}")
        return 0

    if args.command == "scan":
        from vault.retrieval import get_retriever

        candidates, scanned, scan_ms = get_retriever().scan(args.query, types=[args.note_type] if args.note_type else None)
        print(f'Scanned {scanned} note summaries in {scan_ms:.1f}ms for "{args.query}".\n')
        if not candidates:
            print("Nothing scored above zero.")
            return 0
        for candidate in candidates[:15]:
            print(f"  {candidate.score:6.2f}  {candidate.summary.title} ({candidate.relative_path})")
            print(f"          {'; '.join(candidate.reasons)}")
        return 0

    if args.command == "prime":
        from vault.priming import get_primer

        primed = get_primer().prime(args.query, budget_chars=args.budget)
        print(primed.explain())
        print("\n--- context handed to the model ---")
        for name, text in primed.extra_sections().items():
            print(f"\n[{name}] ({len(text)} chars)\n{text[:600]}")
        return 0

    if args.command == "jobs":
        from vault.jobs import get_job_registry

        print(get_job_registry().catalog() or "No Job notes yet.")
        return 0

    if args.command == "skills":
        from vault.skills import get_skill_library

        print(get_skill_library().catalog() or "No Skill notes yet.")
        return 0

    if args.command == "missions":
        from vault.missions import get_mission_store

        store = get_mission_store()
        active = store.active()
        print(f"Active ({len(active)}):")
        for mission in active:
            print(f"  {mission.status:12s} {mission.relative_path}  {mission.goal[:60]}")
        print("\nRecently completed:")
        for mission in store.completed(limit=10):
            print(f"  {mission.status:12s} {mission.relative_path}  {mission.goal[:60]}")
        return 0

    if args.command == "daily":
        from vault.daily import get_journal
        from vault.note import today_stamp

        journal = get_journal()
        note = journal.existing(args.date or today_stamp())
        if note is None:
            print(f"No daily note for {args.date or today_stamp()}.")
            return 1
        content = _vault().read(note.relative_path)
        print(content.to_markdown() if content else "(unreadable)")
        return 0

    if args.command == "recover":
        from vault.startup import recover_session

        recovery = recover_session(mark_interrupted=False, create_today=False)
        print(json.dumps(recovery.describe(), indent=2))
        print("\n--- context a new session starts with ---")
        print(recovery.context_text() or "(nothing recovered)")
        return 0

    if args.command == "learn":
        from vault.learning import classify_feedback, get_correction_learner, rewrite_as_rule

        feedback = classify_feedback(args.text)
        print(json.dumps(feedback.describe(), indent=2))
        print(f"\nAs a rule: {rewrite_as_rule(args.text) or '(nothing usable)'}")
        if not args.apply:
            print("\n(dry run -- pass --apply to actually write it)")
            return 0
        outcome = get_correction_learner().apply(args.text, feedback=feedback)
        print("\n" + json.dumps(outcome.describe(), indent=2))
        return 0 if outcome.applied else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
