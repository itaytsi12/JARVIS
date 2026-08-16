from __future__ import annotations
import argparse,json,re,sqlite3
from pathlib import Path
from .sanitizer import PATTERNS, sanitize_text
from .schema import VALID_EVENTS,VALID_LABELS

LEAKS=[re.compile(r"(?i)\b(?:sk-[a-z0-9_-]{12,}|gh[pousr]_[a-z0-9]{12,})\b"),re.compile(r"(?i)Bearer\s+(?!<REDACTED>)[a-z0-9._~+/-]+"),re.compile(r"(?i)\"(?:password|token|authorization|cookie|api.?key)\"\s*:\s*\"(?!<REDACTED>)[^\"]+\""),re.compile(r"-----BEGIN .*PRIVATE KEY-----")]
def _has_secret(value):
    text=json.dumps(value,ensure_ascii=False)
    return any(p.search(text) for p in LEAKS)
def validate_database(path):
    errors=[]; c=sqlite3.connect(path); c.row_factory=sqlite3.Row
    try:
        for iid,rows in _groups(c,"SELECT * FROM raw_events ORDER BY interaction_id,sequence_number"):
            expected=1;prepared=set();terminal=set();task_completed=False;task_failed=False;final_success=None
            for r in rows:
                if r["event_type"] not in VALID_EVENTS:errors.append(f"invalid event type {r['event_type']}")
                if r["sequence_number"]!=expected:errors.append(f"invalid sequence {iid}")
                expected+=1
                try:payload=json.loads(r["payload_json"])
                except Exception:errors.append(f"corrupt event JSON {r['event_id']}");continue
                if _has_secret(payload):errors.append(f"secret pattern {r['event_id']}")
                if r["event_type"]=="TASK_COMPLETED":task_completed=True
                if r["event_type"]=="TASK_FAILED":task_failed=True
                if r["event_type"]=="FINAL_RESPONSE" and isinstance(payload.get("success"),bool):final_success=payload["success"]
                action_id=payload.get("action_id") if isinstance(payload,dict) else None
                if r["event_type"]=="TOOL_CALL" and action_id:
                    prepared.add(action_id)
                    if payload.get("status")!="prepared":errors.append(f"invalid prepared action status {r['event_id']}")
                if r["event_type"]=="TOOL_RESULT" and action_id:
                    if action_id not in prepared:errors.append(f"action result without prepared action {r['event_id']}")
                    status=payload.get("status")
                    if status not in {"committed","failed","not_executed"}:errors.append(f"invalid terminal action status {r['event_id']}")
                    if status=="committed" and payload.get("success") is not True or status in {"failed","not_executed"} and payload.get("success") is not False:errors.append(f"contradictory action result {r['event_id']}")
                    terminal.add(action_id)
            for action_id in prepared-terminal:errors.append(f"prepared action without terminal result {iid}:{action_id}")
            if final_success is False and task_completed or final_success is True and task_failed:errors.append(f"contradictory task/final outcome {iid}")
        for r in c.execute("SELECT * FROM training_examples"):
            if r["quality_label"] not in VALID_LABELS:errors.append(f"invalid label {r['example_id']}")
            if r["training_eligible"] and r["quality_label"] in {"FAILED","REGRESSION","ROLLED_BACK","USER_REJECTED","BLOCKED","AMBIGUOUS"}:errors.append(f"ineligible label exported positive {r['example_id']}")
            for col in ("quality_evidence_json","input_json","context_json","output_json","event_ids_json"):
                try:value=json.loads(r[col])
                except Exception:errors.append(f"corrupt example JSON {r['example_id']}:{col}");continue
                if _has_secret(value):errors.append(f"secret pattern {r['example_id']}:{col}")
    finally:c.close()
    return errors
def _groups(c,sql):
    grouped={}
    for r in c.execute(sql):grouped.setdefault(r["interaction_id"],[]).append(r)
    return grouped.items()
def main():
    p=argparse.ArgumentParser();p.add_argument("--database",default="data/training_dataset.sqlite3");a=p.parse_args(); errors=validate_database(a.database); print("Dataset valid" if not errors else "\n".join(errors)); raise SystemExit(bool(errors))
if __name__=="__main__":main()
