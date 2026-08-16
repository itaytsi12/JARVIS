"""Conservative, deterministic readiness gate for a future local planner."""
from __future__ import annotations

import argparse,json,sqlite3
from pathlib import Path

from .validator import validate_database


DEFAULT_THRESHOLDS={
    "eligible_examples":500,
    "eligible_action_types":10,
    "eligible_multistep_examples":100,
    "eligible_ui_examples":100,
    "correction_examples":50,
    "cancellation_examples":50,
    "held_out_groups":50,
}
UI_TOOLS={"focus_application","wait_for_window","type_text","press_key","click_at","click_ui_element","inspect_window","browser_click","browser_type","browser_select","browser_scroll"}


def evaluate_readiness(database,thresholds=None):
    path=Path(database);limits={**DEFAULT_THRESHOLDS,**(thresholds or {})};errors=validate_database(path)
    connection=sqlite3.connect(path);connection.row_factory=sqlite3.Row
    examples=list(connection.execute("SELECT interaction_id,task_id,training_eligible FROM training_examples"))
    eligible_ids={row["interaction_id"] for row in examples if row["training_eligible"]}
    actions={};features={iid:{"actions":set(),"multi":False} for iid in eligible_ids}
    for row in connection.execute("SELECT interaction_id,event_type,payload_json FROM raw_events"):
        iid=row["interaction_id"]
        if iid not in eligible_ids:continue
        payload=json.loads(row["payload_json"])
        if row["event_type"]=="TOOL_CALL" and payload.get("name"):
            name=payload["name"];actions[name]=actions.get(name,0)+1;features[iid]["actions"].add(name)
        if row["event_type"]=="PLAN_CREATED" and len(payload.get("actions") or [])>1:features[iid]["multi"]=True
    labels={row[0]:row[1] for row in connection.execute("SELECT quality_label,COUNT(*) FROM training_examples GROUP BY quality_label")}
    groups={row["task_id"] or row["interaction_id"] for row in examples if row["training_eligible"]}
    metrics={
        "eligible_examples":len(eligible_ids),
        "eligible_action_types":len(actions),
        "eligible_multistep_examples":sum(item["multi"] for item in features.values()),
        "eligible_ui_examples":sum(bool(item["actions"]&UI_TOOLS) for item in features.values()),
        "correction_examples":labels.get("CORRECTED",0),
        "cancellation_examples":sum("cancel_active_task" in item["actions"] for item in features.values()),
        "held_out_groups":len(groups),
    }
    connection.close()
    checks={name:{"value":metrics[name],"minimum":minimum,"passed":metrics[name]>=minimum} for name,minimum in limits.items()}
    ready=not errors and all(check["passed"] for check in checks.values())
    return {"ready":ready,"recommendation":"train_bounded_local_planner" if ready else "continue_verified_data_collection","validation_errors":errors,"metrics":metrics,"checks":checks,"eligible_action_counts":actions}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--database",default="data/training_dataset.sqlite3");args=parser.parse_args()
    print(json.dumps(evaluate_readiness(args.database),indent=2,sort_keys=True))


if __name__=="__main__":main()
