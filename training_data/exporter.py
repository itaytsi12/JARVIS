from __future__ import annotations
import argparse,hashlib,json,random,sqlite3
from pathlib import Path
from .validator import validate_database
from .sanitizer import privacy_safe_event,sanitize_user_request

def _rows(db,eligible=None):
    c=sqlite3.connect(db);c.row_factory=sqlite3.Row; sql="SELECT * FROM training_examples"+(" WHERE training_eligible=1" if eligible else "")+" ORDER BY created_at"; rows=[dict(r) for r in c.execute(sql)];c.close();return rows
def _decode(r):
    d=dict(r)
    for k in ("quality_evidence_json","input_json","context_json","output_json","event_ids_json"):d[k[:-5]]=json.loads(d.pop(k))
    d["training_eligible"]=bool(d["training_eligible"])
    if isinstance(d.get("input"),dict) and isinstance(d["input"].get("original_user_text"),str):d["input"]["original_user_text"]=sanitize_user_request(d["input"]["original_user_text"])
    if isinstance(d.get("output"),dict):d["output"]["events"]=[{**e,"payload":privacy_safe_event(e.get("type"),e.get("payload",{}))} for e in d["output"].get("events",[]) if isinstance(e,dict)]
    return d
def format_example(r,fmt):
    d=_decode(r); user=d["input"].get("original_user_text",""); output=d["output"]
    final=output.get("final",{}).get("response") or output.get("final",{}).get("message") or json.dumps(output,ensure_ascii=False)
    if fmt=="sft":return {"messages":[{"role":"user","content":user},{"role":"assistant","content":final}],"metadata":{"example_id":d["example_id"],"quality_evidence":d["quality_evidence"],"group_id":d.get("task_id") or d["interaction_id"]}}
    if fmt=="tools":return {"input":user,"context":d["context"],"trajectory":[e for e in output.get("events",[]) if e["type"] in {"TOOL_CALL","TOOL_RESULT"}],"metadata":{"group_id":d.get("task_id") or d["interaction_id"]}} if d["category"]=="tool" else None
    if fmt=="coding":return d if d["category"]=="coding" else None
    return d
def preference_pairs(rows):
    decoded=[_decode(r) for r in rows]; by_task={}
    for d in decoded:by_task.setdefault(d.get("task_id") or d["interaction_id"],[]).append(d)
    out=[]
    for group in by_task.values():
        chosen=next((x for x in reversed(group) if x["training_eligible"]),None); rejected=next((x for x in group if x["quality_label"] in {"FAILED","REGRESSION","ROLLED_BACK","USER_REJECTED"}),None)
        if chosen and rejected:out.append({"prompt":chosen["input"],"chosen":chosen["output"],"rejected":rejected["output"],"task_id":chosen.get("task_id")})
    return out
def split_grouped(rows,seed=42,validation=.1,test=.1):
    groups={}
    for r in rows:
        group=(r.get("metadata") or {}).get("group_id") if isinstance(r.get("metadata"),dict) else None
        groups.setdefault(r.get("task_id") or r.get("interaction_id") or group,[]).append(r)
    keys=sorted(groups); random.Random(seed).shuffle(keys); n=len(keys); nt=int(n*test); nv=int(n*validation); assignment={k:("test" if i<nt else "validation" if i<nt+nv else "train") for i,k in enumerate(keys)}
    return {name:[r for k in keys if assignment[k]==name for r in groups[k]] for name in ("train","validation","test")}
def export_dataset(database,fmt,output,split=False,seed=42):
    errors=validate_database(database)
    if errors:raise ValueError("Dataset validation failed: "+"; ".join(errors[:5]))
    target=Path(output)
    if target.exists():raise FileExistsError(f"Refusing to overwrite {target}")
    rows=_rows(database,eligible=fmt in {"sft","tools","coding"})
    if fmt=="preferences":items=preference_pairs(_rows(database))
    elif fmt=="raw":
        c=sqlite3.connect(database);c.row_factory=sqlite3.Row;items=[]
        for row in c.execute("SELECT * FROM raw_events ORDER BY interaction_id,sequence_number"):
            item=dict(row);item["payload_json"]=json.dumps(privacy_safe_event(item["event_type"],json.loads(item["payload_json"])),ensure_ascii=False,separators=(",",":"));items.append(item)
        c.close()
    else:items=[x for x in (format_example(r,fmt) for r in rows) if x is not None]
    target.parent.mkdir(parents=True,exist_ok=True)
    if split:
        grouped=split_grouped(items,seed); target.mkdir(parents=True,exist_ok=False)
        for name,values in grouped.items():_write(target/f"{name}.jsonl",values)
    else:_write(target,items)
    return len(items)
def _write(path,items):
    with Path(path).open("x",encoding="utf-8") as f:
        for item in items:f.write(json.dumps(item,ensure_ascii=False)+"\n")
def main():
    p=argparse.ArgumentParser();p.add_argument("--database",default="data/training_dataset.sqlite3");p.add_argument("--format",choices=["sft","tools","preferences","coding","raw"],required=True);p.add_argument("--output",required=True);p.add_argument("--split",action="store_true");p.add_argument("--seed",type=int,default=42);a=p.parse_args();print(f"Exported {export_dataset(a.database,a.format,a.output,a.split,a.seed)} examples")
if __name__=="__main__":main()
