"""Bounded inspection/filtering for the local structured dataset."""
import argparse,json,sqlite3
from .sanitizer import privacy_safe_event,sanitize_user_request


def inspect_examples(database,label=None,action=None,route=None,limit=20):
    connection=sqlite3.connect(database);connection.row_factory=sqlite3.Row;items=[]
    try:
        for row in connection.execute("SELECT * FROM training_examples ORDER BY created_at DESC"):
            output=json.loads(row["output_json"]);events=output.get("events",[])
            if label and row["quality_label"]!=label:continue
            if action and not any(event.get("type")=="TOOL_CALL" and event.get("payload",{}).get("name")==action for event in events):continue
            if route and not any(event.get("type")=="PLAN_CREATED" and event.get("payload",{}).get("route_type")==route for event in events):continue
            input_data=json.loads(row["input_json"])
            if isinstance(input_data.get("original_user_text"),str):input_data["original_user_text"]=sanitize_user_request(input_data["original_user_text"])
            output["events"]=[{**event,"payload":privacy_safe_event(event.get("type"),event.get("payload",{}))} for event in events if isinstance(event,dict)]
            items.append({"example_id":row["example_id"],"interaction_id":row["interaction_id"],"category":row["category"],"quality_label":row["quality_label"],"training_eligible":bool(row["training_eligible"]),"created_at":row["created_at"],"input":input_data,"output":output})
            if len(items)>=max(0,min(limit,100)):break
    finally:connection.close()
    return items


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--database",default="data/training_dataset.sqlite3");parser.add_argument("--label");parser.add_argument("--action");parser.add_argument("--route");parser.add_argument("--limit",type=int,default=20);args=parser.parse_args()
    for item in inspect_examples(args.database,args.label,args.action,args.route,args.limit):print(json.dumps(item,ensure_ascii=False))
if __name__=="__main__":main()
