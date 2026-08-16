"""Opt-in real API benchmark for the read-only fast-question path."""
import argparse,statistics,time,sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.request_intent import classify_request_kind
from brain.web_answer import get_web_answer_service

QUERIES=["Who is the CEO of Nvidia?","What is the latest Python version?","What happened with Bitcoin today?","Who is the president of France?"]
def main():
    parser=argparse.ArgumentParser(description="Opt-in benchmark using real web-answer API calls.");parser.add_argument("--run",action="store_true",help="perform the four real API requests");args=parser.parse_args()
    if not args.run:
        print("Refusing to call the API without --run.")
        return 2
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    service=get_web_answer_service(); totals=[]
    for query in QUERIES:
        started=time.perf_counter();intent_start=time.perf_counter();decision=classify_request_kind(query);intent_ms=(time.perf_counter()-intent_start)*1000;result=service.answer(query);total=(time.perf_counter()-started)*1000;totals.append(result.web_request_ms);print({"query":query,"intent":decision.kind.value,"intent_ms":round(intent_ms,1),"web_ms":round(result.web_request_ms,1),"answer_ms":round(result.answer_processing_ms,1),"total_ms":round(total,1),"success":result.success,"answer":result.answer})
    print({"web_min_ms":round(min(totals),1),"web_median_ms":round(statistics.median(totals),1),"web_max_ms":round(max(totals),1)})
    return 0
if __name__=="__main__":raise SystemExit(main())
