import argparse,json
from .recorder import DatasetRecorder
def main():
    p=argparse.ArgumentParser();p.add_argument("--database",default="data/training_dataset.sqlite3");a=p.parse_args();r=DatasetRecorder(a.database,async_writes=False);print(json.dumps(r.stats(),indent=2));r.close()
if __name__=="__main__":main()
