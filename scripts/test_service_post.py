import argparse,urllib.request, json
parser=argparse.ArgumentParser(description="Explicit bilingual service playback test.");parser.add_argument("--run",action="store_true");args=parser.parse_args()
if not args.run:raise SystemExit("Refusing to play speech without --run.")
url='http://127.0.0.1:5001/synthesize'
for payload in ({'text':'Okay, opening YouTube.','language_id':'en'},{'text':'אוקיי, פותח יוטיוב.','language_id':'he'}):
    data=json.dumps(payload).encode('utf-8')
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=120) as r:
            print('RESP',r.status, r.read()[:200])
    except Exception as e:
        print('ERR',e)
