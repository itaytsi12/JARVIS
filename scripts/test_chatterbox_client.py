import argparse,importlib,traceback
parser=argparse.ArgumentParser(description="Explicit Chatterbox speech smoke test.");parser.add_argument("--run",action="store_true");args=parser.parse_args()
if not args.run:raise SystemExit("Refusing to play speech without --run.")
m=importlib.import_module('voice.tts.chatterbox_tts')
try:
    m.speak('Okay, opening YouTube.', lang='en')
    print('speak call returned')
except Exception as e:
    traceback.print_exc()
    print('speak failed', e)
