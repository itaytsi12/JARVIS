import os
import time
import json
from pathlib import Path
import sys
import argparse

# Ensure project root is on sys.path so local packages import correctly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser=argparse.ArgumentParser(description="Synthetic STT pipeline test; desktop actions are opt-in.")
parser.add_argument("--execute-actions",action="store_true",help="allow recognized test phrases to execute through JARVIS")
args=parser.parse_args()

phrases = [
    "Open YouTube",
    "Search YouTube for Jude Law",
    "I want to watch Jude Law on YouTube",
    "Open Notepad",
    "Turn the volume down",
    "Make it a little louder",
]

extra_extraction_tests = [
    ("I want to watch Veritasium on YouTube", "Veritasium"),
    ("I'd like to watch Iron Man trailer on YouTube", "Iron Man trailer"),
    ("Show me Linus Tech Tips on YouTube", "Linus Tech Tips"),
    ("Look up neural networks on YouTube", "neural networks"),
    ("Play Minecraft music on YouTube", "Minecraft music"),
]

out_dir = Path("./tmp_stt_tests")
out_dir.mkdir(exist_ok=True)

# Synthesise WAVs using pyttsx3
try:
    import pyttsx3
except Exception as e:
    print('pyttsx3 not available:', e)
    raise SystemExit(1)

engine = pyttsx3.init()
# Try to set to a reasonable rate
try:
    engine.setProperty('rate', 150)
except Exception:
    pass

wav_paths = []
for i, p in enumerate(phrases, start=1):
    path = out_dir / f"test_{i}.wav"
    wav_paths.append(str(path))
    engine.save_to_file(p, str(path))

engine.runAndWait()

# Allow a small delay to ensure files are flushed
time.sleep(0.5)

# Run through pipeline
from voice.speech_to_text import transcribe_audio, is_available as stt_available
from voice.text_normalizer import normalize_transcript
from brain.local_intent_model import predict_local_intent
from brain.router import route_command
from brain.agent import run_agent

results = []

for phrase, wav in zip(phrases, wav_paths):
    item = {"phrase": phrase, "wav": wav}
    try:
        raw = transcribe_audio(wav)
    except Exception as e:
        raw = None
        item['error'] = str(e)

    item['raw_stt'] = raw

    if raw:
        normalized, wake_removed = normalize_transcript(raw)
    else:
        normalized, wake_removed = None, False

    item['normalized'] = normalized
    item['wake_removed'] = wake_removed

    # Local model prediction (may return None if service down)
    try:
        pred = predict_local_intent(normalized or "")
    except Exception as e:
        pred = None
    item['local_model_prediction'] = pred

    # Route via deterministic router
    try:
        route = route_command(normalized or "")
    except Exception as e:
        route = {"error": str(e)}
    item['route'] = route

    # Real desktop/browser/audio side effects require explicit opt-in.
    exec_result = None
    exec_error = None
    if args.execute_actions:
        try:
            res = run_agent(normalized or "")
            exec_result = res
        except Exception as e:
            exec_error = str(e)
    else:
        exec_result = "SKIPPED: pass --execute-actions to run desktop actions"

    item['execution_result'] = exec_result
    item['execution_error'] = exec_error

    results.append(item)

# Run extraction-only tests
extraction_results = []
from brain.local_intent_model import extract_youtube_query

for text, expected in extra_extraction_tests:
    extracted = extract_youtube_query(text)
    extraction_results.append({
        "text": text,
        "expected": expected,
        "extracted": extracted,
    })

print('\nExtraction-only tests:')
print(json.dumps(extraction_results, ensure_ascii=False, indent=2))

# Print compact JSON
print(json.dumps(results, ensure_ascii=False, indent=2))
