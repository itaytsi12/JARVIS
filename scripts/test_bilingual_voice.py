import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice.text_normalizer import normalize_transcript
from voice.language_utils import detect_dominant_language
from brain.router import route_command
from voice.response_formatter import format_spoken_response

examples = [
    "היי Jarvis תפתח notepad",
    "Jarvis open notepad",
    "תחפש ביוטיוב Jude Law",
    "search ביוטיוב for Veritasium",
    "היי Jarvis תנמיך volume",
]

for cmd in examples:
    print('ORIGINAL:', cmd)
    cleaned, wake_removed = normalize_transcript(cmd)
    print('CLEANED:', cleaned, 'wake_removed=', wake_removed)
    route = route_command(cleaned)
    print('ROUTE_TYPE:', route.get('type'))
    lang = detect_dominant_language(cmd)
    spoken = format_spoken_response(cleaned, route, 'SIMULATED_TOOL_OUTPUT', lang=lang)
    print('DETECTED_LANG:', lang)
    print('SPOKEN:', spoken)
    print('---')
