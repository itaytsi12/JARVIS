import json
import random
from pathlib import Path
import re

random.seed(42)
DATA_DIR = Path("training/data")
train_file = DATA_DIR / "train.jsonl"
val_file = DATA_DIR / "validation.jsonl"
test_file = DATA_DIR / "test.jsonl"

def load(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]

train = load(train_file)
val = load(val_file)
test = load(test_file)

intents = sorted({ex['label'] for ex in train+val+test})

SAMPLE = { 'train':10, 'val':5, 'test':5 }

# Helper pools to detect mismatches
APPS = set([
    "notepad", "calculator", "vscode", "visual studio code", "spotify",
    "discord", "chrome", "google chrome", "paint", "file explorer",
    "slack", "teams", "word", "excel", "chrome beta", "edge",
    "powerpoint", "outlook", "terminal", "cmd", "photos",
    "settings", "task manager", "onenote", "explorer",
])
WEBSITES = set(["youtube","google","github","reddit","wikipedia","netflix","twitch","instagram","stackoverflow","bing","amazon","ebay","cnn","bbc","twitter"])
FOLDER_KEYWORDS = set(["folder","directory","downloads","documents","desktop","pictures","music","projects","work","invoices"])
FILE_EXT_RE = re.compile(r"\.[a-zA-Z0-9]{1,5}$")
HOTKEY_SIGNS = re.compile(r"\b(ctrl|alt|shift|\bctrl\b|\balt\b|\bshift\b|\bmeta\b|\bwin\b|\bcommand\b)\b", re.I)
KEY_NAMES = set(["enter","escape","space","tab","backspace","delete","home","end","page up","page down"] + [f"f{i}" for i in range(1,13)] + ["arrow up","arrow down","arrow left","arrow right"])

# Function to sample

def sample_examples(label, split_list, n):
    items = [ex['text'] for ex in split_list if ex['label']==label]
    if not items:
        return []
    n = min(n, len(items))
    return random.sample(items, n)

# Print samples
print('--- Dataset inspection samples ---')
for intent in intents:
    print(f"\n== Intent: {intent} ==")
    t_samples = sample_examples(intent, train, SAMPLE['train'])
    v_samples = sample_examples(intent, val, SAMPLE['val'])
    s_samples = sample_examples(intent, test, SAMPLE['test'])
    print('\n-- Train samples --')
    for s in t_samples:
        print('- ', s)
    print('\n-- Validation samples --')
    for s in v_samples:
        print('- ', s)
    print('\n-- Test samples --')
    for s in s_samples:
        print('- ', s)

# Now run checks
print('\n\n--- Automatic quality checks ---')
all_examples = {'train':train, 'val':val, 'test':test}

problems = []

# Basic stats
counts = { 'train': len(train), 'val': len(val), 'test': len(test) }
print('Counts:', counts)

# Check duplicates exact
all_texts = [e['text'] for e in train+val+test]
dups = len(all_texts) - len(set(all_texts))
print('Exact duplicate count across all splits:', dups)

# Heuristic checks per example
for split_name, split_list in all_examples.items():
    for ex in split_list:
        text = ex['text']
        label = ex['label']
        lower = text.lower()

        # 1. placeholder/template artifacts
        if '{' in text or '}' in text:
            problems.append((label, split_name, text, 'placeholder artifact'))
            continue

        # 2. broken english heuristics: repeated punctuation or odd symbols
        if re.search(r'[^\x00-\x7F]', text):
            problems.append((label, split_name, text, 'non-ascii characters'))
        if re.search(r'(\!{3,}|\?{3,}|\.{4,})', text):
            problems.append((label, split_name, text, 'excessive punctuation'))
        if len(text.strip())==0:
            problems.append((label, split_name, text, 'empty text'))

        # 3. wrong labels / confusion heuristics
        # open_application vs open_website
        if label=='open_application' and any(w in lower for w in WEBSITES):
            problems.append((label, split_name, text, 'mentions website but labeled open_application'))
        if label=='open_website' and any(a in lower for a in APPS):
            problems.append((label, split_name, text, 'mentions app but labeled open_website'))

        # web_search vs youtube_search
        if label=='web_search' and 'youtube' in lower:
            problems.append((label, split_name, text, 'mentions youtube but labeled web_search'))
        if label=='youtube_search' and 'youtube' not in lower and any(q in lower for q in ['video','watch','play']):
            # might still be youtube search, but flag if no youtube token
            problems.append((label, split_name, text, 'youtube-style phrasing but no youtube token'))

        # volume up/down
        if label in ('volume_up','volume_down'):
            up_words = ['up','louder','increase','raise']
            down_words = ['down','quieter','decrease','lower']
            has_up = any(w in lower for w in up_words)
            has_down = any(w in lower for w in down_words)
            if has_up and label=='volume_down':
                problems.append((label, split_name, text, 'text indicates increase but labeled volume_down'))
            if has_down and label=='volume_up':
                problems.append((label, split_name, text, 'text indicates decrease but labeled volume_up'))

        # mute/unmute
        if label=='mute_volume' and 'unmute' in lower:
            problems.append((label, split_name, text, 'says unmute but labeled mute_volume'))
        if label=='unmute_volume' and 'mute' in lower and 'unmute' not in lower:
            problems.append((label, split_name, text, 'says mute but labeled unmute_volume'))

        # close_application vs close_window
        if label=='close_application' and 'window' in lower:
            problems.append((label, split_name, text, 'mentions window but labeled close_application'))
        if label=='close_window' and any(a in lower for a in APPS):
            problems.append((label, split_name, text, 'mentions app but labeled close_window'))

        # press_key vs hotkey
        if label=='press_key' and HOTKEY_SIGNS.search(lower):
            problems.append((label, split_name, text, 'contains modifier keys but labeled press_key'))
        if label=='hotkey' and not HOTKEY_SIGNS.search(lower) and not any(k in lower for k in KEY_NAMES):
            problems.append((label, split_name, text, 'hotkey label but no modifier or known key'))

        # open_file vs open_folder
        if label=='open_file' and not FILE_EXT_RE.search(text) and not any(f in lower for f in ['file','document']):
            problems.append((label, split_name, text, 'open_file but no file extension or file word'))
        if label=='open_folder' and any(FILE_EXT_RE.search(text) for _ in [1]) and 'folder' not in lower:
            problems.append((label, split_name, text, 'open_folder but looks like a file path'))

        # switch_window vs open_application
        if label=='switch_window' and any(w in lower for w in ['open','launch','start']) and any(a in lower for a in APPS):
            problems.append((label, split_name, text, 'switch_window phrasing looks like open_application'))

        # placeholder empty entity: e.g., 'open ' or 'press  '
        if re.search(r'\b(open|launch|start|press|type|run)\s*$' , text, re.I):
            problems.append((label, split_name, text, 'missing argument after verb'))

        # dangerous run_command
        if label=='run_command':
            dangerous = ['rm -rf','rm -r','shutdown','reboot','format','del ', 'rd ', 'mkfs', 'dd if=', 'poweroff']
            if any(d in lower for d in dangerous):
                problems.append((label, split_name, text, 'dangerous command'))

# Summarize problems
print('\nFound problems:', len(problems))
if problems:
    # group by problem type and show examples
    by_type = {}
    for p in problems:
        by_type.setdefault(p[3], []).append(p)
    for typ, items in by_type.items():
        print(f"\nProblem type: {typ} (examples up to 5)")
        for it in items[:5]:
            print(f" - [{it[1]}] ({it[0]}) {it[2]}")

# Confusion-prone pair checks: show sample cross-label hints
print('\n\n--- Confusion-prone pair sampling checks ---')
# open_application vs open_website: examples where text contains website and label open_application
for ex in train+val+test:
    t = ex['text'].lower()
    if ex['label']=='open_application' and any(w in t for w in WEBSITES):
        print('open_application but mentions website:', ex['text'])
        break

for ex in train+val+test:
    t = ex['text'].lower()
    if ex['label']=='open_website' and any(a in t for a in APPS):
        print('open_website but mentions app:', ex['text'])
        break

# web_search vs youtube_search
for ex in train+val+test:
    t = ex['text'].lower()
    if ex['label']=='web_search' and 'youtube' in t:
        print('web_search but mentions youtube:', ex['text'])
        break
for ex in train+val+test:
    t = ex['text'].lower()
    if ex['label']=='youtube_search' and 'youtube' not in t and any(w in t for w in ['video','watch','play']):
        print('youtube_search without youtube token:', ex['text'])
        break

print('\nInspection complete.')
