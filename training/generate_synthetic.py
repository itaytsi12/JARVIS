import json
import random
import re
from pathlib import Path
from collections import Counter

random.seed(42)

OUTPUT_DIR = Path("training/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target sizes per intent
TRAIN_PER_INTENT = 1000
VALIDATION_PER_INTENT = 150
TEST_PER_INTENT = 200


# ============================================================
# Dynamic value pools (English-only)
# ============================================================

APPS = [
    "notepad", "calculator", "vscode", "visual studio code", "spotify",
    "discord", "chrome", "google chrome", "paint", "file explorer",
    "slack", "teams", "word", "excel", "chrome beta", "edge",
    "powerpoint", "outlook", "terminal", "cmd", "photos",
    "settings", "task manager", "calculator", "onenote", "explorer",
]

WEBSITES = [
    "youtube", "google", "github", "reddit", "wikipedia",
    "netflix", "twitch", "instagram", "stackoverflow", "bing",
    "amazon", "ebay", "cnn", "bbc", "twitter",
]

SEARCH_QUERIES = [
    "Jude Law", "Minecraft", "Iron Man trailer", "Python tutorial",
    "Veritasium", "AI news", "how to learn machine learning",
    "best Minecraft builds", "Linus Tech Tips", "Marvel",
    "chess openings", "neural networks", "computer vision",
    "Docker tutorial", "time complexity of quicksort", "weather today",
    "coffee shops near me", "nearby restaurants", "movie times",
    "how to tie a tie", "home workout routines", "best laptops 2026",
]

FILES = ["resume.pdf", "notes.txt", "presentation.pptx", "report.docx"]
FOLDERS = ["downloads", "documents", "desktop", "pictures", "music", "projects", "work", "invoices"]

KEYS = [
    "enter", "escape", "space", "tab", "backspace", "delete", "home", "end",
    "page up", "page down", "pageup", "pagedown",
] + [f"f{i}" for i in range(1, 13)] + ["arrow up", "arrow down", "arrow left", "arrow right"]
HOTKEYS = [
    "ctrl+c", "ctrl+v", "ctrl+shift+esc", "alt+tab", "ctrl+alt+del",
    "ctrl+z", "ctrl+y", "ctrl+s", "ctrl+p", "ctrl+shift+s",
    "control shift escape",
]
COMMANDS = [
    # safe, benign commands only
    "ipconfig", "whoami", "ping localhost", "git status", "pip list",
    "python main.py", "code .", "dir", "echo hello", "hostname", "tasklist",
]


# ============================================================
# Templates for each intent (train / val / test separately)
# Keep phrasing differences across splits to avoid identical overlap
# ============================================================

def templates_for_intent(intent):
    t = {"train": [], "val": [], "test": []}

    if intent == "open_application":
        t["train"] = [
            "open {app}", "launch {app}", "start {app}", "can you open {app}",
            "please open {app}", "jarvis open {app}", "bring up {app}",
            "get {app} running",
        ]
        t["val"] = ["could you launch {app} for me", "I need {app} open", "open {app} for me"]
        t["test"] = ["I want to use {app}", "get {app} running for me", "open up {app} now"]

    elif intent == "close_application":
        t["train"] = ["close {app}", "quit {app}", "shut down {app}", "please close {app}"]
        t["val"] = ["exit {app}", "terminate {app}", "close the {app} app"]
        t["test"] = ["shut {app} down", "please quit {app} now", "I want {app} closed"]

    elif intent == "open_website":
        t["train"] = [
            "open {website}", "go to {website}", "please open {website}",
            "navigate to {website}", "bring up {website}", "load {website}",
        ]
        t["val"] = ["take me to {website}", "bring up {website}", "load {website}"]
        t["test"] = ["send me over to {website}", "I want to check {website}", "open the {website} site"]

    elif intent == "web_search":
        t["train"] = ["search the web for {query}", "google {query}", "look up {query}", "search for {query}"]
        t["val"] = ["find information about {query}", "look up {query} on the web", "web search for {query}"]
        t["test"] = ["search online for {query}", "do a web search for {query}", "find {query} online"]

    elif intent == "youtube_search":
        t["train"] = [
            "search YouTube for {query}", "play {query} on YouTube", "I want to watch {query} on YouTube",
            "show me {query} on YouTube", "look up {query} on YouTube",
        ]
        t["val"] = ["get me YouTube results for {query}", "find {query} videos on YouTube"]
        t["test"] = ["pull up {query} videos on YouTube", "show me {query} videos on YouTube"]

    elif intent in ("volume_up", "volume_down"):
        verb = "up" if intent == "volume_up" else "down"
        variants = [
            f"turn the volume {verb}", f"volume {verb}", f"please make it {verb}",
            f"could you turn it {verb}", f"make it a little {verb}", f"turn it {verb} a bit",
            f"make the sound {verb}", f"raise the volume" if verb == "up" else f"lower the volume",
            f"make it louder" if verb == "up" else f"make it quieter",
            f"increase the volume", f"decrease the volume", f"turn speakers {verb}",
            f"crank it {verb}", f"please make the audio {verb}", f"could you make it {verb}",
        ]
        t["train"] = variants + [v + " please" for v in variants]
        t["val"] = [f"adjust the volume {verb}", f"{verb} the audio", f"please {verb} the volume"]
        t["test"] = [f"bring the sound {verb}", f"change volume {verb}", f"{verb} the sound now"]

    elif intent == "mute_volume":
        t["train"] = [
            "mute", "mute the sound", "turn the sound off", "silence audio",
            "please mute", "cut the sound", "switch audio off", "turn speakers off",
            "make it silent",
        ]
        t["val"] = ["please mute", "turn audio off", "mute it now"]
        t["test"] = ["mute the audio now", "stop sound", "silence the audio please"]

    elif intent == "unmute_volume":
        t["train"] = [
            "unmute", "turn the sound back on", "enable audio", "turn audio on",
            "please unmute", "restore sound", "switch audio on", "turn speakers on",
        ]
        t["val"] = ["please unmute", "turn audio back on", "unmute it now"]
        t["test"] = ["unmute now", "resume audio", "restore the audio"]

    elif intent == "type_text":
        t["train"] = [
            "type {text}", "write {text}", "enter {text}", "please type {text}",
            "could you type {text}", "input the text {text}", "add text {text}", "put {text} here",
            "write down {text}", "insert {text}", "transcribe {text}",
        ]
        t["val"] = ["could you type {text}", "type in {text}", "please write {text}"]
        t["test"] = ["write the text {text}", "input {text}", "type this: {text}"]

    elif intent == "press_key":
        t["train"] = [
            "press {key}", "hit {key}", "press the {key} key", "please press {key}",
            "tap {key}", "push {key}", "hit the {key} button", "press {key} now",
        ]
        t["val"] = ["hit the {key} button", "press {key} now", "please press {key}"]
        t["test"] = ["press {key} please", "tap {key}", "press the {key} key now"]

    elif intent == "hotkey":
        t["train"] = [
            "press {keys}", "use {keys}", "hit {keys}", "press the keys {keys}",
            "use the shortcut {keys}", "trigger {keys}", "do {keys}", "execute {keys}",
        ]
        t["val"] = ["trigger {keys}", "use the shortcut {keys}", "please {keys}"]
        t["test"] = ["perform {keys}", "activate {keys}", "press {keys} now"]

    elif intent == "switch_window":
        t["train"] = [
            "switch to {app}", "go back to {app}", "focus {app}",
            "bring {app} forward", "switch windows to {app}",
        ]
        t["val"] = ["switch windows to {app}", "bring {app} forward"]
        t["test"] = ["change to {app}", "switch window to {app}"]

    elif intent == "minimize_window":
        t["train"] = [
            "minimize this window", "minimize the window", "make this smaller",
            "send to tray", "shrink the window", "reduce the window size", "hide this window",
        ]
        t["val"] = ["send to tray", "minimize", "please minimize"]
        t["test"] = ["minimize now", "minimize the active window", "minimize this"]

    elif intent == "maximize_window":
        t["train"] = [
            "maximize this window", "make this full screen", "maximize",
            "expand to full screen", "make it full screen", "zoom this window",
        ]
        t["val"] = ["make full screen", "maximize the window", "please maximize"]
        t["test"] = ["maximize now", "expand window to full screen", "maximize this"]

    elif intent == "close_window":
        t["train"] = [
            "close this window", "close window", "close the active window", "shut this window",
            "please close this", "terminate this window", "close it now",
        ]
        t["val"] = ["shut this window", "close it", "please close the window"]
        t["test"] = ["close the current window", "please close window", "close this"]

    elif intent == "take_screenshot":
        t["train"] = [
            "take a screenshot", "capture my screen", "screenshot", "grab a screenshot",
            "capture screen now", "save screen image", "take a screen capture",
        ]
        t["val"] = ["capture screenshot", "take a screen capture", "please screenshot"]
        t["test"] = ["screenshot now", "take a screenshot for me", "capture the screen"]

    elif intent == "open_file":
        t["train"] = [
            "open {file}", "open the file {file}", "open {file} please", "load {file}",
            "please open {file}", "show me {file}", "display {file}", "open the document {file}",
        ]
        t["val"] = ["please open {file}", "load {file}", "open {file} now"]
        t["test"] = ["open file {file}", "open the document {file}", "open {file}"]

    elif intent == "open_folder":
        t["train"] = [
            "open {folder}", "open my {folder} folder", "go to {folder}", "show me the {folder} folder",
            "please open {folder}", "open the {folder} directory",
        ]
        t["val"] = ["open the {folder} folder", "show me {folder}", "please open {folder}"]
        t["test"] = ["open {folder} now", "bring up the {folder} folder", "open folder {folder}"]

    elif intent == "run_command":
        t["train"] = [
            "run {cmd}", "execute {cmd}", "please run {cmd}", "start {cmd}", "launch {cmd}",
            "execute the command {cmd}", "run this: {cmd}", "please execute {cmd}",
        ]
        t["val"] = ["execute command {cmd}", "run the command {cmd}", "please run {cmd}"]
        t["test"] = ["run {cmd} now", "please execute {cmd}", "execute {cmd} now"]

    return t


def fill_template(template: str):
    return template.format(
        app=random.choice(APPS),
        website=random.choice(WEBSITES),
        query=random.choice(SEARCH_QUERIES),
        file=random.choice(FILES),
        folder=random.choice(FOLDERS),
        key=random.choice(KEYS),
        keys=random.choice(HOTKEYS),
        text=random.choice([
            "hello world", "test message", "this is a test", "the quick brown fox",
            "remember to buy milk", "call mom", "meeting at 3pm", "open the door",
            "lorem ipsum dolor sit amet", "notes for the project", "save the file",
        ]),
        cmd=random.choice(COMMANDS),
    )


def programmatic_variants_for_intent(intent, base_templates):
    """Return an expanded list of templates for intents where templates lack diversity.
    We favor quality: polite/casual/wake-word prefixes, optional intensifiers/fillers,
    and for volume intents numeric percent deltas. For `press_key` and `open_file`
    we return templates with placeholders replaced by realistic keys/files so that
    each generated example mentions a concrete key or filename.
    """
    variants = set()
    prefixes = ["", "please ", "please, ", "could you ", "could you please ", "hey jarvis ", "jarvis ", "hey "]
    suffixes = ["", " please", " for me", " now", " now please"]
    fillers = ["", " uh", " um", ", if you can", " if possible"]

    if intent in ("volume_up", "volume_down"):
        if intent == "volume_up":
            cores = [
                "turn it up", "turn the volume up", "raise the volume",
                "make it louder", "increase the volume", "turn the sound up",
                "raise it", "increase the sound", "boost the volume",
            ]
        else:
            cores = [
                "turn it down", "turn the volume down", "lower the volume",
                "make it quieter", "decrease the volume", "turn the sound down",
                "lower it", "decrease the sound", "reduce the volume",
            ]
        natural_prefixes = ["", "please ", "could you ", "Jarvis, ", "hey Jarvis, "]
        modifiers = ["", " a bit", " slightly", " a little", " by 5 percent", " by 10 percent", " by 20 percent"]
        natural_suffixes = ["", " please", " for me", " now", " if possible"]
        for core in cores:
            for prefix in natural_prefixes:
                for modifier in modifiers:
                    for suffix in natural_suffixes:
                        if prefix.strip().lower() == "please" and suffix == " please":
                            continue
                        if core in ("make it louder", "make it quieter") and modifier in (" a bit", " slightly", " a little"):
                            adjective = core.rsplit(" ", 1)[1]
                            text = f"{prefix}make it{modifier} {adjective}{suffix}".strip()
                        else:
                            text = f"{prefix}{core}{modifier}{suffix}".strip()
                        variants.add(" ".join(text.split()))

    elif intent in ("mute_volume", "unmute_volume"):
        extras = ["", " please", " now", " for me", " quickly"]
        for t in base_templates:
            for p in prefixes:
                for e in extras:
                    for f in fillers:
                        variants.add(" ".join(f"{p}{t}{e}{f}".split()))

    elif intent in ("minimize_window", "maximize_window", "close_window", "take_screenshot"):
        for t in base_templates:
            for p in prefixes:
                for s in suffixes:
                    for f in fillers:
                        variants.add(" ".join(f"{p}{t}{s}{f}".split()))
        # add a few alternative verbs
        if intent == "minimize_window":
            alt = ["shrink the window", "send it to the tray", "hide this window"]
        elif intent == "maximize_window":
            alt = ["make it full screen", "expand to full screen", "zoom window"]
        elif intent == "close_window":
            alt = ["close it", "shut this window", "terminate the window"]
        else:
            alt = ["capture screen", "save a screenshot", "grab a screenshot"]
        for a in alt:
            for p in prefixes:
                for s in suffixes:
                    variants.add(" ".join(f"{p}{a}{s}".split()))

    elif intent == "press_key":
        # expand each template with every realistic key name
        for t in base_templates:
            # detect placeholder {key} and replace with each key
            if "{key}" in t:
                for k in KEYS:
                    for p in prefixes:
                        # avoid prefix duplication when template already begins with same phrase
                        if p.strip() and t.lower().lstrip().startswith(p.strip()):
                            continue
                        for s in suffixes:
                            # avoid suffix duplication
                            formatted = t.format(key=k)
                            if s.strip() and formatted.lower().rstrip().endswith(s.strip()):
                                continue
                            for f in fillers:
                                text = f"{p}{formatted}{s}{f}"
                                variants.add(" ".join(text.split()))
            else:
                for p in prefixes:
                    for s in suffixes:
                        for f in fillers:
                            variants.add(" ".join(f"{p}{t}{s}{f}".split()))

    elif intent == "open_file":
        for t in base_templates:
            if "{file}" in t:
                for file in FILES + ["budget.xlsx", "photo.png", "project.zip", "main.py", "notes_old.txt", "todo.md", "invoice_2026.pdf"]:
                    for p in prefixes:
                        for s in suffixes:
                            for f in fillers:
                                variants.add(" ".join(f"{p}{t.format(file=file)}{s}{f}".split()))
            else:
                for p in prefixes:
                    for s in suffixes:
                        for f in fillers:
                            variants.add(" ".join(f"{p}{t}{s}{f}".split()))

    else:
        # default conservative expansion
        for t in base_templates:
            for p in prefixes:
                for s in suffixes:
                    variants.add(" ".join(f"{p}{t}{s}".split()))

    # Keep order stable by returning a sorted list
    return sorted(variants)


def apply_noise(text: str):
    # Lightweight English-only variations
    prefixes = ["", "", "jarvis ", "hey jarvis ", "please ", "could you "]
    suffixes = ["", "", " please", " for me"]
    fillers = ["", "uh ", "hey "]

    def collapse_repeats(s: str) -> str:
        # Collapse repeated single-word duplicates: "please please" -> "please"
        s = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", s, flags=re.I)
        # Collapse repeated two-word duplicates: "could you could you" -> "could you"
        s = re.sub(r"\b(\w+\s+\w+)(?:\s+\1\b)+", r"\1", s, flags=re.I)
        # Collapse repeated multi-word small phrases (up to 4 words)
        s = re.sub(r"\b((?:\w+\s+){2,3}\w+)(?:\s+\1\b)+", r"\1", s, flags=re.I)
        # Remove doubled suffixes like "for me for me"
        s = re.sub(r"(for me)(?:\s+\1)+", r"\1", s, flags=re.I)
        # Collapse obvious "hey hey" variants
        s = re.sub(r"\b(hey)(?:\s+\1\b)+", r"\1", s, flags=re.I)
        return s

    # Choose at most one prefix/filler/suffix and avoid creating duplicates
    if random.random() < 0.18:
        p = random.choice(prefixes)
        has_leadin = re.match(r"^(?:please\b|could you\b|hey\b|jarvis\b)", text, re.I)
        if p.strip() and not has_leadin:
            text = p + text
        elif not p.strip():
            text = p + text
    if random.random() < 0.12:
        f = random.choice(fillers)
        has_leadin = re.match(r"^(?:please\b|could you\b|hey\b|jarvis\b)", text, re.I)
        if f.strip() and not has_leadin:
            text = f + text
        elif not f.strip():
            text = f + text
    if random.random() < 0.14:
        s = random.choice(suffixes)
        if s.strip() and not text.lower().rstrip().endswith(s.strip()):
            text = text + s
        elif not s.strip():
            text = text + s
    if random.random() < 0.10:
        text = text.replace("?", "").replace("!", "")

    text = " ".join(text.split()).strip()
    text = collapse_repeats(text)
    return text


def generate_split_for_intent(intent, split_name, count, used_texts):
    base_templates = templates_for_intent(intent)[split_name]
    generated = []
    used_local = set()

    # Use programmatic expansion for selected low-variation intents
    if intent in ("volume_up", "volume_down", "mute_volume", "unmute_volume",
                  "minimize_window", "maximize_window", "close_window", "take_screenshot",
                  "press_key", "open_file"):
        variants = programmatic_variants_for_intent(intent, base_templates)
        if intent == "press_key":
            random.shuffle(variants)
        # Fill templates (for any remaining placeholders) and apply noise
        for v in variants:
            text = v
            # If placeholders remain (unlikely), fill them
            if "{" in text:
                text = fill_template(text)
            if intent not in ("volume_up", "volume_down"):
                text = apply_noise(text)
            if text and text not in used_texts and text not in used_local:
                generated.append(text)
                used_local.add(text)
                used_texts.add(text)
            if len(generated) >= count:
                break

    else:
        templates = base_templates
        attempts = 0
        max_attempts = count * 50
        gen_set = set()
        while len(gen_set) < count and attempts < max_attempts:
            attempts += 1
            template = random.choice(templates)
            text = fill_template(template)
            text = apply_noise(text)
            if text and text not in used_texts:
                gen_set.add(text)
                used_texts.add(text)
        generated = list(gen_set)

    if len(generated) < count:
        print(f"WARNING: {intent} {split_name}: requested {count}, generated {len(generated)}")

    return generated


def main():
    intents = [
        "open_application", "close_application", "open_website", "web_search",
        "youtube_search", "volume_up", "volume_down", "mute_volume", "unmute_volume",
        "type_text", "press_key", "hotkey", "switch_window", "minimize_window",
        "maximize_window", "close_window", "take_screenshot", "open_file", "open_folder",
        "run_command",
    ]

    stats = Counter()
    used_texts = set()

    train_examples = []
    val_examples = []
    test_examples = []

    for intent in intents:
        train = generate_split_for_intent(intent, "train", TRAIN_PER_INTENT, used_texts)
        val = generate_split_for_intent(intent, "val", VALIDATION_PER_INTENT, used_texts)
        test = generate_split_for_intent(intent, "test", TEST_PER_INTENT, used_texts)

        train_examples.extend({"text": t, "label": intent} for t in train)
        val_examples.extend({"text": t, "label": intent} for t in val)
        test_examples.extend({"text": t, "label": intent} for t in test)

        stats[intent + "_train"] = len(train)
        stats[intent + "_val"] = len(val)
        stats[intent + "_test"] = len(test)

    # Shuffle
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    random.shuffle(test_examples)

    # Write files
    def write_jsonl(path, examples):
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    write_jsonl(OUTPUT_DIR / "train.jsonl", train_examples)
    write_jsonl(OUTPUT_DIR / "validation.jsonl", val_examples)
    write_jsonl(OUTPUT_DIR / "test.jsonl", test_examples)

    # Stats
    total = len(train_examples) + len(val_examples) + len(test_examples)
    duplicates = 0  # we prevented exact duplicates across splits
    overlap = 0

    print("Generation stats:")
    print(f"Intents: {len(intents)}")
    print(f"Train size: {len(train_examples)}")
    print(f"Validation size: {len(val_examples)}")
    print(f"Test size: {len(test_examples)}")
    print(f"Total examples: {total}")

    for k, v in stats.items():
        print(f"{k}: {v}")

    # Additional diagnostics requested by the user
    # Duplicate count across all examples
    all_texts = [e["text"] for e in train_examples + val_examples + test_examples]
    unique_texts = set(all_texts)
    duplicates = len(all_texts) - len(unique_texts)
    print(f"Duplicate examples across splits: {duplicates}")

    # Under-target intents
    under_target = []
    for intent in intents:
        if stats[intent + "_train"] < TRAIN_PER_INTENT or stats[intent + "_val"] < VALIDATION_PER_INTENT or stats[intent + "_test"] < TEST_PER_INTENT:
            under_target.append((intent, stats[intent + "_train"], stats[intent + "_val"], stats[intent + "_test"]))

    if under_target:
        print("\nUnder-target intents (train,val,test):")
        for it, tr, va, te in under_target:
            print(f" - {it}: {tr},{va},{te}")
    else:
        print("\nAll intents met or exceeded requested targets.")


if __name__ == '__main__':
    main()
