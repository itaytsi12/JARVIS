"""Deterministic, local-only music intent parsing (no LLM call).

This is a first-class capability module in the same family as
`brain/local_planner.py` / `brain/intent_router.py` / `brain/request_intent.py`
-- it does NOT compete with `brain/router.py`, it is called BY it (see the
music block near the top of `route_command`). It classifies obvious,
Alexa-like music requests into one of the `MusicIntentType` values below and
extracts whatever entities (song/artist/album/playlist/mood/provider) the
sentence actually contains, using layered regex/keyword matching -- no
hardcoded exhaustive phrase list, no network/model call.

`route_music_command(text)` is the single function `brain.router.route_command`
calls. It turns a classified `MusicIntent` into the same route-dict shape
every other classifier in this codebase already produces (`{"type": "tool",
...}` / `{"type": "local_plan", ...}`), so everything downstream (the
speculative partial-action ledger, the parallel-independent-action executor,
recording/verification) works completely unchanged -- see
`brain/safe_tools.py` and `brain/speculative_execution.py` for why only a
small subset of the tool names below are registered as context-independent.

Apple Music Web (https://music.apple.com) is the only fully implemented
provider (see `tools/music/`). An explicit "on Spotify" / "on YouTube"
qualifier is honored by building a `local_plan` that reuses the EXISTING
generic browser-search-and-click-first-result machinery
(`tools/browser_agent.py`) instead of a second bespoke integration.
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum


class MusicIntentType(str, Enum):
    OPEN_MUSIC = "OPEN_MUSIC"
    PLAY_GENERIC = "PLAY_GENERIC"
    PLAY_SONG = "PLAY_SONG"
    PLAY_ARTIST = "PLAY_ARTIST"
    PLAY_ALBUM = "PLAY_ALBUM"
    PLAY_PLAYLIST = "PLAY_PLAYLIST"
    PLAY_QUERY = "PLAY_QUERY"  # ambiguous song/artist -- resolved by search-result scoring
    PLAY_LAST_PLAYED = "PLAY_LAST_PLAYED"
    PLAY_RECENT = "PLAY_RECENT"
    RESUME_LAST_SESSION = "RESUME_LAST_SESSION"
    PLAY_FAVORITES = "PLAY_FAVORITES"
    PLAY_LIBRARY = "PLAY_LIBRARY"
    PLAY_MOOD = "PLAY_MOOD"
    PLAY_MORE_BY_ARTIST = "PLAY_MORE_BY_ARTIST"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    STOP = "STOP"
    NEXT = "NEXT"
    PREVIOUS = "PREVIOUS"
    RESTART_TRACK = "RESTART_TRACK"
    SHUFFLE_ON = "SHUFFLE_ON"
    SHUFFLE_OFF = "SHUFFLE_OFF"
    REPEAT_TRACK = "REPEAT_TRACK"
    REPEAT_OFF = "REPEAT_OFF"
    QUEUE_TRACK = "QUEUE_TRACK"
    PLAY_NEXT = "PLAY_NEXT"
    ADD_TO_LIBRARY = "ADD_TO_LIBRARY"
    ADD_TO_FAVORITES = "ADD_TO_FAVORITES"
    NOW_PLAYING = "NOW_PLAYING"


# Tool names for the always-safe-to-fire-early / safe-to-run-concurrently
# subset (mirrors `brain/safe_tools.py::CONTEXT_INDEPENDENT_TOOLS`). Kept
# here as the single source of truth for which music tools are simple,
# reversible, single-shot player commands with no entity resolution.
FAST_PATH_TOOLS = frozenset({
    "open_music", "music_pause", "music_resume", "music_stop",
    "music_next", "music_previous",
})

PROVIDERS = {"apple music": "apple_music", "apple": "apple_music", "itunes": "apple_music",
             "spotify": "spotify", "youtube": "youtube", "youtube music": "youtube"}

_MOODS = (
    "relaxing", "relax", "chill", "chilled", "calm", "calming", "mellow",
    "workout", "work out", "gym", "exercise", "cardio",
    "study", "studying", "focus", "focusing", "concentration",
    "upbeat", "energetic", "energizing", "hype", "party", "dance",
    "sad", "happy", "romantic", "love", "sleep", "sleeping", "chilling",
    "driving", "road trip", "morning", "wake up",
)


def _strip_wrapper(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"^(?:hey\s+jarvis|jarvis)[,]?\s+", "", t)
    t = re.sub(r"^(?:can|could|would)\s+you\s+(?:please\s+)?", "", t)
    t = re.sub(r"^please\s+", "", t)
    t = re.sub(r"[.?!]+$", "", t)
    t = re.sub(r"\s+(?:please|for me)$", "", t)
    return t.strip()


@dataclass
class MusicIntent:
    intent: MusicIntentType
    raw_text: str
    song: str | None = None
    artist: str | None = None
    album: str | None = None
    playlist: str | None = None
    mood: str | None = None
    provider: str | None = None  # None == default (Apple Music); explicit otherwise
    scope: str | None = None  # e.g. "random_user_playlist"
    contextual: bool = False  # "it" / "this" -- resolve against current playback
    shuffle: bool = False
    aspect: str | None = None  # for NOW_PLAYING: "song" | "artist"
    extra: dict = field(default_factory=dict)


def _extract_provider(text: str) -> tuple[str, str | None]:
    match = re.search(r"^(.*?)\s+(?:on|using|via|in)\s+(apple music|apple|itunes|spotify|youtube music|youtube)$", text)
    if not match:
        return text, None
    remainder, provider_phrase = match.groups()
    return remainder.strip(), PROVIDERS.get(provider_phrase)


_CONTROL_PATTERNS: list[tuple[re.Pattern, MusicIntentType]] = [
    (re.compile(r"^(?:pause(?: the)? music|pause it|pause)$"), MusicIntentType.PAUSE),
    (re.compile(r"^(?:stop(?: the)? music|stop playing(?: the music)?)$"), MusicIntentType.STOP),
    (re.compile(r"^(?:resume(?: the)? music|resume it|resume playing|resume|unpause(?: it)?)$"), MusicIntentType.RESUME),
    (re.compile(r"^(?:next(?: song| track)?|skip(?: this)?(?: song| track)?)$"), MusicIntentType.NEXT),
    (re.compile(r"^(?:previous(?: song| track)?|go back a (?:song|track)|play (?:the )?previous (?:song|track))$"), MusicIntentType.PREVIOUS),
    (re.compile(r"^(?:restart(?: this| the)? (?:song|track)|replay(?: this)? song|start (?:this|the) song over)$"), MusicIntentType.RESTART_TRACK),
    (re.compile(r"^(?:turn (?:shuffle|shuffling) off|disable shuffle|shuffle off)$"), MusicIntentType.SHUFFLE_OFF),
    (re.compile(r"^(?:turn (?:shuffle|shuffling) on|enable shuffle|shuffle(?: it)?|shuffle my (?:music|songs|library|playlist))$"), MusicIntentType.SHUFFLE_ON),
    (re.compile(r"^(?:turn repeat off|disable repeat|stop repeating|repeat off)$"), MusicIntentType.REPEAT_OFF),
    (re.compile(r"^(?:repeat(?: this)? (?:song|track)|turn repeat on|enable repeat|loop(?: this)? (?:song|track)|repeat)$"), MusicIntentType.REPEAT_TRACK),
    (re.compile(r"^(?:add this(?: song)? to (?:my )?library|add to (?:my )?library|save this to (?:my )?library)$"), MusicIntentType.ADD_TO_LIBRARY),
    (re.compile(r"^(?:add this(?: song)? to (?:my )?favou?rites|favou?rite this(?: song)?|like this song|add to (?:my )?favou?rites)$"), MusicIntentType.ADD_TO_FAVORITES),
    (re.compile(r"^(?:what song is (?:this|playing)|what'?s playing|what is playing|what is this song|name this song)$"), MusicIntentType.NOW_PLAYING),
    (re.compile(r"^(?:who is (?:this|the) artist|who is this|who sings this|who made this song)$"), MusicIntentType.NOW_PLAYING),
    (re.compile(r"^(?:play more (?:songs |music )?by this artist|play more from this artist|play more like this)$"), MusicIntentType.PLAY_MORE_BY_ARTIST),
    # NOTE: bare "queue this"/"play this next" are deliberately NOT listed
    # here -- the dedicated regexes just below the control-pattern loop
    # handle every "queue <X>" / "play <X> next" phrasing (including the
    # contextual "this"/"it"/"that" forms) AND extract `song`/`contextual`,
    # which a fixed-phrase entry here would silently discard.
    (re.compile(r"^(?:open (?:the )?(?:apple )?music(?: app)?|launch (?:the )?(?:apple )?music(?: app)?|open my music)$"), MusicIntentType.OPEN_MUSIC),
]

_NOW_PLAYING_ARTIST = re.compile(r"^(?:who is (?:this|the) artist|who is this|who sings this|who made this song)$")

# ---------------------------------------------------------------------
# Hebrew (VOICE_LANGUAGE=he) -- a deliberately separate, self-contained
# pattern set rather than folded into the English cascade above. Hebrew's
# definite article ("ה") attaches directly to the following word with no
# space ("את המוזיקה" = "את" + "המוזיקה", not "את" + "ה" + " " + "מוזיקה"),
# so these patterns concatenate "ה" straight onto the noun rather than
# treating it as its own token. Entities are extracted with the SAME
# `_titlecase_span` span-recovery helper the English patterns use --
# Hebrew has no case, so `.lower()` there is a no-op and the original
# Unicode text is preserved byte-for-byte, never translated or
# transliterated.
# ---------------------------------------------------------------------
_HEBREW_CHAR = re.compile(r"[֐-׿]")


def _contains_hebrew(text: str) -> bool:
    return bool(_HEBREW_CHAR.search(text))


_HEBREW_CONTROL_PATTERNS: list[tuple[re.Pattern, MusicIntentType, str | None]] = [
    (re.compile(r"^(?:פתח|תפתח) (?:את )?ה?מוזיקה$"), MusicIntentType.OPEN_MUSIC, None),
    (re.compile(r"^(?:תעצור|עצור|תפסיק|הפסק)(?: (?:את )?ה?מוזיקה)?$"), MusicIntentType.STOP, None),
    # "continue THE MUSIC" / "continue from what I heard" (checked before
    # the bare "continue" pattern below, which must only match when there
    # is genuinely no music-context suffix -- mirrors the English
    # RESUME_LAST_SESSION vs. bare RESUME distinction).
    (re.compile(r"^(?:תמשיך|המשך) (?:את )?ה?מוזיקה$"), MusicIntentType.RESUME_LAST_SESSION, None),
    (re.compile(r"^(?:תמשיך|המשך) (?:ל|עם )?ממה ששמעתי$"), MusicIntentType.RESUME_LAST_SESSION, None),
    (re.compile(r"^(?:תמשיך|המשך)$"), MusicIntentType.RESUME, None),
    (re.compile(r"^ה?שיר (?:ה)?בא$"), MusicIntentType.NEXT, None),
    (re.compile(r"^ה?שיר (?:ה)?קודם$"), MusicIntentType.PREVIOUS, None),
    (re.compile(r"^(?:תתחיל|תפעיל) (?:את ה)?שיר מהתחלה$"), MusicIntentType.RESTART_TRACK, None),
    (re.compile(r"^מה מתנגן(?: עכשיו)?\??$|^איזה שיר (?:זה|מתנגן)\??$|^מה השיר הזה\??$"), MusicIntentType.NOW_PLAYING, "song"),
    (re.compile(r"^מי (?:שר את זה|זה השר|הזמר הזה|מבצע את זה)\??$"), MusicIntentType.NOW_PLAYING, "artist"),
    (re.compile(r"^(?:נגן|תנגן|שים) (?:קצת )?ה?מוזיקה$"), MusicIntentType.PLAY_GENERIC, None),
]


def _classify_hebrew_intent(normalized: str, original: str, provider: str | None) -> MusicIntent | None:
    text = normalized.strip().rstrip("?!.")

    for pattern, intent_type, aspect in _HEBREW_CONTROL_PATTERNS:
        if pattern.match(text):
            return MusicIntent(intent_type, original, provider=provider, aspect=aspect)

    # "play the last song I heard" -- checked before the generic "play the
    # song <X>" pattern below, which would otherwise swallow "האחרון
    # ששמעתי" ("the last one I heard") as a literal (wrong) song title.
    if re.fullmatch(r"(?:נגן|תנגן) (?:את ה)?שיר (?:ה)?אחרון(?: ש)?שמעתי|(?:נגן|תנגן) (?:את )?מה ש(?:שמעתי לאחרונה|שמעתי)", text):
        return MusicIntent(MusicIntentType.PLAY_LAST_PLAYED, original, provider=provider)

    if re.fullmatch(r"(?:נגן|תנגן) (?:משהו )?ש(?:שמעתי לאחרונה|אני שומע לאחרונה)", text):
        return MusicIntent(MusicIntentType.PLAY_RECENT, original, provider=provider)

    # "play the playlist <X>"
    m = re.fullmatch(r"(?:נגן|תנגן|שים) את ה(?:פלייליסט|רשימת ההשמעה)(?: של)? (.+)", text)
    if m:
        playlist = _titlecase_span(original, normalized, m.group(1))
        return MusicIntent(MusicIntentType.PLAY_PLAYLIST, original, playlist=playlist, provider=provider)

    # "play the song <X>" -- strip the "את השיר" wrapper so the extracted
    # entity is exactly <X>, never "את השיר <X>" (Part: preserve entities
    # exactly, e.g. "נגן את השיר שני משוגעים" -> song="שני משוגעים").
    m = re.fullmatch(r"(?:נגן|תנגן) את השיר (.+)", text)
    if m:
        song = _titlecase_span(original, normalized, m.group(1))
        return MusicIntent(MusicIntentType.PLAY_QUERY, original, song=song, provider=provider)

    # Generic "play/put on <X>" -- ambiguous song vs. artist, exactly like
    # the English PLAY_QUERY fallback: resolved by real search-result
    # scoring at execution time, never guessed here.
    m = re.fullmatch(r"(?:נגן|תנגן|שים) (.+)", text)
    if m:
        query = _titlecase_span(original, normalized, m.group(1))
        return MusicIntent(MusicIntentType.PLAY_QUERY, original, song=query, provider=provider)

    return None


def classify_music_intent(text: str) -> MusicIntent | None:
    """Return a `MusicIntent` for an obvious, deterministic music request,
    or None when `text` isn't (confidently) a music command at all -- the
    caller (`brain.router.route_command`) falls through to its normal
    routing in that case, exactly like every other local classifier here."""
    original = text
    normalized = _strip_wrapper(text)
    if not normalized:
        return None
    normalized, provider = _extract_provider(normalized)
    if not normalized:
        return None

    if _contains_hebrew(normalized):
        return _classify_hebrew_intent(normalized, original, provider)

    # "shuffle my <playlist name> playlist" (Part 9/17: play that playlist
    # AND turn shuffle on). Checked before the control-pattern loop's plain
    # "shuffle"/"shuffle my music" entries would otherwise need to, but
    # those are exact fullmatches that a named playlist never satisfies, so
    # order between the two doesn't matter -- kept here for locality with
    # the rest of the playlist-name extraction.
    m = re.fullmatch(r"shuffle (?:my |the )?(.+?) playlist", normalized)
    if m and m.group(1).strip() not in {"my", "the", ""}:
        playlist = _titlecase_span(original, normalized, m.group(1))
        return MusicIntent(MusicIntentType.PLAY_PLAYLIST, original, playlist=playlist, provider=provider, shuffle=True)

    for pattern, intent_type in _CONTROL_PATTERNS:
        if pattern.match(normalized):
            aspect = "artist" if intent_type is MusicIntentType.NOW_PLAYING and _NOW_PLAYING_ARTIST.match(normalized) else ("song" if intent_type is MusicIntentType.NOW_PLAYING else None)
            return MusicIntent(intent_type, original, provider=provider, aspect=aspect)

    # "queue <song>" / "add <song> to the queue"
    m = re.match(r"^(?:queue|add)\s+(.+?)(?:\s+to (?:the|my) queue)?$", normalized)
    if m and ("queue" in normalized):
        song = _titlecase_span(original, normalized, m.group(1))
        return MusicIntent(MusicIntentType.QUEUE_TRACK, original, song=song, provider=provider,
                            contextual=_is_contextual_reference(song))

    # "play <song/this> next"
    m = re.match(r"^play\s+(.+?)\s+next$", normalized)
    if m:
        song = _titlecase_span(original, normalized, m.group(1))
        return MusicIntent(MusicIntentType.PLAY_NEXT, original, song=song, provider=provider,
                            contextual=_is_contextual_reference(song))

    if not normalized.startswith("play") and not normalized.startswith("put on") and not normalized.startswith("resume") and not normalized.startswith("continue"):
        return None

    # RESUME_LAST_SESSION -- checked before generic PAUSE/RESUME control
    # patterns above already returned, so only compound phrasing reaches here.
    if re.search(r"\b(?:continue|resume)\b.*\b(?:my music|what i was listening to|where i left off|listening)\b", normalized) or \
       normalized in {"continue my music", "continue playing my music", "continue playing", "pick up where i left off"}:
        return MusicIntent(MusicIntentType.RESUME_LAST_SESSION, original, provider=provider)

    if not (normalized.startswith("play") or normalized.startswith("put on")):
        return None

    body = re.sub(r"^(?:play|put on)\s+", "", normalized).strip()
    if not body:
        return None

    if body in {"music", "some music", "a song", "some songs", "some tunes", "tunes", "something"}:
        return MusicIntent(MusicIntentType.PLAY_GENERIC, original, provider=provider)

    if re.fullmatch(
        r"(?:the |my )?last (?:song|track|thing)(?: i (?:listened to|played|heard))?"
        r"|(?:the |my )?last played (?:song|track)"
        r"|the song i (?:listened to|played|heard) before"
        r"|what i was listening to earlier"
        r"|what i (?:listened to|played|heard) last"
        r"|the last thing i was listening to",
        body,
    ):
        return MusicIntent(MusicIntentType.PLAY_LAST_PLAYED, original, provider=provider)

    if re.fullmatch(r"something i (?:listened to|played) recently|my recently played (?:music|songs)|something i usually listen to|my recent (?:music|songs|plays)", body):
        return MusicIntent(MusicIntentType.PLAY_RECENT, original, provider=provider)

    if re.fullmatch(r"my favou?rite playlist|my favou?rites|my favou?rite songs|my favou?rite music", body):
        return MusicIntent(MusicIntentType.PLAY_FAVORITES, original, provider=provider)

    if re.fullmatch(r"my library|my music|something from my library|from my library|my saved (?:music|songs)", body):
        return MusicIntent(MusicIntentType.PLAY_LIBRARY, original, provider=provider)

    m = re.fullmatch(r"one of my playlists|a playlist", body)
    if m:
        return MusicIntent(MusicIntentType.PLAY_PLAYLIST, original, provider=provider, scope="random_user_playlist")

    m = re.fullmatch(r"(?:my |the )?(.+?) playlist", body)
    if m:
        playlist = _titlecase_span(original, normalized, m.group(1))
        return MusicIntent(MusicIntentType.PLAY_PLAYLIST, original, playlist=playlist, provider=provider)

    mood_match = re.fullmatch(r"(?:something |some |a bit of )?(" + "|".join(re.escape(w) for w in _MOODS) + r")(?:\s+(?:music|songs|tunes|vibes))?", body)
    if mood_match:
        return MusicIntent(MusicIntentType.PLAY_MOOD, original, mood=mood_match.group(1), provider=provider)

    m = re.fullmatch(r"(?:the )?album (.+?) by (.+)", body)
    if m:
        album = _titlecase_span(original, normalized, m.group(1))
        artist = _titlecase_span(original, normalized, m.group(2))
        return MusicIntent(MusicIntentType.PLAY_ALBUM, original, album=album, artist=artist, provider=provider)

    m = re.fullmatch(r"(.+?) album by (.+)", body)
    if m:
        album = _titlecase_span(original, normalized, m.group(1))
        artist = _titlecase_span(original, normalized, m.group(2))
        return MusicIntent(MusicIntentType.PLAY_ALBUM, original, album=album, artist=artist, provider=provider)

    m = re.fullmatch(r"(.+?) by (.+)", body)
    if m:
        song = _titlecase_span(original, normalized, m.group(1))
        artist = _titlecase_span(original, normalized, m.group(2))
        return MusicIntent(MusicIntentType.PLAY_SONG, original, song=song, artist=artist, provider=provider)

    m = re.fullmatch(r"(?:artist |songs by |music by )(.+)", body)
    if m:
        artist = _titlecase_span(original, normalized, m.group(1))
        return MusicIntent(MusicIntentType.PLAY_ARTIST, original, artist=artist, provider=provider)

    if body in {"it", "this", "that"}:
        return MusicIntent(MusicIntentType.PLAY_QUERY, original, provider=provider, contextual=True)

    # Ambiguous single title/name -- resolved by real search-result scoring
    # at execution time (see tools/music/apple_music_provider.py), never
    # guessed here.
    query = _titlecase_span(original, normalized, body)
    return MusicIntent(MusicIntentType.PLAY_QUERY, original, song=query, provider=provider)


def _is_contextual_reference(text: str) -> bool:
    return bool(re.fullmatch(r"(?:it|this|that)(?: song| track)?", text.strip().lower()))


def _titlecase_span(original: str, normalized: str, lowered_fragment: str) -> str:
    """Best-effort recovery of the original casing for an extracted entity:
    find `lowered_fragment` inside the lowercased original text and slice
    the real text from that span, so "Starboy" / "The Weeknd" reach the
    search layer with their natural capitalization instead of all-lowercase
    (Apple Music's search is case-insensitive either way, but preserving
    case keeps logs/spoken confirmations readable)."""
    fragment = lowered_fragment.strip().rstrip(".?!")
    if not fragment:
        return fragment
    lowered_original = original.lower()
    index = lowered_original.find(fragment)
    if index == -1:
        return fragment
    return original[index:index + len(fragment)].strip()


# ---------------------------------------------------------------------
# Route-dict construction -- the only function `brain.router.route_command`
# calls.
# ---------------------------------------------------------------------

_SIMPLE_TOOL_BY_INTENT = {
    MusicIntentType.OPEN_MUSIC: "open_music",
    MusicIntentType.PAUSE: "music_pause",
    MusicIntentType.RESUME: "music_resume",
    MusicIntentType.STOP: "music_stop",
    MusicIntentType.NEXT: "music_next",
    MusicIntentType.PREVIOUS: "music_previous",
    MusicIntentType.RESTART_TRACK: "music_restart_track",
    MusicIntentType.SHUFFLE_ON: "music_shuffle_on",
    MusicIntentType.SHUFFLE_OFF: "music_shuffle_off",
    MusicIntentType.REPEAT_TRACK: "music_repeat_on",
    MusicIntentType.REPEAT_OFF: "music_repeat_off",
    MusicIntentType.ADD_TO_LIBRARY: "music_add_to_library",
    MusicIntentType.ADD_TO_FAVORITES: "music_add_to_favorites",
    MusicIntentType.PLAY_MORE_BY_ARTIST: "music_artist_more",
}

_PLAY_INTENTS = {
    MusicIntentType.PLAY_GENERIC, MusicIntentType.PLAY_SONG, MusicIntentType.PLAY_ARTIST,
    MusicIntentType.PLAY_ALBUM, MusicIntentType.PLAY_PLAYLIST, MusicIntentType.PLAY_QUERY,
    MusicIntentType.PLAY_LAST_PLAYED, MusicIntentType.PLAY_RECENT, MusicIntentType.RESUME_LAST_SESSION,
    MusicIntentType.PLAY_FAVORITES, MusicIntentType.PLAY_LIBRARY, MusicIntentType.PLAY_MOOD,
}


def _external_search_url(provider: str, query: str) -> str:
    q = urllib.parse.quote_plus(query)
    if provider == "youtube":
        return f"https://www.youtube.com/results?search_query={q}"
    return f"https://open.spotify.com/search/{q}"


def route_music_command(text: str) -> dict | None:
    intent = classify_music_intent(text)
    if intent is None:
        return None

    if intent.intent is MusicIntentType.NOW_PLAYING:
        return {"type": "tool", "tool": "music_now_playing", "arguments": {"aspect": intent.aspect or "song"}}

    if intent.intent is MusicIntentType.QUEUE_TRACK:
        return {"type": "tool", "tool": "music_queue_add", "arguments": {"song": intent.song, "contextual": intent.contextual}}

    if intent.intent is MusicIntentType.PLAY_NEXT:
        return {"type": "tool", "tool": "music_queue_next", "arguments": {"song": intent.song, "contextual": intent.contextual}}

    if intent.intent in _SIMPLE_TOOL_BY_INTENT:
        return {"type": "tool", "tool": _SIMPLE_TOOL_BY_INTENT[intent.intent], "arguments": {}}

    if intent.intent in _PLAY_INTENTS:
        # Explicit non-default provider: reuse the existing generic
        # browser-search-and-click-first-result plan instead of a second
        # bespoke provider integration (only Apple Music Web is fully
        # implemented -- see tools/music/).
        if intent.provider in {"spotify", "youtube"}:
            query = intent.song or intent.artist or intent.album or intent.playlist or intent.mood or intent.raw_text
            from brain.models import Action
            return {"type": "local_plan", "actions": [
                Action("browser_open_url", {"url": _external_search_url(intent.provider, query)}),
                Action("browser_click_first_result", {}),
            ]}
        resolved_intent = MusicIntentType.RESUME_LAST_SESSION if intent.intent is MusicIntentType.RESUME_LAST_SESSION else intent.intent
        return {"type": "tool", "tool": "music_play", "arguments": {
            "intent": resolved_intent.value,
            "song": intent.song, "artist": intent.artist, "album": intent.album,
            "playlist": intent.playlist, "mood": intent.mood, "scope": intent.scope,
            "contextual": intent.contextual, "shuffle": intent.shuffle,
        }}

    return None
