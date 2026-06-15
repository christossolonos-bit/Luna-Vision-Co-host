from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MoodParams:
    temperature: float = 0.75
    speed: float = 1.0
    top_p: float = 0.85
    repetition_penalty: float = 10.0


MOOD_PRESETS: dict[str, MoodParams] = {
    "neutral": MoodParams(),
    "excitement": MoodParams(temperature=0.88, speed=1.12, top_p=0.92, repetition_penalty=8.0),
    "excited": MoodParams(temperature=0.88, speed=1.12, top_p=0.92, repetition_penalty=8.0),
    "anger": MoodParams(temperature=0.82, speed=1.08, top_p=0.78, repetition_penalty=12.0),
    "angry": MoodParams(temperature=0.82, speed=1.08, top_p=0.78, repetition_penalty=12.0),
    "sadness": MoodParams(temperature=0.62, speed=0.82, top_p=0.75, repetition_penalty=10.5),
    "sad": MoodParams(temperature=0.62, speed=0.82, top_p=0.75, repetition_penalty=10.5),
    "happy": MoodParams(temperature=0.84, speed=1.06, top_p=0.9, repetition_penalty=9.0),
    "joy": MoodParams(temperature=0.86, speed=1.1, top_p=0.93, repetition_penalty=8.5),
    "curiosity": MoodParams(temperature=0.78, speed=0.96, top_p=0.88, repetition_penalty=9.5),
    "curious": MoodParams(temperature=0.78, speed=0.96, top_p=0.88, repetition_penalty=9.5),
    "flirty": MoodParams(temperature=0.8, speed=1.0, top_p=0.87, repetition_penalty=9.0),
    "seductive": MoodParams(temperature=0.76, speed=0.98, top_p=0.82, repetition_penalty=10.5),
}

MOOD_ALIASES = {
    "neutral": "neutral",
    "excitement": "excitement",
    "excited": "excitement",
    "anger": "anger",
    "angry": "anger",
    "sadness": "sadness",
    "sad": "sadness",
    "happy": "happy",
    "joy": "joy",
    "curiosity": "curiosity",
    "curious": "curiosity",
    "flirty": "flirty",
    "seductive": "seductive",
}

MOOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "anger": (
        "angry",
        "mad",
        "furious",
        "rage",
        "hate",
        "damn",
        "wtf",
        "unfair",
        "idiot",
    ),
    "sadness": (
        "sad",
        "sorry",
        "miss",
        "cry",
        "hurt",
        "loss",
        "unlucky",
        "rip",
        "died",
    ),
    "excitement": (
        "wow",
        "insane",
        "hype",
        "let's go",
        "pog",
        "clutch",
        "pentakill",
        "amazing",
    ),
    "joy": ("yay", "love", "awesome", "great", "win", "won", "celebrate", "happy"),
    "happy": ("nice", "good job", "well done", "sweet", "fun"),
    "curiosity": ("why", "how", "what", "hmm", "wonder", "curious", "really"),
    "flirty": ("cute", "handsome", "pretty", "tease", "flirt", "wink", "date"),
    "seductive": ("whisper", "close", "slow", "soft", "intimate", "velvet"),
}


def list_moods() -> list[str]:
    return [
        "neutral",
        "excitement",
        "anger",
        "sadness",
        "happy",
        "joy",
        "curiosity",
        "flirty",
        "seductive",
    ]


def normalize_mood(name: str | None, default: str = "neutral") -> str:
    if not name:
        return normalize_mood(default, "neutral")
    key = name.strip().lower()
    return MOOD_ALIASES.get(key, default if default in MOOD_PRESETS else "neutral")


def mood_params(name: str | None, default: str = "neutral") -> MoodParams:
    canonical = normalize_mood(name, default)
    return MOOD_PRESETS.get(canonical, MOOD_PRESETS["neutral"])


def detect_mood(text: str, default: str = "neutral") -> str:
    lowered = text.lower()
    scores: dict[str, int] = {}

    for mood, keywords in MOOD_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in lowered:
                score += 2 if " " in keyword else 1
        if score:
            scores[mood] = score

    if "!" in text:
        scores["excitement"] = scores.get("excitement", 0) + min(text.count("!"), 3)
    if re.search(r"\?\?+", text):
        scores["curiosity"] = scores.get("curiosity", 0) + 2

    if not scores:
        return normalize_mood(default)

    best = max(scores, key=scores.get)
    return normalize_mood(best, default)
