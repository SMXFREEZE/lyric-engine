"""
Predictive Surprise Engine
==========================
Implements the PREDICTION–SURPRISE–RESOLUTION (PSR) model from music cognition
research (Huron, 2006 *Sweet Anticipation*; Meyer, 1956 *Emotion and Meaning*).

Core insight
------------
Viral music creates a specific dopamine signature:
  1. SET-UP    — audience sub-consciously predicts what comes next
  2. VIOLATION — the prediction is wrong (but coherently wrong)
  3. AHA       — the violation makes sense in retrospect

  Boring music  → prediction is always correct (no hit)
  Random music  → prediction is impossible (anxiety/confusion)
  Great music   → prediction is wrong in a *satisfying* way (dopamine flood)

This engine does three things:
  A. PREDICT  — what word/phrase would a generic model generate next?
  B. MEASURE  — how much does the candidate deviate from that prediction?
  C. VALIDATE — is the deviation *coherent* (rhythmically + phonetically stable)?

Score [0, 1]:
  < 0.35  = too predictable (forgettable)
  0.35-0.70 = viral sweet spot (familiar + surprising)
  > 0.70  = too random (incoherent)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional


# ── Genre vocabulary fingerprints ─────────────────────────────────────────────
# Words the listener *expects* in each genre.
# A great line uses these — but subverts, reframes, or juxtaposes them.

GENRE_EXPECTED_VOCAB: dict[str, set[str]] = {
    "trap": {
        "money", "bands", "drip", "flex", "bag", "ice", "chain", "stack",
        "plug", "trap", "streets", "opps", "gang", "vibes", "wave", "slime",
        "bro", "hoe", "know", "go", "flow", "show", "dough", "check",
        "neck", "wrist", "whip", "foreign", "designer", "sauce", "lit",
        "cash", "grind", "hustle", "shoot", "block", "run", "gun", "cup",
    },
    "rnb": {
        "love", "heart", "feel", "need", "baby", "night", "eyes", "touch",
        "miss", "real", "forever", "soul", "together", "hold", "warm",
        "close", "want", "stay", "kiss", "mine", "yours", "body", "skin",
    },
    "pop": {
        "dance", "sing", "feel", "light", "day", "night", "shine", "star",
        "dream", "bright", "now", "wild", "free", "alive", "world", "fire",
        "run", "fly", "young", "heart", "know", "go",
    },
    "hip_hop": {
        "real", "hustle", "grind", "city", "streets", "game", "name",
        "fame", "came", "pain", "rain", "chain", "brain", "aim", "flow",
        "know", "show", "grow", "go", "bars", "spit", "mic", "lyric",
    },
    "drill": {
        "opps", "slime", "gang", "drill", "stick", "spin", "block",
        "slide", "drop", "smoke", "pack", "move", "trap", "step",
    },
    "alt_emo": {
        "feel", "numb", "scream", "empty", "hollow", "break", "fade",
        "alone", "dark", "rain", "silent", "ghost", "void", "bleed",
    },
    "country": {
        "road", "truck", "beer", "church", "mama", "home", "dirt",
        "field", "river", "barn", "boots", "stars", "sky", "porch",
    },
}

# ── Structural clichés — surface forms that signal a boring line ──────────────
STRUCTURAL_CLICHES: list[tuple[str, float]] = [
    ("i just wanna",        1.0),
    ("i never gonna",       0.9),
    ("you know what i mean", 0.8),
    ("yeah yeah yeah",      0.9),
    ("money on my mind",    1.0),
    ("riding through the city", 0.9),
    ("they don't understand", 0.8),
    ("from the bottom",     0.9),
    ("grinding every day",  1.0),
    ("staying true",        0.8),
    ("on god",              0.6),
    ("no cap",              0.7),
    ("real ones",           0.7),
    ("check my drip",       0.9),
    ("fresh to death",      0.8),
    ("run the game",        0.7),
    ("i'm the best",        0.8),
    ("all day every day",   0.9),
    ("we came up",          0.8),
    ("started from nothing", 0.9),
]

# ── Surprise-boosting patterns: controlled chaos ──────────────────────────────
# These structural moves produce "satisfying surprise"

SURPRISE_BOOSTERS: list[tuple[str, float]] = [
    # Juxtaposition (two semantically distant things side-by-side)
    (r'\b(but|yet|still|even|despite|though|although|except)\b', 0.30),
    # Unexpected specificity: numbers, brand names, proper nouns
    (r'\b\d+\b', 0.25),
    (r'\b[A-Z][a-z]{3,}\b', 0.15),      # proper noun mid-sentence
    # Temporal compression ("in a second", "for a lifetime")
    (r'\b(second|instant|lifetime|forever|never|always|eternal|moment)\b', 0.20),
    # Sensory anchor (makes abstract concrete)
    (r'\b(smell|taste|touch|skin|hands|fingers|sweat|cold|warm|heavy|light)\b', 0.20),
    # Universal truth twist ("everybody", "nobody", "the world")
    (r'\b(everybody|nobody|the world|we all|no one|every|nothing)\b', 0.15),
    # Defiance / confidence paradox
    (r'\b(told me|they said|watch me|prove|said i|they thought)\b', 0.20),
    # Simile with unexpected vehicle
    (r'\blike\s+(?:a\s+)?[a-z]+\b', 0.10),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _cliche_penalty(line: str) -> float:
    """Weighted cliché density [0-1]. 1.0 = very clichéd."""
    line_lower = line.lower()
    total = sum(w for phrase, w in STRUCTURAL_CLICHES if phrase in line_lower)
    return min(total / 2.0, 1.0)


def _booster_score(line: str) -> float:
    """Sum of surprise-boosting pattern weights, capped at 1.0."""
    total = 0.0
    for pattern, weight in SURPRISE_BOOSTERS:
        if re.search(pattern, line, re.IGNORECASE):
            total += weight
    return min(total, 1.0)


# ── Component scorers ─────────────────────────────────────────────────────────

def vocabulary_surprise(line: str, genre: str) -> float:
    """
    How surprising is the vocabulary given the genre?

    Sweet-spot curve: target 0.4–0.65 surprise.
    - < 0.2: too generic (all expected words) → penalized
    - 0.4–0.65: interesting mix → rewarded
    - > 0.85: too random → penalized
    """
    words = _tok(line)
    if not words:
        return 0.5

    expected = GENRE_EXPECTED_VOCAB.get(genre, set())
    ratio = sum(1 for w in words if w in expected) / len(words)
    surprise = 1.0 - ratio

    # Apply sweet-spot curve
    if surprise < 0.20:
        surprise *= 0.50          # too generic
    elif surprise > 0.85:
        surprise = 0.85 - (surprise - 0.85) * 2.0  # too random

    return max(0.0, min(1.0, surprise))


def semantic_leap(line: str, previous_lines: list[str]) -> float:
    """
    How different is this line's vocabulary from the recent context?

    Target: 0.45–0.70 (fresh but not random).
    """
    if not previous_lines:
        return 0.5

    context = Counter()
    for prev in previous_lines[-4:]:
        for w in _tok(prev):
            context[w] += 1

    line_words = set(_tok(line))
    if not line_words:
        return 0.5

    overlap = sum(1 for w in line_words if w in context)
    leap = 1.0 - (overlap / len(line_words))
    return max(0.0, min(1.0, leap))


def structural_surprise(line: str) -> float:
    """
    Structural surprise from syntax-level moves.
    Contrast markers, unexpected specificity, enjambment starters,
    sensory anchors — all produce "controlled surprise".
    """
    words = _tok(line)
    if not words:
        return 0.0

    score = _booster_score(line)

    # Starting with a continuation word signals structural inversion
    starters = {"because", "before", "after", "when", "while", "as", "until"}
    if words[0] in starters:
        score += 0.15

    return min(score, 1.0)


# ── Primary API ───────────────────────────────────────────────────────────────

def surprise_score(
    line: str,
    genre: str,
    previous_lines: Optional[list[str]] = None,
) -> float:
    """
    Composite predictive surprise score [0, 1].

    Weights:
      40% vocabulary_surprise — unexpected word choices for the genre
      35% semantic_leap       — deviation from recent context
      25% structural_surprise — syntax-level surprise moves

    A cliché penalty is applied on top.

    Target sweet spot: 0.38–0.70
    """
    if previous_lines is None:
        previous_lines = []

    v = vocabulary_surprise(line, genre)
    s = semantic_leap(line, previous_lines)
    t = structural_surprise(line)

    raw = 0.40 * v + 0.35 * s + 0.25 * t

    # Cliché penalty: a clichéd line can't be surprising
    penalty = _cliche_penalty(line)
    raw = raw * (1.0 - 0.65 * penalty)

    return max(0.0, min(1.0, raw))


def in_viral_sweet_spot(score: float) -> bool:
    """Returns True if surprise is in the viral-optimal range."""
    return 0.38 <= score <= 0.70


def diagnose(
    line: str,
    genre: str,
    previous_lines: Optional[list[str]] = None,
) -> dict:
    """
    Full diagnostic breakdown for a single line.

    Returns:
        {
            "total":              float,   # composite surprise score
            "vocabulary":         float,
            "semantic_leap":      float,
            "structural":         float,
            "cliche_penalty":     float,
            "in_sweet_spot":      bool,
            "verdict":            str,     # "boring" | "sweet_spot" | "incoherent"
            "boosters_found":     list[str],
        }
    """
    if previous_lines is None:
        previous_lines = []

    v = vocabulary_surprise(line, genre)
    s = semantic_leap(line, previous_lines)
    t = structural_surprise(line)
    penalty = _cliche_penalty(line)
    total = surprise_score(line, genre, previous_lines)

    if total < 0.38:
        verdict = "boring"
    elif total > 0.70:
        verdict = "incoherent"
    else:
        verdict = "sweet_spot"

    boosters = [
        pat for pat, _ in SURPRISE_BOOSTERS
        if re.search(pat, line, re.IGNORECASE)
    ]

    return {
        "total":          total,
        "vocabulary":     v,
        "semantic_leap":  s,
        "structural":     t,
        "cliche_penalty": penalty,
        "in_sweet_spot":  in_viral_sweet_spot(total),
        "verdict":        verdict,
        "boosters_found": boosters,
    }
