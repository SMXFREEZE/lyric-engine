"""
Phoneme annotation pipeline.
For each word in a lyric line: looks up CMU Pronouncing Dictionary,
falls back to rule-based estimation for unknown words.

Output per word: {word, phonemes: [...], stress: int (phoneme index of the
primary-stressed vowel, -1 if none), syllable_count: int, syllable_stress:
per-syllable binary stress string derived from CMU stress digits}.
"""

import re
from dataclasses import dataclass, asdict
from typing import Optional

import pronouncing


# ── Simple rule-based syllable counter (fallback) ──────────────────────────────

_VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)

def count_syllables_rule(word: str) -> int:
    word = word.lower().strip("'\".,!?;:-")
    if not word:
        return 0
    count = len(_VOWEL_RE.findall(word))
    if word.endswith("e") and len(word) > 2 and not word.endswith("le"):
        count = max(1, count - 1)
    return max(1, count)


# ── Phoneme lookup ─────────────────────────────────────────────────────────────

@dataclass
class WordPhoneme:
    word: str
    phonemes: list[str]       # e.g. ["M", "UW1", "V", "IH0", "NG"]
    stress: int               # primary stress position (index in PHONEMES), -1 if none
    syllable_count: int
    from_cmu: bool            # True = CMU dict hit, False = estimated
    syllable_stress: str = "" # per-SYLLABLE binary stress, e.g. "10" for "silence"


_STRESSED_DIGITS = {"1", "2"}  # primary + secondary stress both count as stressed


def syllable_stress_from_phones(phones: list[str]) -> str:
    """
    Derive a per-syllable binary stress string from CMU/ARPAbet phonemes.

    CMU marks stress as a digit suffix on VOWEL phonemes (0 = unstressed,
    1 = primary, 2 = secondary), and each vowel phoneme is one syllable:
      "silence"  S AY1 L AH0 N S    -> "10"
      "guitar"   G IH0 T AA1 R      -> "01"
      "banana"   B AH0 N AE1 N AH0  -> "010"
    Secondary stress is treated as stressed for rhythm purposes.
    """
    out = []
    for p in phones:
        if p and p[-1].isdigit():
            out.append("1" if p[-1] in _STRESSED_DIGITS else "0")
    return "".join(out)


def get_word_phoneme(word: str) -> WordPhoneme:
    clean = re.sub(r"[^a-zA-Z'']", "", word).lower()
    phones_list = pronouncing.phones_for_word(clean)

    if phones_list:
        phones = phones_list[0].split()
        syllables = pronouncing.syllable_count(phones_list[0])
        # Find primary stress (1) position (index into the PHONEME list)
        stress_pos = next(
            (i for i, p in enumerate(phones) if p.endswith("1")), -1
        )
        return WordPhoneme(
            word=word,
            phonemes=phones,
            stress=stress_pos,
            syllable_count=syllables,
            from_cmu=True,
            syllable_stress=syllable_stress_from_phones(phones),
        )
    else:
        # Rule-based fallback: assume initial stress (most common in English)
        syl = count_syllables_rule(clean)
        return WordPhoneme(
            word=word,
            phonemes=[f"~{clean.upper()}"],  # synthetic token
            stress=0,
            syllable_count=syl,
            from_cmu=False,
            syllable_stress="1" + "0" * (syl - 1) if syl > 0 else "",
        )


# ── Line-level annotation ──────────────────────────────────────────────────────

@dataclass
class LineAnnotation:
    text: str
    words: list[WordPhoneme]
    total_syllables: int
    end_phoneme: Optional[str]   # last vowel+coda — used for rhyme matching
    stress_pattern: str          # e.g. "0101" per syllable


def get_end_phoneme(phones: list[str]) -> Optional[str]:
    """Extract the rhyme-relevant vowel+coda from a phoneme list."""
    vowel_phones = [p for p in phones if any(c.isdigit() for c in p)]
    if not vowel_phones:
        return None
    last_vowel = vowel_phones[-1]
    last_vowel_idx = phones.index(last_vowel)  # type: ignore[arg-type]
    # Include all phonemes from last vowel onward (vowel + coda)
    return " ".join(phones[last_vowel_idx:])


def annotate_line(line: str) -> LineAnnotation:
    words_raw = line.split()
    word_phonemes = [get_word_phoneme(w) for w in words_raw]

    total_syl = sum(w.syllable_count for w in word_phonemes)

    # Build stress pattern string across all syllables.
    # Uses CMU stress digits on vowel phonemes (via WordPhoneme.syllable_stress),
    # so "guitar" contributes "01" and "silence" contributes "10".
    stress_chars = []
    for wp in word_phonemes:
        s = wp.syllable_stress
        if len(s) != wp.syllable_count:  # defensive: keep pattern aligned
            s = (s + "0" * wp.syllable_count)[: wp.syllable_count]
        stress_chars.append(s)
    stress_pattern = "".join(stress_chars)

    # End phoneme from last non-empty word
    end_phoneme = None
    for wp in reversed(word_phonemes):
        if wp.phonemes:
            end_phoneme = get_end_phoneme(wp.phonemes)
            break

    return LineAnnotation(
        text=line,
        words=word_phonemes,
        total_syllables=total_syl,
        end_phoneme=end_phoneme,
        stress_pattern=stress_pattern,
    )


def annotate_lyrics(lyrics: str) -> list[LineAnnotation]:
    """Annotate all non-empty lines in a lyrics block."""
    lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
    return [annotate_line(line) for line in lines]


def annotation_to_dict(ann: LineAnnotation) -> dict:
    return {
        "text": ann.text,
        "total_syllables": ann.total_syllables,
        "end_phoneme": ann.end_phoneme,
        "stress_pattern": ann.stress_pattern,
        "words": [asdict(w) for w in ann.words],
    }


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_lines = [
        "I been movin' in silence, they can't feel my weight",
        "Every step I take, yeah I'm moving with fate",
    ]
    for line in test_lines:
        ann = annotate_line(line)
        print(f"\n'{line}'")
        print(f"  syllables : {ann.total_syllables}")
        print(f"  end_phoneme: {ann.end_phoneme}")
        print(f"  stress     : {ann.stress_pattern}")
        for w in ann.words:
            marker = "←stress" if w.stress >= 0 else ""
            print(f"  {w.word:20s} {' '.join(w.phonemes):30s} syl={w.syllable_count} {marker}")
