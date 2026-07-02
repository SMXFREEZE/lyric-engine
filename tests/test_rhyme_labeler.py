"""Tests for phoneme-distance rhyme detection and scheme labeling."""

from src.data.phoneme_annotator import annotate_line
from src.data.rhyme_labeler import (
    detect_scheme,
    internal_rhyme_density,
    phoneme_edit_distance,
    rhymes,
)

AABB_VERSE = [
    "I been movin' in silence, they can't feel my weight",
    "Every step I take, yeah I'm moving with fate",
    "They say the game is cold but I turn up the heat",
    "Diamonds on my wrist while I dance to the beat",
]


def _end(line: str):
    return annotate_line(line).end_phoneme


def test_perfect_rhyme():
    assert rhymes(_end("feel my weight"), _end("moving with fate"))
    assert rhymes(_end("turn up the heat"), _end("dance to the beat"))


def test_non_rhyme():
    assert not rhymes(_end("feel my weight"), _end("dance to the beat"))


def test_phoneme_edit_distance_identity():
    assert phoneme_edit_distance("EY1 T", "EY1 T") == 0.0
    assert phoneme_edit_distance(None, "EY1 T") == 1.0


def test_edit_distance_ignores_stress_digits():
    assert phoneme_edit_distance("EY1 T", "EY0 T") == 0.0


def test_detect_scheme_aabb():
    result = detect_scheme(AABB_VERSE)
    assert result["scheme_type"] == "AABB"
    assert result["scheme_str"] == "AABB"
    assert result["rhyme_density"] == 1.0
    assert result["line_labels"] == ["A", "A", "B", "B"]


def test_detect_scheme_free():
    result = detect_scheme(["completely unrelated", "words that share", "nothing phonetic"])
    assert result["scheme_type"] == "free"


def test_internal_rhyme_density():
    assert internal_rhyme_density("cat hat bat") > 0.9
    assert internal_rhyme_density("single") == 0.0
