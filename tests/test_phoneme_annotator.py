"""Tests for CMU-based phoneme annotation and per-syllable stress extraction.

The stress cases here are the canonical regression tests for the bug where
stress was compared against a PHONEME index instead of a SYLLABLE index:
  "silence"  (S AY1 L AH0 N S)   must be "10", not "01"
  "guitar"   (G IH0 T AA1 R)     must be "01", not "10"
  "banana"   (B AH0 N AE1 N AH0) must be "010", not "100"
"""

import pytest

from src.data.phoneme_annotator import (
    annotate_line,
    count_syllables_rule,
    get_word_phoneme,
    syllable_stress_from_phones,
)


@pytest.mark.parametrize(
    "word, expected",
    [
        ("silence", "10"),   # trochee: SI-lence
        ("guitar", "01"),    # iamb: gui-TAR
        ("banana", "010"),   # amphibrach: ba-NA-na
    ],
)
def test_word_syllable_stress(word, expected):
    wp = get_word_phoneme(word)
    assert wp.from_cmu, f"{word} should be in the CMU dictionary"
    assert wp.syllable_stress == expected
    assert len(wp.syllable_stress) == wp.syllable_count


@pytest.mark.parametrize(
    "word, expected",
    [
        ("silence", "10"),
        ("guitar", "01"),
        ("banana", "010"),
    ],
)
def test_line_stress_pattern_single_word(word, expected):
    assert annotate_line(word).stress_pattern == expected


def test_line_stress_pattern_multiword():
    ann = annotate_line("silence guitar")
    assert ann.stress_pattern == "1001"
    assert ann.total_syllables == 4


def test_syllable_stress_from_phones_direct():
    assert syllable_stress_from_phones(["S", "AY1", "L", "AH0", "N", "S"]) == "10"
    assert syllable_stress_from_phones(["G", "IH0", "T", "AA1", "R"]) == "01"
    # secondary stress counts as stressed
    assert syllable_stress_from_phones(["AH2", "N", "D", "ER0", "S", "T", "AE1", "N", "D"]) == "101"
    # no vowels -> empty
    assert syllable_stress_from_phones(["S", "T"]) == ""


def test_stress_pattern_length_matches_syllables():
    line = "I been movin' in silence, they can't feel my weight"
    ann = annotate_line(line)
    assert len(ann.stress_pattern) == ann.total_syllables
    assert set(ann.stress_pattern) <= {"0", "1"}


def test_end_phoneme_extraction():
    ann = annotate_line("they can't feel my weight")
    # "weight" = W EY1 T -> rhyme part is the final vowel + coda
    assert ann.end_phoneme == "EY1 T"


def test_unknown_word_fallback():
    wp = get_word_phoneme("blorptastic")
    assert not wp.from_cmu
    assert wp.syllable_count >= 1
    assert len(wp.syllable_stress) == wp.syllable_count
    assert wp.syllable_stress[0] == "1"  # initial-stress heuristic


def test_count_syllables_rule():
    assert count_syllables_rule("cat") == 1
    assert count_syllables_rule("table") == 2
    assert count_syllables_rule("") == 0
