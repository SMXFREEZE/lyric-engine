"""Tests for the Flow DNA engine (rhythmic fingerprint matching)."""

import pytest

from src.data.phoneme_annotator import annotate_line
from src.generation.flow_dna import (
    TRAP_FLOWS,
    _extract_stress,
    diagnose,
    flow_score,
    pick_target_flow,
    rank_all_flows,
    score_against_flow,
)


def test_extract_stress_matches_annotator():
    """flow_dna and phoneme_annotator must agree on per-syllable stress."""
    for line in ["silence", "guitar", "banana", "I been movin' in silence"]:
        assert _extract_stress(line) == annotate_line(line).stress_pattern


def test_extract_stress_regression_cases():
    assert _extract_stress("silence") == "10"
    assert _extract_stress("guitar") == "01"
    assert _extract_stress("banana") == "010"


def test_flow_templates_are_consistent():
    """Every flow's stress template must be one bar long (= syllables_per_bar)."""
    for flow in TRAP_FLOWS:
        assert len(flow.stress_template) == flow.syllables_per_bar, flow.name
        assert set(flow.stress_template) <= {"0", "1"}, flow.name
        assert 0.0 <= flow.weight <= 1.0, flow.name


@pytest.mark.parametrize(
    "section, arc, expected",
    [
        ("CHORUS", "[PEAK]", "carti_abstract"),
        ("CHORUS", "[RELEASE]", "uzi_bounce"),
        ("CHORUS", "[SETUP]", "travis_melodic"),
        ("VERSE", "[BUILD]", "triplet_ride"),
        ("VERSE", "[SETUP]", "21_monotone"),
        ("VERSE", "[REFRAME]", "kendrick_switch"),
        ("BRIDGE", "[BUILD]", "kendrick_switch"),
        ("PRECHORUS", "[BUILD]", "roddy_chop"),
        ("HOOK", "[PEAK]", "uzi_bounce"),
        ("UNKNOWN", "[???]", "triplet_ride"),
    ],
)
def test_pick_target_flow(section, arc, expected):
    assert pick_target_flow(section, arc).name == expected


def test_flow_score_bounds():
    score = flow_score("I been movin' in silence, they can't feel my weight")
    assert 0.0 <= score <= 1.0


def test_score_against_flow_shape():
    result = score_against_flow("gold chains and fast lanes", TRAP_FLOWS[0])
    assert set(result) == {"syllable_fit", "stress_fit", "density_fit", "total"}
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_rank_all_flows():
    ranking = rank_all_flows("running with the night, chasing every light")
    assert len(ranking) == len(TRAP_FLOWS)
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True)


def test_diagnose_keys():
    d = diagnose("I been movin' in silence", section="VERSE", arc_token="[BUILD]")
    assert d["target_flow"] == "triplet_ride"
    assert d["actual_stress"] == annotate_line("I been movin' in silence").stress_pattern
    assert len(d["ranking"]) == 4
