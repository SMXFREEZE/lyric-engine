"""Tests for the metacognitive workspace: module roster, MSV switching,
candidate evaluation, and self-model learning. Pure Python — no torch."""

from src.model.metacognitive_engine import (
    DEFAULT_MODULE_WEIGHTS,
    MetacognitiveStateVector,
    MetacognitiveWorkspace,
)

EXPECTED_MODULES = {
    "phonology", "stress", "emotion", "semantic", "structure",
    "texture", "dopamine", "surprise", "flow",
}

CANDIDATES = [
    "I been movin' in silence, they can't feel my weight",
    "Gold chains and fast lanes, running with the night",
    "Remember when we had nothing but dreams in our pockets",
]


def test_workspace_has_nine_modules():
    ws = MetacognitiveWorkspace()
    assert set(ws.modules) == EXPECTED_MODULES


def test_default_weights_match_modules_and_sum_to_one():
    assert set(DEFAULT_MODULE_WEIGHTS) == EXPECTED_MODULES
    assert abs(sum(DEFAULT_MODULE_WEIGHTS.values()) - 1.0) < 1e-9


def test_evaluate_candidates_full_traces():
    ws = MetacognitiveWorkspace()
    traces = ws.evaluate_candidates(candidates=CANDIDATES, accepted_lines=[], line_idx=0)
    assert len(traces) == 3
    scores = [t.total_score for t in traces]
    assert scores == sorted(scores, reverse=True)
    for t in traces:
        assert set(t.module_scores) == EXPECTED_MODULES
        assert all(t.module_reasoning.get(m) for m in EXPECTED_MODULES)
        assert t.decision in {"ACCEPT", "REVISE", "REGENERATE"}


def test_cold_start_uses_system2():
    """No accepted lines => unfamiliar territory => deliberative mode."""
    ws = MetacognitiveWorkspace()
    traces = ws.evaluate_candidates(candidates=CANDIDATES, accepted_lines=[], line_idx=0)
    assert traces[0].system_mode == "system2"


def test_msv_switching_rules():
    # comfortable defaults -> creative flow
    assert MetacognitiveStateVector().system_mode() == "system1"
    # any single stress signal -> deliberative mode
    assert MetacognitiveStateVector(output_confidence=0.4).system_mode() == "system2"
    assert MetacognitiveStateVector(conflict_level=0.6).system_mode() == "system2"
    assert MetacognitiveStateVector(task_importance=0.9).system_mode() == "system2"
    assert MetacognitiveStateVector(experience_match=0.2).system_mode() == "system2"


def test_msv_cold_start_boundary_is_inclusive():
    """Regression: experience_match=0.3 (the cold-start value) must trip the switch."""
    assert MetacognitiveStateVector(experience_match=0.3).system_mode() == "system2"


def test_self_model_learning():
    ws = MetacognitiveWorkspace()
    traces = ws.evaluate_candidates(candidates=CANDIDATES, accepted_lines=[], line_idx=0)
    ws.accept_line(traces[0])
    report = ws.get_session_report()
    assert report["total_generations"] == 1
    assert report["system2_ratio"] == 1.0
    assert set(report["module_reliability"]) == EXPECTED_MODULES


def test_rhyme_miss_is_flagged():
    ws = MetacognitiveWorkspace()
    traces = ws.evaluate_candidates(
        candidates=["I dance to the beat"],           # IY1 T
        target_end_phoneme="EY1 T",                    # requires -ate rhyme
        accepted_lines=["they can't feel my weight"],
        line_idx=1,
    )
    assert "RHYME_MISS" in traces[0].all_flags


def test_rhyme_hit_is_not_flagged():
    ws = MetacognitiveWorkspace()
    traces = ws.evaluate_candidates(
        candidates=["I'm moving with fate"],           # EY1 T
        target_end_phoneme="EY1 T",
        accepted_lines=["they can't feel my weight"],
        line_idx=1,
    )
    assert "RHYME_MISS" not in traces[0].all_flags
