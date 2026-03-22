"""
Flow DNA Engine
===============
Trap music is rhythm-FIRST. Before content, before rhyme, before meaning —
there is FLOW: the rhythmic pattern of syllables against a 4/4 beat grid.

This engine encodes the rhythmic "DNA" of viral trap flows as syllable-stress
templates, then scores how well a generated line matches those fingerprints.

Key insight
-----------
The same words rapped in different flows feel completely different.
"Money" in a triplet feels different from "money" in a half-time drag.
The model must FEEL the beat, not just count syllables.

Flow is represented as a binary stress string:
  "1" = on-beat / stressed syllable
  "0" = off-beat / unstressed syllable

Canonical flows are extracted from the defining viral moments of each artist.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from src.data.phoneme_annotator import annotate_line


# ── Flow Pattern Model ────────────────────────────────────────────────────────

@dataclass
class FlowPattern:
    name: str
    description: str
    syllables_per_bar: int       # target syllable count in one 4-beat bar
    stress_template: str         # binary stress string ("10100110...")
    key_feature: str             # the distinctive rhythmic quality
    artists: list[str]           # artists who popularised this flow
    weight: float                # relevance in current trap landscape [0-1]
    section_affinity: list[str]  # sections where this flow works best
    arc_affinity: list[str]      # arc states where this flow works best


# ── Canonical Trap Flow Library ───────────────────────────────────────────────

TRAP_FLOWS: list[FlowPattern] = [
    FlowPattern(
        name="triplet_ride",
        description="Classic trap triplet — 3 syllables per beat, 12 per bar",
        syllables_per_bar=12,
        stress_template="100100100100",
        key_feature="Groupings of 3; middle syllable glides; constant forward motion",
        artists=["Migos", "Young Thug", "Future", "Offset"],
        weight=0.92,
        section_affinity=["VERSE", "PRECHORUS"],
        arc_affinity=["[BUILD]", "[RELEASE]"],
    ),
    FlowPattern(
        name="21_monotone",
        description="Half-time drag — minimal stress, flat pitch, behind-beat",
        syllables_per_bar=6,
        stress_template="101010",
        key_feature="Even spacing; constant pitch; long pauses; threat via understatement",
        artists=["21 Savage"],
        weight=0.88,
        section_affinity=["VERSE"],
        arc_affinity=["[SETUP]", "[REFRAME]"],
    ),
    FlowPattern(
        name="travis_melodic",
        description="Sung-rap hybrid — stressed syllables go melodic, vowels elongate",
        syllables_per_bar=8,
        stress_template="10100110",
        key_feature="Pitch bends on stressed syllables; melodic endpoints; auto-tune poetry",
        artists=["Travis Scott"],
        weight=0.90,
        section_affinity=["CHORUS", "HOOK", "BRIDGE"],
        arc_affinity=["[PEAK]", "[RELEASE]", "[OUTRO]"],
    ),
    FlowPattern(
        name="roddy_chop",
        description="Double-time with rhythmic chop — 16 syllables per bar",
        syllables_per_bar=16,
        stress_template="1010101010101010",
        key_feature="Rapid fire; even spacing; staccato delivery; no pauses",
        artists=["Roddy Ricch", "NBA YoungBoy", "Polo G"],
        weight=0.82,
        section_affinity=["VERSE", "PRECHORUS"],
        arc_affinity=["[BUILD]", "[PEAK]"],
    ),
    FlowPattern(
        name="carti_abstract",
        description="Abstract phonetic babbling — rhythm over semantics",
        syllables_per_bar=10,
        stress_template="1100110011",
        key_feature="Phonetic texture prioritised over meaning; almost percussive delivery",
        artists=["Playboi Carti"],
        weight=0.78,
        section_affinity=["CHORUS", "HOOK"],
        arc_affinity=["[PEAK]", "[RELEASE]"],
    ),
    FlowPattern(
        name="gunna_drip",
        description="Laid-back drip — behind-the-beat with vowel elongation",
        syllables_per_bar=9,
        stress_template="100010010",
        key_feature="Deliberately late; vowels stretched; melodic relaxation; effortlessness",
        artists=["Gunna", "Lil Baby"],
        weight=0.85,
        section_affinity=["VERSE", "BRIDGE"],
        arc_affinity=["[SETUP]", "[REFRAME]", "[OUTRO]"],
    ),
    FlowPattern(
        name="uzi_bounce",
        description="High-energy bounce — rapid alternating stress, no pause",
        syllables_per_bar=14,
        stress_template="10101010101010",
        key_feature="Infectious momentum; no breath gaps; drives chorus energy",
        artists=["Lil Uzi Vert"],
        weight=0.80,
        section_affinity=["CHORUS", "PRECHORUS"],
        arc_affinity=["[BUILD]", "[PEAK]"],
    ),
    FlowPattern(
        name="kendrick_switch",
        description="Flow switch mid-bar — changes pattern to create narrative surprise",
        syllables_per_bar=10,
        stress_template="1010011010",
        key_feature="Intentional mid-line flow change; syncopated rhythm; narrative emphasis",
        artists=["Kendrick Lamar", "J. Cole"],
        weight=0.85,
        section_affinity=["VERSE", "BRIDGE"],
        arc_affinity=["[REFRAME]", "[OUTRO]"],
    ),
]

# Quick lookup by name
_FLOW_BY_NAME: dict[str, FlowPattern] = {f.name: f for f in TRAP_FLOWS}


# ── Stress Pattern Extraction ─────────────────────────────────────────────────

def _extract_stress(line: str) -> str:
    """
    Extract binary stress pattern from a line using CMU phoneme data.
    Returns a string like "10100110".
    """
    ann = annotate_line(line)
    parts: list[str] = []
    for wp in ann.words:
        for i in range(wp.syllable_count):
            parts.append("1" if i == wp.stress else "0")
    return "".join(parts)


def _lcs_similarity(a: str, b: str) -> float:
    """
    Longest-common-subsequence similarity, normalised to [0, 1].
    More robust than positional matching for variable-length flows.
    """
    if not a or not b:
        return 0.5

    la, lb = len(a), len(b)
    # Dynamic programming LCS
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[la][lb]
    return lcs / max(la, lb)


def _positional_similarity(a: str, b: str) -> float:
    """Position-aligned match over the shorter string."""
    if not a or not b:
        return 0.5
    m = min(len(a), len(b))
    matches = sum(1 for i in range(m) if a[i] == b[i])
    return matches / max(len(a), len(b))


def _rhythmic_density(pattern: str) -> float:
    """Fraction of stressed syllables in a stress pattern."""
    if not pattern:
        return 0.5
    return pattern.count("1") / len(pattern)


# ── Scoring API ───────────────────────────────────────────────────────────────

def score_against_flow(line: str, flow: FlowPattern) -> dict:
    """
    Score a line against a specific flow pattern.

    Returns:
        {
            "syllable_fit":  float,  # how close syllable count is to target
            "stress_fit":    float,  # stress pattern match quality
            "density_fit":   float,  # stressed-syllable ratio match
            "total":         float,  # weighted composite [0-1]
        }
    """
    ann = annotate_line(line)
    actual_syl = ann.total_syllables
    actual_stress = _extract_stress(line)

    # Syllable fit: smooth Gaussian-like decay from target
    target_syl = flow.syllables_per_bar
    syl_diff = abs(actual_syl - target_syl)
    syllable_fit = math.exp(-(syl_diff ** 2) / (2 * (max(target_syl * 0.35, 2) ** 2)))

    # Stress pattern fit: blend LCS + positional
    lcs_fit = _lcs_similarity(actual_stress, flow.stress_template)
    pos_fit = _positional_similarity(actual_stress, flow.stress_template)
    stress_fit = 0.55 * lcs_fit + 0.45 * pos_fit

    # Density fit: does the line have the right proportion of stressed syllables?
    actual_density = _rhythmic_density(actual_stress)
    target_density = _rhythmic_density(flow.stress_template)
    density_fit = 1.0 - abs(actual_density - target_density)

    total = 0.40 * syllable_fit + 0.45 * stress_fit + 0.15 * density_fit

    return {
        "syllable_fit": syllable_fit,
        "stress_fit":   stress_fit,
        "density_fit":  density_fit,
        "total":        total,
    }


def pick_target_flow(section: str, arc_token: str) -> FlowPattern:
    """
    Select the most appropriate flow for a given section + arc state.

    Rules:
    ┌────────────┬──────────────┬──────────────────────┐
    │  section   │  arc_token   │  recommended flow    │
    ├────────────┼──────────────┼──────────────────────┤
    │ CHORUS     │ [PEAK]       │ carti_abstract       │
    │ CHORUS     │ [RELEASE]    │ uzi_bounce           │
    │ CHORUS     │ *            │ travis_melodic       │
    │ VERSE      │ [BUILD]      │ triplet_ride         │
    │ VERSE      │ [SETUP]      │ 21_monotone          │
    │ VERSE      │ [REFRAME]    │ kendrick_switch      │
    │ VERSE      │ *            │ gunna_drip           │
    │ BRIDGE     │ *            │ kendrick_switch      │
    │ PRECHORUS  │ *            │ roddy_chop           │
    │ *          │ *            │ triplet_ride         │
    └────────────┴──────────────┴──────────────────────┘
    """
    s = section.upper().strip()
    a = arc_token.upper().strip()

    if s == "CHORUS":
        if "[PEAK]" in a:
            return _FLOW_BY_NAME["carti_abstract"]
        if "[RELEASE]" in a:
            return _FLOW_BY_NAME["uzi_bounce"]
        return _FLOW_BY_NAME["travis_melodic"]

    if s == "VERSE":
        if "[BUILD]" in a:
            return _FLOW_BY_NAME["triplet_ride"]
        if "[SETUP]" in a:
            return _FLOW_BY_NAME["21_monotone"]
        if "[REFRAME]" in a:
            return _FLOW_BY_NAME["kendrick_switch"]
        return _FLOW_BY_NAME["gunna_drip"]

    if s == "BRIDGE":
        return _FLOW_BY_NAME["kendrick_switch"]

    if s in {"PRECHORUS", "PRE-CHORUS", "PRE_CHORUS"}:
        return _FLOW_BY_NAME["roddy_chop"]

    if s in {"HOOK"}:
        return _FLOW_BY_NAME["uzi_bounce"]

    return _FLOW_BY_NAME["triplet_ride"]


def flow_score(
    line: str,
    section: str = "VERSE",
    arc_token: str = "[BUILD]",
    target_flow: Optional[FlowPattern] = None,
) -> float:
    """
    Single-number flow quality score [0, 1].
    Uses pick_target_flow() if no explicit target provided.
    """
    if target_flow is None:
        target_flow = pick_target_flow(section, arc_token)
    result = score_against_flow(line, target_flow)
    return result["total"]


def rank_all_flows(line: str) -> list[tuple[FlowPattern, float]]:
    """
    Score the line against every flow and return them ranked best-first.
    Useful for diagnostic / user-facing explanations.
    """
    results = []
    for flow in TRAP_FLOWS:
        r = score_against_flow(line, flow)
        results.append((flow, r["total"]))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def diagnose(
    line: str,
    section: str = "VERSE",
    arc_token: str = "[BUILD]",
) -> dict:
    """
    Full flow diagnostic for a single line.

    Returns:
        {
            "target_flow":    str,    # name of the target flow
            "score":          float,  # match score against target
            "syllable_fit":   float,
            "stress_fit":     float,
            "actual_syllables": int,
            "target_syllables": int,
            "actual_stress":  str,    # binary stress pattern
            "ranking":        list,   # [(flow_name, score), ...]
        }
    """
    target = pick_target_flow(section, arc_token)
    detail = score_against_flow(line, target)
    ann = annotate_line(line)
    ranking = [(f.name, s) for f, s in rank_all_flows(line)]

    return {
        "target_flow":      target.name,
        "target_description": target.description,
        "score":            detail["total"],
        "syllable_fit":     detail["syllable_fit"],
        "stress_fit":       detail["stress_fit"],
        "density_fit":      detail["density_fit"],
        "actual_syllables": ann.total_syllables,
        "target_syllables": target.syllables_per_bar,
        "actual_stress":    _extract_stress(line),
        "target_stress":    target.stress_template,
        "key_feature":      target.key_feature,
        "artists":          target.artists,
        "ranking":          ranking[:4],  # top 4
    }
