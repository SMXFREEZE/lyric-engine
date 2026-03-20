"""
Predictive Coder
================
Implements Huron's ITPRA model of musical expectation as a lyric-generation
constraint system.

THE MODEL (Huron, 2006 — "Sweet Anticipation")
-----------------------------------------------
ITPRA = Imagination → Tension → Prediction → Reaction → Appraisal

The brain is a prediction machine.  When listening to music, it continuously
predicts what comes next.  The emotional response to music is largely the
response to those predictions being confirmed, violated, or delayed.

Applied to lyric writing:

  IMAGINATION  — "What could logically follow from these lines?"
                  The mental rehearsal of possible next moves.

  TENSION      — "How much suspense exists in this moment?"
                  Built by unresolved rhyme, rising arousal, dark content.
                  Tension creates the NEED for prediction resolution.

  PREDICTION   — "What specifically is expected?"
                  The brain locks onto the most probable completion.
                  For lyrics: predicted rhyme, predicted emotional direction,
                  predicted syllable count, predicted motif return.

  REACTION     — "Did the line match, exceed, or violate the prediction?"
                  Immediate gut response before conscious evaluation.
                  Match = satisfaction. Violation with resolution = goosebumps.
                  Violation without resolution = confusion/disappointment.

  APPRAISAL    — "In retrospect, does this feel right?"
                  Slower conscious evaluation.
                  A violated prediction can be re-appraised as brilliant
                  if it resolves into a satisfying new frame.

WHY THIS MATTERS FOR GENERATION
---------------------------------
Current scoring systems evaluate candidates in isolation.
The ITPRA model evaluates candidates in *relation to what the brain expects*.

The same line can score high in isolation and devastatingly low in context
if it fails the reaction/appraisal stages.

Or it can score mediocre in isolation but score extremely high in context
because it violates expectations in exactly the right way — producing the
goosebump moment that makes a song legendary.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.data.phoneme_annotator import annotate_line
from src.data.rhyme_labeler import rhymes
from src.data.valence_scorer import score_line
from src.model.emotional_geometry import line_emotion, get_target_point


# ── ITPRA result ──────────────────────────────────────────────────────────────

@dataclass
class ITPRAResult:
    """Full ITPRA evaluation of a candidate line."""

    # Imagination: how many plausible completions exist? (diversity of options)
    imagination_breadth: float  # [0, 1] — 0=constrained, 1=wide open

    # Tension: how much suspense existed BEFORE this line was evaluated
    tension_before: float       # [0, 1]

    # Prediction: what was specifically expected
    predicted_phoneme: Optional[str]
    predicted_valence: float
    predicted_syllables: int

    # Reaction scores
    phoneme_match: bool          # did the rhyme land as expected?
    valence_surprise: float      # |actual_valence - predicted_valence| [0, 2]
    syllable_surprise: float     # |actual - predicted| / predicted [0, 1]

    # Appraisal: retrospective quality of the violation
    violation_score: float       # [0, 1] — how unexpected was it?
    resolution_score: float      # [0, 1] — does it make sense in retrospect?
    itpra_score: float           # composite [0, 1]

    # Reaction type
    reaction_type: str           # "confirmation" | "pleasant_surprise" | "violation" | "confusion"

    def describe(self) -> str:
        return (
            f"ITPRA({self.reaction_type}) "
            f"violation={self.violation_score:.2f} "
            f"resolution={self.resolution_score:.2f} "
            f"score={self.itpra_score:.2f}"
        )


# ── Expectation builder ───────────────────────────────────────────────────────

@dataclass
class Expectation:
    """What the brain predicts will come next."""
    phoneme: Optional[str]       # expected end rhyme
    valence: float               # expected emotional direction
    arousal: float               # expected energy level
    syllables: int               # expected syllable count
    motifs: list[str]            # expected recurring words
    confidence: float            # how locked-in is the prediction [0, 1]


def build_expectation(
    accepted_lines: list[str],
    genre: str,
    section: str,
    target_end_phoneme: Optional[str],
    target_syllables: int,
) -> Expectation:
    """
    Build the brain's current prediction given accepted context.
    Confidence grows with more context (more lines = tighter prediction).
    """
    if not accepted_lines:
        target = get_target_point(genre, section)
        return Expectation(
            phoneme=target_end_phoneme,
            valence=float(target.valence),
            arousal=float(target.arousal),
            syllables=target_syllables,
            motifs=[],
            confidence=0.2,
        )

    # Predict valence: recent trajectory direction
    recent_emotions = [score_line(line) for line in accepted_lines[-4:]]
    if recent_emotions:
        avg_valence = float(np.mean([e.valence for e in recent_emotions]))
        avg_arousal = float(np.mean([e.arousal for e in recent_emotions]))
        # Project forward: if valence has been rising, brain predicts continued rise
        if len(recent_emotions) >= 2:
            valence_drift = recent_emotions[-1].valence - recent_emotions[0].valence
            predicted_valence = avg_valence + valence_drift * 0.3
        else:
            predicted_valence = avg_valence
    else:
        target = get_target_point(genre, section)
        predicted_valence = float(target.valence)
        avg_arousal = float(target.arousal)

    # Predict syllables: average of recent lines
    recent_syllables = []
    for line in accepted_lines[-4:]:
        ann = annotate_line(line)
        if ann.total_syllables > 0:
            recent_syllables.append(ann.total_syllables)
    predicted_syllables = (
        round(float(np.mean(recent_syllables))) if recent_syllables else target_syllables
    )

    # Extract motifs: words appearing >1 time in recent lines
    word_counts: dict[str, int] = {}
    stop = {"i", "you", "the", "a", "and", "or", "in", "on", "my", "your", "it", "is", "was"}
    for line in accepted_lines[-8:]:
        for w in re.findall(r"[a-z']+", line.lower()):
            if w not in stop and len(w) > 3:
                word_counts[w] = word_counts.get(w, 0) + 1
    motifs = [w for w, c in word_counts.items() if c >= 2][:6]

    # Confidence grows with context depth
    confidence = float(np.clip(len(accepted_lines) / 12.0, 0.15, 0.90))

    return Expectation(
        phoneme=target_end_phoneme,
        valence=float(np.clip(predicted_valence, -1, 1)),
        arousal=float(np.clip(avg_arousal, -1, 1)),
        syllables=predicted_syllables,
        motifs=motifs,
        confidence=confidence,
    )


# ── ITPRA evaluator ───────────────────────────────────────────────────────────

class PredictiveCoder:
    """
    Evaluates candidate lines through the full ITPRA cycle.
    The key insight: a line's quality is relative to expectations, not absolute.
    """

    def evaluate(
        self,
        candidate: str,
        expectation: Expectation,
        tension_before: float,
        previous_line: Optional[str] = None,
    ) -> ITPRAResult:
        """Run the full ITPRA evaluation on a candidate line."""

        # ── Imagination breadth ───────────────────────────────────────────
        # High tension + strong expectation = narrow imagination (constrained)
        # Low tension + weak expectation = wide imagination (open field)
        imagination_breadth = float(np.clip(
            1.0 - expectation.confidence * 0.6 - tension_before * 0.2,
            0.0, 1.0
        ))

        # ── Reaction: measure actual vs predicted ─────────────────────────
        emotion = score_line(candidate)
        ann = annotate_line(candidate)

        # Phoneme match
        phoneme_match = False
        if expectation.phoneme and ann.end_phoneme:
            phoneme_match = rhymes(ann.end_phoneme, expectation.phoneme)

        # Valence surprise: how far from predicted emotional direction
        valence_surprise = float(abs(emotion.valence - expectation.valence))

        # Syllable surprise: relative deviation from expected count
        actual_syl = max(ann.total_syllables, 1)
        expected_syl = max(expectation.syllables, 1)
        syllable_surprise = float(abs(actual_syl - expected_syl) / expected_syl)

        # ── Violation score: overall unexpectedness ───────────────────────
        rhyme_violation = 0.0 if phoneme_match else (0.7 if expectation.phoneme else 0.0)
        violation_score = float(np.clip(
            valence_surprise * 0.45
            + rhyme_violation * 0.35
            + syllable_surprise * 0.20,
            0.0, 1.0
        ))

        # ── Resolution score: does the violation make sense in retrospect? ─
        # Proxy: hook DNA strength + semantic coherence + grammatical integrity
        # A line that's unexpected but uses strong hook patterns resolves well.
        try:
            from src.model.dopamine_arc import hook_dna_score
            hook = hook_dna_score(candidate, previous_line)
        except Exception:
            hook = 0.3

        # Semantic resolution: does the line have real content words?
        content_words = [
            w for w in re.findall(r"[A-Za-z]+", candidate)
            if len(w) > 3
        ]
        semantic_density = float(np.clip(len(content_words) / 8.0, 0.0, 1.0))

        # Motif continuity: does it connect to established themes?
        candidate_words = set(candidate.lower().split())
        if expectation.motifs:
            motif_continuity = len(candidate_words & set(expectation.motifs)) / max(len(expectation.motifs), 1)
        else:
            motif_continuity = 0.5  # neutral if no motifs established yet

        resolution_score = float(np.clip(
            hook * 0.40
            + semantic_density * 0.30
            + motif_continuity * 0.30,
            0.0, 1.0
        ))

        # ── ITPRA composite score ─────────────────────────────────────────
        # The key formula: dopamine = tension × violation × resolution
        # - No tension: surprise feels random, not earned
        # - No violation: confirmation is satisfying but not memorable
        # - No resolution: violation is just confusion

        goosebump_potential = tension_before * violation_score * resolution_score

        # Pure confirmation bonus: expected line still has value
        confirmation_bonus = (1.0 - violation_score) * expectation.confidence * 0.4

        # Weighted combination
        itpra_score = float(np.clip(
            goosebump_potential * 0.60
            + confirmation_bonus * 0.20
            + resolution_score * 0.20,
            0.0, 1.0
        ))

        # ── Reaction type classification ──────────────────────────────────
        if violation_score < 0.25:
            reaction_type = "confirmation"
        elif violation_score >= 0.25 and resolution_score >= 0.55:
            reaction_type = "pleasant_surprise"
        elif violation_score >= 0.25 and resolution_score < 0.55:
            reaction_type = "violation"
        else:
            reaction_type = "confusion"

        return ITPRAResult(
            imagination_breadth=imagination_breadth,
            tension_before=tension_before,
            predicted_phoneme=expectation.phoneme,
            predicted_valence=expectation.valence,
            predicted_syllables=expectation.syllables,
            phoneme_match=phoneme_match,
            valence_surprise=valence_surprise,
            syllable_surprise=syllable_surprise,
            violation_score=violation_score,
            resolution_score=resolution_score,
            itpra_score=itpra_score,
            reaction_type=reaction_type,
        )

    def batch_evaluate(
        self,
        candidates: list[str],
        expectation: Expectation,
        tension_before: float,
        previous_line: Optional[str] = None,
    ) -> list[ITPRAResult]:
        """Evaluate all candidates at once."""
        return [
            self.evaluate(c, expectation, tension_before, previous_line)
            for c in candidates
        ]
