"""
Surprise Engine
===============
Implements the "unexpected but inevitable" line selection mechanism.

THE CORE INSIGHT
----------------
The best lyric lines are ones the listener never saw coming but immediately
feels was the *only* possible thing that could have been said.

This is not randomness.  This is structured violation of expectation with
satisfying resolution — what neuroscientist David Huron calls "expectation
violation with appraisal" and what musicians call "the money line."

The formula:
    surprise_value = violation × resolution

  violation  — how far the line departs from what was expected
  resolution — how satisfying / coherent it is despite the violation

  High violation + High resolution = goosebump line
  High violation + Low resolution  = confusing, bad
  Low violation  + High resolution = technically good but forgettable
  Low violation  + Low resolution  = bad on every axis

THE BUDGET SYSTEM
-----------------
Human writers don't surprise every line — they build trust through
confirmation first, then spend that trust on a single unexpected moment.

The surprise budget models this:
  - Starts near zero (build the world first)
  - Grows as tension accumulates
  - Peaks at bridge / last 2 bars of verse
  - Spends itself when a surprise line is selected
  - Resets after spending (writer must rebuild trust)

Chorus is explicitly locked to near-zero budget — the chorus MUST be
predictable because its whole function is to be the resolution the brain
has been craving.  Surprising the chorus breaks the fundamental contract.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.model.predictive_coder import Expectation, ITPRAResult, PredictiveCoder


# ── Section budget table ──────────────────────────────────────────────────────
# How much surprise license each section carries, at full tension.

SECTION_SURPRISE_BUDGET: dict[str, float] = {
    "intro":      0.05,   # introduce the world — no surprises yet
    "verse1":     0.12,   # one unexpected image is fine
    "verse2":     0.22,   # earned more trust now
    "pre_chorus": 0.15,   # building toward release
    "chorus":     0.02,   # NEVER surprise the chorus — it's the resolution
    "chorus2":    0.02,
    "hook":       0.03,
    "bridge":     0.60,   # bridge IS the surprise — maximum license
    "outro":      0.25,   # subvert the resolution
}

# After a surprise line is selected, the budget is temporarily reduced
SURPRISE_FATIGUE_FACTOR = 0.35   # budget × this after spending


@dataclass
class SurpriseDecision:
    """Result of the surprise selection process."""
    chosen_text: str
    was_surprise: bool
    surprise_probability: float
    violation_score: float
    resolution_score: float
    reason: str


class SurpriseEngine:
    """
    Controls when and how the system departs from safe/expected line selection.

    The engine maintains a running surprise budget that accumulates with tension
    and section position, then spends it on the highest-value unexpected line
    when the budget, moment, and candidate pool align.
    """

    def __init__(self):
        self._coder = PredictiveCoder()
        self._last_surprise_line: int = -999   # line index of last surprise
        self._surprise_fatigue: float = 1.0    # multiplier, reduces after spending

    def compute_budget(
        self,
        tension: float,
        section: str,
        bar_position: int,   # 0–7 within 8-bar section
        lines_since_surprise: int,
    ) -> float:
        """
        Compute the current surprise budget.

        Budget = base_section_budget × tension_factor × position_factor × recovery_factor

        Recovery factor: writers need time between surprises to rebuild trust.
        Position factor: last 2 bars of a section are the prime surprise window.
        """
        base = SECTION_SURPRISE_BUDGET.get(section.lower(), 0.10)

        # Tension factor: can only surprise if tension was built
        # Non-linear: low tension produces almost no budget even in bridge
        tension_factor = tension ** 1.4

        # Position factor: last 2 bars of any section are resolution zones
        position_factor = 1.6 if bar_position >= 6 else (1.2 if bar_position >= 4 else 1.0)

        # Recovery: must have written at least 3 lines since last surprise
        recovery_factor = float(np.clip(lines_since_surprise / 4.0, 0.0, 1.0))

        # Surprise fatigue: reduced budget after recent surprise
        budget = base * tension_factor * position_factor * recovery_factor * self._surprise_fatigue

        return float(np.clip(budget, 0.0, 0.80))

    def select(
        self,
        candidates: list[str],
        expectation: Expectation,
        tension: float,
        section: str,
        bar_position: int,
        line_idx: int,
        previous_line: Optional[str] = None,
        rng_seed: Optional[int] = None,
    ) -> SurpriseDecision:
        """
        Select the best line, potentially choosing a surprising one.

        Returns the chosen line and metadata about the decision.
        The choice is probabilistic — not deterministic — like a human writer
        who sometimes surprises themselves.
        """
        if not candidates:
            return SurpriseDecision(
                chosen_text="[no candidate generated]",
                was_surprise=False,
                surprise_probability=0.0,
                violation_score=0.0,
                resolution_score=0.0,
                reason="empty candidate pool",
            )

        lines_since_surprise = line_idx - self._last_surprise_line
        budget = self.compute_budget(tension, section, bar_position, lines_since_surprise)

        # Evaluate all candidates through ITPRA
        itpra_results: list[ITPRAResult] = self._coder.batch_evaluate(
            candidates, expectation, tension, previous_line
        )

        # Find the safe winner (highest base score — should be candidates[0])
        safe_text = candidates[0]
        safe_itpra = itpra_results[0]

        # Find the best surprise candidate: high violation × high resolution
        # Minimum resolution threshold: 0.45 — below this it's just noise
        MIN_RESOLUTION = 0.45
        MIN_VIOLATION  = 0.28   # must actually be surprising

        surprise_pool = [
            (candidates[i], itpra_results[i])
            for i in range(len(candidates))
            if (itpra_results[i].violation_score >= MIN_VIOLATION
                and itpra_results[i].resolution_score >= MIN_RESOLUTION
                and itpra_results[i].itpra_score > safe_itpra.itpra_score * 0.7)
        ]

        if not surprise_pool:
            return SurpriseDecision(
                chosen_text=safe_text,
                was_surprise=False,
                surprise_probability=budget,
                violation_score=safe_itpra.violation_score,
                resolution_score=safe_itpra.resolution_score,
                reason=f"no viable surprise candidates (budget={budget:.2f})",
            )

        # Pick best surprise: highest violation × resolution product
        best_surprise_text, best_surprise_itpra = max(
            surprise_pool,
            key=lambda x: x[1].violation_score * x[1].resolution_score
        )

        # Effective probability: budget × how good the surprise actually is
        surprise_strength = best_surprise_itpra.violation_score * best_surprise_itpra.resolution_score
        effective_prob = float(np.clip(budget * surprise_strength, 0.0, 0.75))

        # Weighted coin flip — human-like probabilistic selection
        rng = random.Random(rng_seed) if rng_seed is not None else random
        if rng.random() < effective_prob:
            # Surprise wins — apply fatigue
            self._last_surprise_line = line_idx
            self._surprise_fatigue = max(SURPRISE_FATIGUE_FACTOR, self._surprise_fatigue * 0.6)
            return SurpriseDecision(
                chosen_text=best_surprise_text,
                was_surprise=True,
                surprise_probability=effective_prob,
                violation_score=best_surprise_itpra.violation_score,
                resolution_score=best_surprise_itpra.resolution_score,
                reason=(
                    f"surprise selected (p={effective_prob:.2f}, "
                    f"tension={tension:.2f}, "
                    f"reaction={best_surprise_itpra.reaction_type})"
                ),
            )

        # Safe wins — slowly recover fatigue
        self._surprise_fatigue = min(1.0, self._surprise_fatigue + 0.08)
        return SurpriseDecision(
            chosen_text=safe_text,
            was_surprise=False,
            surprise_probability=effective_prob,
            violation_score=safe_itpra.violation_score,
            resolution_score=safe_itpra.resolution_score,
            reason=f"safe selected (budget={budget:.2f}, p_surprise={effective_prob:.2f})",
        )

    def reset_for_section(self, section: str):
        """Reset fatigue at section boundaries."""
        if section.lower() in {"chorus", "chorus2", "hook"}:
            # Lock to safe mode for chorus
            self._surprise_fatigue = 0.05
        else:
            # Partial recovery
            self._surprise_fatigue = min(1.0, self._surprise_fatigue + 0.25)
