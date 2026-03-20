"""
Flow Controller
===============
Models Csikszentmihalyi's "flow state" and its effect on lyric generation.

THE PSYCHOLOGY
--------------
Flow state (Csikszentmihalyi, 1990) is a mental state of complete absorption
where action feels effortless and automatic.  Professional rappers describe
this as "being in the pocket" — lines arrive without conscious effort.

In cognitive terms: System 2 (deliberate, effortful) recedes and System 1
(automatic, intuitive) dominates.  This produces:
  - Higher creative risk-taking
  - Faster generation (less internal filtering)
  - Stronger thematic coherence (unconscious mind is better at holistic patterns)
  - More emotionally authentic lines

Counter-intuitively, flow state does NOT always produce the best lines.
Flow produces volume and authenticity.  The deliberate System 2 mode produces
precision and craft.  The best writers switch fluidly between both.

FLOW DETECTION
--------------
We detect flow from the quality trajectory of recent lines:
  - If recent quality scores are high AND stable → in flow
  - If quality is dropping → exit flow, increase deliberation
  - If quality suddenly spikes → brief flow entry, reduce filtering

REVISION INSTINCT
-----------------
Human writers don't replace bad lines wholesale — they mutate specific
problematic tokens while preserving what was working.

"I had nothing left to lose / now I run this city"
         ↓ bad rhyme, good first half
"I had nothing left to lose / but the scars that still remind me"

The revision instinct identifies WHICH specific failure mode is occurring
(rhyme, rhythm, valence, specificity) and requests targeted repair
rather than full regeneration.

INHIBITION OF RETURN
---------------------
A human writer's brain automatically suppresses recently used vocabulary
(Klein, 1988 — Inhibition of Return).  This is why skilled writers don't
repeat words without intention — the brain marks used words as "visited"
and routes attention away from them.

We implement this as a decaying lexical suppression map.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Flow state ────────────────────────────────────────────────────────────────

@dataclass
class FlowSnapshot:
    """Snapshot of the flow state at a given moment."""
    is_in_flow: bool
    flow_intensity: float     # [0, 1]
    recent_quality: float     # average quality of last N lines
    quality_trend: float      # positive = improving, negative = declining
    system2_pressure: float   # [0, 1] — how much deliberation is needed
    suggested_temperature: float
    suggested_beam_size: int
    description: str


class FlowController:
    """
    Tracks the writer's flow state and modulates generation parameters.

    In flow:
      - Higher temperature (less filtering, more creative risk)
      - Smaller beam (trust the first good idea)
      - Reduce metacognitive second-guessing

    Out of flow:
      - Lower temperature (more deliberate)
      - Larger beam (consider more options)
      - Increase System 2 intervention
    """

    FLOW_WINDOW       = 5      # how many recent lines to evaluate
    FLOW_THRESHOLD    = 0.62   # quality average above which flow begins
    FLOW_EXIT_SPEED   = 0.65   # how quickly flow exits (lower = stickier)

    def __init__(self, base_beam_size: int = 8):
        self._quality_history: deque[float] = deque(maxlen=self.FLOW_WINDOW)
        self._flow_intensity: float = 0.0
        self._base_beam_size = base_beam_size
        self._line_count: int = 0

    def update(self, quality_score: float) -> FlowSnapshot:
        """
        Update flow state after a line is accepted.
        quality_score should be the total_score of the accepted line.
        """
        self._quality_history.append(quality_score)
        self._line_count += 1

        if len(self._quality_history) < 2:
            return self._make_snapshot()

        recent_quality = float(np.mean(self._quality_history))
        quality_trend  = float(self._quality_history[-1] - self._quality_history[0]) / max(len(self._quality_history), 1)

        # Flow entry: sustained high quality
        if recent_quality > self.FLOW_THRESHOLD and quality_trend >= -0.05:
            target_intensity = float(np.clip(
                (recent_quality - self.FLOW_THRESHOLD) / (1.0 - self.FLOW_THRESHOLD),
                0.0, 1.0
            ))
            self._flow_intensity = self._flow_intensity * 0.7 + target_intensity * 0.3
        else:
            # Flow exit: quality declining
            self._flow_intensity *= self.FLOW_EXIT_SPEED

        self._flow_intensity = float(np.clip(self._flow_intensity, 0.0, 1.0))
        return self._make_snapshot()

    def _make_snapshot(self) -> FlowSnapshot:
        recent_quality = float(np.mean(self._quality_history)) if self._quality_history else 0.5
        quality_trend = 0.0
        if len(self._quality_history) >= 2:
            q = list(self._quality_history)
            quality_trend = (q[-1] - q[0]) / max(len(q), 1)

        fi = self._flow_intensity
        is_in_flow = fi > 0.40

        # In flow: hotter + smaller beam. Out of flow: cooler + larger beam.
        suggested_temperature = float(np.clip(0.75 + fi * 0.35, 0.6, 1.15))
        suggested_beam_size   = max(3, self._base_beam_size - round(fi * 4))

        # System 2 pressure: how much deliberate evaluation is needed
        system2_pressure = float(np.clip(
            0.3 + (1.0 - fi) * 0.5 + max(-quality_trend * 2.0, 0.0),
            0.0, 1.0
        ))

        if fi > 0.7:
            desc = "deep flow — generation automatic, trust the first instinct"
        elif fi > 0.4:
            desc = "light flow — generation fluid, some deliberation"
        elif quality_trend < -0.05:
            desc = "exiting flow — quality declining, increase System 2"
        else:
            desc = "deliberate mode — full metacognitive evaluation"

        return FlowSnapshot(
            is_in_flow=is_in_flow,
            flow_intensity=fi,
            recent_quality=recent_quality,
            quality_trend=quality_trend,
            system2_pressure=system2_pressure,
            suggested_temperature=suggested_temperature,
            suggested_beam_size=suggested_beam_size,
            description=desc,
        )

    def get_current_snapshot(self) -> FlowSnapshot:
        return self._make_snapshot()

    def force_system2(self):
        """Force exit from flow — used when a constraint is critically violated."""
        self._flow_intensity = max(0.0, self._flow_intensity - 0.4)


# ── Revision instinct ─────────────────────────────────────────────────────────

@dataclass
class RevisionTarget:
    """Describes what specifically needs fixing in a rejected candidate."""
    failure_mode: str          # "rhyme" | "rhythm" | "valence" | "specificity" | "length"
    severity: float            # [0, 1]
    suggested_repair: str      # human-readable repair instruction for the prompt
    temperature_modifier: float # adjust generation temperature for repair


class RevisionInstinct:
    """
    Identifies the specific failure mode in a bad candidate and prescribes
    targeted repair — mutation, not wholesale replacement.

    Human writers don't bin a whole line because one word is wrong.
    They fix the one word.  This models that.
    """

    def diagnose(
        self,
        candidate: str,
        score: "CandidateScore",  # forward ref — avoid circular import
        target_syllables: int,
        target_phoneme: Optional[str],
    ) -> Optional[RevisionTarget]:
        """
        Diagnose the primary failure mode of a rejected candidate.
        Returns None if the line is acceptable (shouldn't be revised).
        """
        issues: list[tuple[str, float, str, float]] = []
        # (failure_mode, severity, repair_hint, temp_modifier)

        # Rhyme failure — most important constraint
        if target_phoneme and score.phonetic_score < 0.35:
            issues.append((
                "rhyme",
                1.0 - score.phonetic_score,
                f"end the line with a word rhyming with /{target_phoneme}/",
                -0.15,   # cooler for rhyme precision
            ))

        # Rhythm failure — syllable count off
        if not score.syllable_ok:
            ann_syl = getattr(score, "_actual_syllables", 0)
            delta = abs(ann_syl - target_syllables) if ann_syl > 0 else 0
            issues.append((
                "rhythm",
                float(np.clip(delta / 4.0, 0.0, 1.0)),
                f"write approximately {target_syllables} syllables",
                -0.10,
            ))

        # Valence failure — wrong emotional direction
        if score.valence_fit < 0.35:
            issues.append((
                "valence",
                1.0 - score.valence_fit,
                "shift the emotional tone to match the section's target feeling",
                0.05,    # slightly hotter to explore emotional space
            ))

        # Specificity failure — too generic
        if score.introspection < 0.25 and score.novelty_score < 0.30:
            issues.append((
                "specificity",
                0.6,
                "add a specific concrete image or personal detail",
                0.10,    # hotter for specificity
            ))

        if not issues:
            return None

        # Return the highest-severity issue
        primary = max(issues, key=lambda x: x[1])
        return RevisionTarget(
            failure_mode=primary[0],
            severity=primary[1],
            suggested_repair=primary[2],
            temperature_modifier=primary[3],
        )

    def build_repair_prompt(self, original_prompt: str, target: RevisionTarget) -> str:
        """
        Augment the generation prompt with a specific repair instruction.
        The model sees this as additional context before generating the fix.
        """
        repair_token = f"[REPAIR: {target.suggested_repair}]"
        return original_prompt.rstrip("\n") + f"\n{repair_token}\n"


# ── Inhibition of return ──────────────────────────────────────────────────────

class InhibitionOfReturn:
    """
    Tracks recently used vocabulary and suppresses its reuse.
    Models the cognitive mechanism that prevents verbal perseveration.

    Activation decays over 6–8 lines, allowing words to return
    after enough time has passed — mirrors human memory dynamics.
    """

    HALF_LIFE_LINES = 5   # how many lines until suppression halves

    STOP_WORDS = {
        "i", "you", "he", "she", "we", "they", "it", "me", "my", "your",
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "have", "had", "do", "did", "not", "no", "so",
        "that", "this", "just", "like", "get", "got", "will", "can",
    }

    def __init__(self):
        self._suppression: dict[str, float] = {}   # word → suppression level [0, 1]
        self._line_count: int = 0

    def mark_used(self, line: str):
        """Mark all content words in a line as recently used."""
        self._line_count += 1
        words = [
            w.lower().strip(".,!?'\"")
            for w in re.findall(r"[A-Za-z']+", line)
            if w.lower() not in self.STOP_WORDS and len(w) > 3
        ]
        # Apply decay first
        self._decay()
        # Mark new words
        for word in words:
            self._suppression[word] = 1.0

    def suppression_score(self, candidate: str) -> float:
        """
        Return a suppression score [0, 1] for a candidate line.
        0 = no suppressed words used (good)
        1 = heavily uses recently suppressed vocabulary (bad)
        """
        words = [
            w.lower().strip(".,!?'\"")
            for w in re.findall(r"[A-Za-z']+", candidate)
            if w.lower() not in self.STOP_WORDS and len(w) > 3
        ]
        if not words:
            return 0.0
        total = sum(self._suppression.get(w, 0.0) for w in words)
        return float(np.clip(total / len(words), 0.0, 1.0))

    def novelty_bonus(self, candidate: str) -> float:
        """
        Inverse of suppression — bonus for introducing fresh vocabulary.
        """
        return 1.0 - self.suppression_score(candidate)

    def _decay(self):
        """Apply exponential decay based on line count."""
        decay_factor = 0.5 ** (1.0 / self.HALF_LIFE_LINES)
        self._suppression = {
            word: level * decay_factor
            for word, level in self._suppression.items()
            if level * decay_factor > 0.05
        }
