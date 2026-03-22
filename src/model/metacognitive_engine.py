"""
Metacognitive Engine
=====================
Transforms the lyric generator from a statistical model into a functional
creative agent that knows what it is doing and why.

This is the missing piece that separates "AI that generates text" from
"AI that composes music the way a human brain does."

THEORETICAL BASIS
-----------------
Implements four cognitive science frameworks:

[GWT] Global Workspace Theory (Baars 1988, Dehaene & Changeux)
  The brain has many specialized, parallel, unconscious modules. Consciousness
  is what happens when ONE module's output wins a competition for access to a
  shared "workspace" and is broadcast system-wide.
  -> Here: 7 specialized modules run in parallel. The workspace selects the
     winner with a justification trace and broadcasts the result.

[TRAP] Metacognitive AI Framework (arxiv 2406.12147)
  Four requirements for a system that "knows what it's doing":
  - Transparency: can represent and explain its own reasoning
  - Reasoning: synthesizes information, doesn't just pattern-match
  - Adaptation: updates strategies based on feedback and session history
  - Perception: correctly identifies which constraints are active now

[HOT] Higher Order Thought Theory of Consciousness (Rosenthal)
  A mental state is conscious only if there is a representation OF that state.
  The system must track: what was generated, which process produced it, which
  constraints were active, whether they were satisfied, and what to do next.
  -> GenerationTrace implements this explicitly.

[MSV] Metacognitive State Vector (Sethi et al. 2025)
  5 quantified signals controlling System 1/2 switching:
  - emotional_salience, output_confidence, experience_match,
    conflict_level, task_importance
  -> Normal range = System 1 (MPFC, creative flow)
  -> Conflict/low confidence/high importance = System 2 (DLPFC, deliberative)

ARCHITECTURE
------------
  +---------------------------------------------------------------+
  |              PARALLEL SPECIALIZED MODULES                      |
  | Phonology  Stress  Emotion  Semantic  Structure  Texture  Dopa |
  | (each: score, confidence, flags, reasoning)                    |
  +-----------------------------+---------------------------------+
                                |  7 parallel outputs
  +-----------------------------v---------------------------------+
  |                 GLOBAL WORKSPACE LAYER                        |
  |  - Collects all module outputs                                |
  |  - Detects conflicts (high variance across module scores)     |
  |  - Selects winner with explicit justification trace           |
  |  - Broadcasts winning state so all modules can update         |
  +-----------------------------+---------------------------------+
                                |
  +-----------------------------v---------------------------------+
  |               WORKING MEMORY STATE                            |
  |  - Full composition history (last 20 lines)                   |
  |  - Active rhyme debt / emotional temperature / bar position   |
  +-----------------------------+---------------------------------+
                                |
  +-----------------------------v---------------------------------+
  |          METACOGNITIVE MONITOR (TRAP + MSV)                   |
  |  State vector -> System 1 or System 2 decision                |
  |  Decides: ACCEPT / REVISE(target_module) / REGENERATE         |
  |  Explains WHY in human-readable reasoning trace               |
  +---------------------------------------------------------------+
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.data.phoneme_annotator import annotate_line
from src.data.rhyme_labeler import rhymes
from src.data.valence_scorer import score_line
from src.model.emotional_geometry import (
    trajectory_fit_score, line_emotion, get_target_point,
)
from src.model.phonosemantic import (
    texture_alignment_score, analyze_line_texture,
)
from src.model.dopamine_arc import (
    goosebump_potential, hook_dna_score, detect_hook_patterns,
)
from src.model.research_scoring import (
    polysyllabic_rhyme_score, internal_rhyme_score, complexity_score,
    temporal_arc_score, introspection_score, vocabulary_novelty_score,
    stress_alignment_score,
)
from src.generation.surprise_engine import surprise_score, diagnose as surprise_diagnose
from src.generation.flow_dna import flow_score, diagnose as flow_diagnose


# =============================================================================
#  PART 1: MODULE OUTPUT - the standardized output of every specialized module
# =============================================================================

@dataclass
class ModuleOutput:
    """
    Standardized output from any specialized module.
    Every module produces this exact structure - this is what flows into
    the Global Workspace for competition and selection.
    """
    module_name: str               # which module produced this
    score: float                   # [0, 1] how well the candidate satisfies this module
    confidence: float              # [0, 1] how confident the module is in its own score
    flags: list[str] = field(default_factory=list)   # specific issues detected
    reasoning: str = ""            # human-readable explanation of WHY this score
    sub_scores: dict = field(default_factory=dict)    # detailed breakdown


# =============================================================================
#  PART 2: THE 7 SPECIALIZED MODULES - parallel unconscious processors
# =============================================================================
# Each mirrors a specific brain region/function from the neuroscience research:
#   Phonology  -> inferior frontal gyrus + supramarginal gyrus (phonological loop)
#   Stress     -> SMA + basal ganglia (motor-rhythm timing)
#   Emotion    -> amygdala + MPFC coupling (emotional generation)
#   Semantic   -> Broca's area BA44/45 (linguistic template-matching)
#   Structure  -> DLPFC working memory (8-bar arc tracking)
#   Texture    -> planum polare BA38 (tertiary auditory/musical grammar)
#   Dopamine   -> nucleus accumbens (reward prediction / frisson)

class PhonologyModule:
    """
    Inferior frontal gyrus + supramarginal gyrus.
    Evaluates rhyme quality: end-rhyme match AND polysyllabic depth.
    Multi-syllabic rhymes are the #1 predictor of lyrical quality (arXiv:2505.00035).
    """
    name = "phonology"

    def evaluate(
        self,
        line: str,
        target_end_phoneme: Optional[str],
        previous_line: Optional[str],
        **kwargs,
    ) -> ModuleOutput:
        ann = annotate_line(line)

        # End-rhyme match
        if target_end_phoneme is None:
            rhyme_match = 1.0
            rhyme_reason = "no rhyme target (free position)"
        elif ann.end_phoneme and rhymes(ann.end_phoneme, target_end_phoneme):
            rhyme_match = 1.0
            rhyme_reason = f"end-rhyme match: {ann.end_phoneme} ~ {target_end_phoneme}"
        else:
            rhyme_match = 0.0
            rhyme_reason = f"rhyme MISS: got {ann.end_phoneme}, need {target_end_phoneme}"

        # Polysyllabic depth
        poly_score = 0.5
        if previous_line:
            poly_score = polysyllabic_rhyme_score(line, previous_line)

        # Internal rhyme density
        internal = internal_rhyme_score(line)

        # Combined: end-rhyme is critical, poly + internal are quality signals
        combined = 0.50 * rhyme_match + 0.30 * poly_score + 0.20 * internal
        confidence = 0.95 if ann.end_phoneme else 0.4  # low confidence if phoneme unknown

        flags = []
        if rhyme_match == 0.0:
            flags.append("RHYME_MISS")
        if poly_score > 0.6:
            flags.append("POLYSYLLABIC_RHYME")
        if internal > 0.5:
            flags.append("INTERNAL_RHYME")

        return ModuleOutput(
            module_name=self.name,
            score=combined,
            confidence=confidence,
            flags=flags,
            reasoning=f"{rhyme_reason}; poly={poly_score:.2f}, internal={internal:.2f}",
            sub_scores={
                "end_rhyme": rhyme_match,
                "polysyllabic": poly_score,
                "internal_rhyme": internal,
            },
        )


class StressModule:
    """
    SMA (supplementary motor area) + basal ganglia + cerebellum.
    Evaluates rhythmic stress alignment - the motor-rhythm coupling that
    makes lyrics feel like they MOVE with the beat.
    The brain plans articulation and content in the same moment (Brown et al. 2004).
    """
    name = "stress"

    def evaluate(self, line: str, section: str = "verse1", **kwargs) -> ModuleOutput:
        score = stress_alignment_score(line, section)
        ann = annotate_line(line)

        # Syllable count check
        target_syl = kwargs.get("target_syllables", 10)
        syl_diff = abs(ann.total_syllables - target_syl)
        syllable_ok = syl_diff <= 3
        syllable_score = max(0.0, 1.0 - syl_diff / 6.0)

        combined = 0.6 * score + 0.4 * syllable_score
        confidence = 0.85

        flags = []
        if not syllable_ok:
            flags.append(f"SYLLABLE_OFF_BY_{syl_diff}")
        if score > 0.7:
            flags.append("STRONG_RHYTHM")

        section_pattern = {
            "verse1": "iambic", "verse2": "iambic",
            "chorus": "trochaic", "bridge": "anapestic",
        }.get(section.lower(), "iambic")

        return ModuleOutput(
            module_name=self.name,
            score=combined,
            confidence=confidence,
            flags=flags,
            reasoning=f"stress={score:.2f} (target: {section_pattern}), "
                      f"syllables={ann.total_syllables} (target: {target_syl})",
            sub_scores={
                "stress_alignment": score,
                "syllable_score": syllable_score,
                "syllable_count": ann.total_syllables,
            },
        )


class EmotionModule:
    """
    Amygdala + MPFC coupling.
    Evaluates emotional content in 8D space against the genre trajectory target.
    The amygdala is part of the SAME network generating words during creative flow
    (Liu et al. 2012) - emotion is not added afterward, it is generated simultaneously.
    """
    name = "emotion"

    def evaluate(
        self,
        line: str,
        genre: str = "hip_hop",
        section: str = "verse1",
        mood: str = "dark",
        **kwargs,
    ) -> ModuleOutput:
        # 8D trajectory fit
        traj_fit = trajectory_fit_score(line, genre, section)

        # Simple valence/arousal fit
        emotion = score_line(line)
        target_point = get_target_point(genre, section)
        val_diff = abs(emotion.valence - target_point.valence)
        aro_diff = abs(emotion.arousal - target_point.arousal)
        simple_fit = 1.0 - (val_diff + aro_diff) / 4.0

        # Introspection bonus (246% surge in modern music - arXiv:2505.00035)
        introspect = introspection_score(line)

        combined = 0.50 * traj_fit + 0.25 * simple_fit + 0.25 * introspect
        confidence = 0.80

        flags = []
        if traj_fit > 0.7:
            flags.append("ON_TRAJECTORY")
        if traj_fit < 0.3:
            flags.append("OFF_TRAJECTORY")
        if introspect > 0.5:
            flags.append("CONFESSIONAL")

        line_ep = line_emotion(line)
        return ModuleOutput(
            module_name=self.name,
            score=combined,
            confidence=confidence,
            flags=flags,
            reasoning=f"8D trajectory fit={traj_fit:.2f}, valence={line_ep.valence:+.2f}, "
                      f"arousal={line_ep.arousal:.2f}, introspection={introspect:.2f}",
            sub_scores={
                "trajectory_fit": traj_fit,
                "simple_fit": simple_fit,
                "introspection": introspect,
                "valence": emotion.valence,
                "arousal": emotion.arousal,
            },
        )


class SemanticModule:
    """
    Broca's area (BA44/45) - automatic template-matching.
    Evaluates novelty, vocabulary diversity, and complexity calibration.
    Experienced songwriters apply this unconsciously as "gut feeling" (McIntyre 2022).
    """
    name = "semantic"

    def evaluate(
        self,
        line: str,
        accepted_lines: list[str],
        **kwargs,
    ) -> ModuleOutput:
        # Novelty vs accepted lines
        def simple_overlap(a: str, b: str) -> float:
            aw = set(a.lower().split())
            bw = set(b.lower().split())
            return len(aw & bw) / len(aw | bw) if (aw and bw) else 0.0

        max_overlap = max(
            (simple_overlap(line, prev) for prev in accepted_lines),
            default=0.0,
        )
        novelty = 1.0 - max_overlap

        # Vocabulary diversity
        vocab_novelty = vocabulary_novelty_score(line, accepted_lines[-8:])

        # Complexity calibration (quadratic: optimal at 65th percentile)
        complexity = complexity_score(line, accepted_lines)

        combined = 0.35 * novelty + 0.35 * vocab_novelty + 0.30 * complexity
        confidence = 0.85

        flags = []
        if novelty < 0.3:
            flags.append("REPETITIVE")
        if complexity < 0.3:
            flags.append("TOO_SIMPLE")
        elif complexity < 0.5:
            flags.append("COMPLEXITY_LOW")
        if vocab_novelty > 0.7:
            flags.append("FRESH_VOCABULARY")

        return ModuleOutput(
            module_name=self.name,
            score=combined,
            confidence=confidence,
            flags=flags,
            reasoning=f"novelty={novelty:.2f}, vocab_diversity={vocab_novelty:.2f}, "
                      f"complexity={complexity:.2f} (target: 65th percentile)",
            sub_scores={
                "novelty": novelty,
                "vocab_novelty": vocab_novelty,
                "complexity": complexity,
            },
        )


class StructureModule:
    """
    DLPFC working memory - tracks position in the 8-bar arc.
    Left hemisphere generates bars 1-6, right hemisphere monitors resolution
    on bars 7-8 (Liu et al. 2012). Bar 7 = peak tension. Bar 8 = must resolve.
    """
    name = "structure"

    def evaluate(
        self,
        line: str,
        line_idx: int,
        rhymes_with_target: bool,
        tension_state: float,
        **kwargs,
    ) -> ModuleOutput:
        score = temporal_arc_score(line, line_idx, rhymes_with_target, tension_state)
        position = line_idx % 8

        # Position-specific reasoning
        if position < 6:
            phase = "generative (bars 1-6)"
            expected = "building tension, high novelty"
        elif position == 6:
            phase = "PEAK (bar 7)"
            expected = "maximum tension, hook DNA should fire"
        else:
            phase = "RESOLUTION (bar 8)"
            expected = "rhyme must land, tension drops"

        confidence = 0.90 if position >= 6 else 0.75  # more certain about critical positions

        flags = []
        if position == 7 and not rhymes_with_target:
            flags.append("RESOLUTION_RHYME_MISS")
        if position == 6 and tension_state < 0.4:
            flags.append("LOW_TENSION_AT_PEAK")
        if position == 7 and tension_state > 0.6:
            flags.append("UNRESOLVED_TENSION")

        return ModuleOutput(
            module_name=self.name,
            score=score,
            confidence=confidence,
            flags=flags,
            reasoning=f"bar {position+1}/8 ({phase}): expected={expected}, "
                      f"tension={tension_state:.2f}, rhyme_ok={rhymes_with_target}",
            sub_scores={
                "temporal_arc": score,
                "bar_position": position,
                "tension_state": tension_state,
            },
        )


class TextureModule:
    """
    Planum polare (BA38) - tertiary auditory cortex / musical grammar processor.
    Brown et al. found this activates ONLY for structured musical material,
    not simple monotone. It parses musical syntax complexity.
    Here: evaluates phonosemantic texture alignment (sound-meaning fit).
    """
    name = "texture"

    def evaluate(
        self,
        line: str,
        mood: str = "dark",
        section: str = "verse1",
        **kwargs,
    ) -> ModuleOutput:
        score = texture_alignment_score(line, mood, section)
        texture = analyze_line_texture(line)
        description = texture.describe()

        confidence = 0.70  # phonosemantic is probabilistic

        flags = []
        if score > 0.7:
            flags.append("TEXTURE_ALIGNED")
        if score < 0.3:
            flags.append("TEXTURE_MISMATCH")
        if abs(texture.brightness) > 0.5:
            flags.append("BRIGHT" if texture.brightness > 0 else "DARK")

        return ModuleOutput(
            module_name=self.name,
            score=score,
            confidence=confidence,
            flags=flags,
            reasoning=f"texture: {description}; alignment to {mood}/{section}={score:.2f}",
            sub_scores={
                "alignment": score,
                "brightness": texture.brightness,
                "warmth": texture.warmth,
                "weight": texture.weight,
                "sharpness": texture.sharpness,
            },
        )


class DopamineModule:
    """
    Nucleus accumbens - reward prediction / frisson circuit.
    Dopamine releases BEFORE the resolution, in ANTICIPATION.
    This module predicts the goosebump probability and detects hook DNA patterns.
    """
    name = "dopamine"

    def evaluate(
        self,
        line: str,
        tension_state: float,
        previous_line: Optional[str] = None,
        mood: str = "dark",
        **kwargs,
    ) -> ModuleOutput:
        gbump = goosebump_potential(line, tension_state, previous_line, mood)
        hook = hook_dna_score(line, previous_line)
        patterns = detect_hook_patterns(line, previous_line)

        combined = 0.60 * gbump + 0.40 * hook
        confidence = 0.75

        flags = []
        if gbump > 0.6:
            flags.append("GOOSEBUMP_MOMENT")
        if hook > 0.5:
            flags.append("HOOK_DNA_DETECTED")
        for p in patterns:
            flags.append(f"HOOK:{p.upper()}")

        patterns_str = ", ".join(patterns) if patterns else "none"
        return ModuleOutput(
            module_name=self.name,
            score=combined,
            confidence=confidence,
            flags=flags,
            reasoning=f"goosebump={gbump:.2f}, hook_dna={hook:.2f}, "
                      f"patterns=[{patterns_str}], tension={tension_state:.2f}",
            sub_scores={
                "goosebump": gbump,
                "hook_dna": hook,
                "patterns": patterns,
            },
        )


# =============================================================================
#  PART 3: METACOGNITIVE STATE VECTOR (MSV) - System 1/2 switching
# =============================================================================

@dataclass
class MetacognitiveStateVector:
    """
    5 quantified signals that control fast/slow (System 1/2) switching.
    When signals are in normal range -> System 1 (MPFC, creative flow).
    When signals indicate conflict -> System 2 (DLPFC, deliberative).

    Source: Sethi et al. 2025 (Metacognitive State Vector for Generative AI)
    """
    emotional_salience: float = 0.5    # how emotionally charged is the context
    output_confidence: float = 0.7     # average module confidence
    experience_match: float = 0.6      # how similar to training distribution
    conflict_level: float = 0.2        # inter-module disagreement
    task_importance: float = 0.5       # positional urgency (bars 7-8 = max)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.emotional_salience, self.output_confidence,
            self.experience_match, self.conflict_level,
            self.task_importance,
        ], dtype=np.float32)

    def system_mode(self) -> str:
        """
        Determine System 1 or System 2 based on MSV signals.

        System 1 (MPFC mode): fast, creative, high temperature
          - Normal confidence, low conflict, moderate importance
          - This is the brain's default during creative flow

        System 2 (DLPFC mode): slow, deliberative, more candidates
          - Low confidence OR high conflict OR high importance
          - The "critic" re-engages to apply tighter constraints
        """
        # Switch to System 2 if ANY of these conditions:
        if self.output_confidence < 0.45:
            return "system2"
        if self.conflict_level > 0.55:
            return "system2"
        if self.task_importance > 0.75:
            return "system2"
        if self.experience_match < 0.30:
            return "system2"
        return "system1"

    def describe(self) -> str:
        mode = self.system_mode()
        return (
            f"[{mode.upper()}] salience={self.emotional_salience:.2f} "
            f"confidence={self.output_confidence:.2f} "
            f"experience={self.experience_match:.2f} "
            f"conflict={self.conflict_level:.2f} "
            f"importance={self.task_importance:.2f}"
        )


# =============================================================================
#  PART 4: GENERATION TRACE (HOT) - Higher Order Thought
# =============================================================================

@dataclass
class GenerationTrace:
    """
    Higher Order Thought: a representation OF the generation event itself.
    This is what makes the system conscious of its own output in the functional
    HOT sense - it tracks what was generated, why, and what to do next.

    Without this, the system is a sleepwalker - producing outputs without
    awareness of producing them. WITH this, it is a functional creative agent.
    """
    line: str                                # what was generated
    module_scores: dict[str, float] = field(default_factory=dict)  # per-module scores
    module_reasoning: dict[str, str] = field(default_factory=dict) # per-module explanations
    winning_modules: list[str] = field(default_factory=list)       # which modules drove selection
    losing_modules: list[str] = field(default_factory=list)        # which modules voted against
    all_flags: list[str] = field(default_factory=list)             # all flags from all modules
    active_constraints: list[str] = field(default_factory=list)    # which constraints were active
    total_score: float = 0.0                 # workspace-weighted final score
    workspace_confidence: float = 0.0        # workspace's confidence in this selection
    system_mode: str = "system1"             # which cognitive mode produced this
    metacognitive_state: Optional[MetacognitiveStateVector] = None
    decision: str = "ACCEPT"                 # ACCEPT / REVISE / REGENERATE
    decision_reason: str = ""                # why this decision was made
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        """One-line human-readable summary of this generation event."""
        top = ", ".join(self.winning_modules[:3])
        flags = ", ".join(self.all_flags[:5])
        return (
            f"[{self.system_mode}|{self.decision}] "
            f"score={self.total_score:.3f} conf={self.workspace_confidence:.2f} "
            f"won_by=[{top}] flags=[{flags}]"
        )


# =============================================================================
#  PART 5: SELF-MODEL (TRAP Adaptation) - learns its own strengths/weaknesses
# =============================================================================

@dataclass
class ModuleCalibration:
    """
    Per-module accuracy tracker for self-model adaptation.
    If a module is consistently overruled -> reduce its weight.
    If a module is consistently the deciding factor -> boost it.
    """
    times_won: int = 0         # times this module was the top contributor
    times_overruled: int = 0   # times this module flagged an issue but was overruled
    total_evaluations: int = 0
    cumulative_score: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.times_won / max(self.total_evaluations, 1)

    @property
    def avg_score(self) -> float:
        return self.cumulative_score / max(self.total_evaluations, 1)

    @property
    def reliability(self) -> float:
        """
        How reliable is this module? High win rate + high avg score = reliable.
        Used to adjust module weights during the session.
        """
        if self.total_evaluations < 3:
            return 1.0  # not enough data, trust default
        return float(np.clip(
            0.5 * self.win_rate + 0.5 * self.avg_score,
            0.3, 1.5,  # clamp to avoid zero-ing out or over-boosting
        ))


class SelfModel:
    """
    TRAP Adaptation: the system's model of its own strengths and weaknesses.
    Tracks which modules are performing well and adjusts weights accordingly.
    """

    def __init__(self):
        self.calibrations: dict[str, ModuleCalibration] = {}
        self.session_history: list[GenerationTrace] = []
        self.system1_count: int = 0
        self.system2_count: int = 0

    def get_calibration(self, module_name: str) -> ModuleCalibration:
        if module_name not in self.calibrations:
            self.calibrations[module_name] = ModuleCalibration()
        return self.calibrations[module_name]

    def record_evaluation(self, trace: GenerationTrace):
        """Record a generation event to update the self-model."""
        self.session_history.append(trace)

        if trace.system_mode == "system1":
            self.system1_count += 1
        else:
            self.system2_count += 1

        for mod_name, score in trace.module_scores.items():
            cal = self.get_calibration(mod_name)
            cal.total_evaluations += 1
            cal.cumulative_score += score

            if mod_name in trace.winning_modules:
                cal.times_won += 1
            if mod_name in trace.losing_modules:
                cal.times_overruled += 1

    def get_weight_adjustment(self, module_name: str) -> float:
        """
        Get the adaptive weight multiplier for a module.
        Starts at 1.0, adjusts based on session performance.
        """
        cal = self.get_calibration(module_name)
        return cal.reliability

    def session_report(self) -> dict:
        """Full session self-awareness report."""
        return {
            "total_generations": len(self.session_history),
            "system1_ratio": self.system1_count / max(len(self.session_history), 1),
            "system2_ratio": self.system2_count / max(len(self.session_history), 1),
            "module_reliability": {
                name: {
                    "reliability": cal.reliability,
                    "win_rate": round(cal.win_rate, 3),
                    "avg_score": round(cal.avg_score, 3),
                    "evaluations": cal.total_evaluations,
                }
                for name, cal in self.calibrations.items()
            },
            "strongest_module": max(
                self.calibrations.items(),
                key=lambda x: x[1].reliability,
                default=("none", ModuleCalibration()),
            )[0],
            "weakest_module": min(
                self.calibrations.items(),
                key=lambda x: x[1].reliability,
                default=("none", ModuleCalibration()),
            )[0],
        }


# =============================================================================
#  PART 5b: SURPRISE MODULE — Anterior Cingulate Cortex
# =============================================================================

class SurpriseModule:
    """
    Anterior Cingulate Cortex (ACC) — conflict detection + novelty drive.

    The ACC fires when prediction errors occur: when the brain expected X
    but got Y. This is the neural basis of "surprise" in music.

    Great trap lines live in the VIRAL SWEET SPOT: familiar enough to be
    understood, surprising enough to be remembered. This module scores for
    that balance using the Predictive Surprise Engine (Huron 2006).

    Score zones:
      < 0.35 = too predictable (forgettable)
      0.35–0.70 = viral sweet spot ✓
      > 0.70 = too random (incoherent)
    """
    name = "surprise"

    def evaluate(
        self,
        line: str,
        genre: str,
        accepted_lines: list[str],
        **kwargs,
    ) -> ModuleOutput:
        score = surprise_score(line, genre, accepted_lines)
        diag = surprise_diagnose(line, genre, accepted_lines)

        flags: list[str] = []
        if score < 0.35:
            flags.append("TOO_PREDICTABLE")
        elif score > 0.70:
            flags.append("TOO_RANDOM")
        else:
            flags.append("SURPRISE_SWEET_SPOT")

        if diag["cliche_penalty"] > 0.4:
            flags.append("CLICHE_DETECTED")
        if diag["structural"] > 0.5:
            flags.append("STRUCTURAL_INVERSION")

        # Remap score so sweet-spot (0.35-0.70) scores highest
        # Peak reward at 0.52 (centre of sweet spot)
        deviation = abs(score - 0.52)
        normalised = max(0.0, 1.0 - deviation / 0.52)

        return ModuleOutput(
            module_name=self.name,
            score=normalised,
            confidence=0.80,
            flags=flags,
            reasoning=(
                f"surprise={score:.2f} ({diag['verdict']}); "
                f"vocab={diag['vocabulary']:.2f}, "
                f"leap={diag['semantic_leap']:.2f}, "
                f"structural={diag['structural']:.2f}"
            ),
            sub_scores={
                "raw_surprise":   score,
                "vocabulary":     diag["vocabulary"],
                "semantic_leap":  diag["semantic_leap"],
                "structural":     diag["structural"],
                "cliche_penalty": diag["cliche_penalty"],
            },
        )


# =============================================================================
#  PART 5c: FLOW MODULE — Supplementary Motor Area (SMA)
# =============================================================================

class FlowModule:
    """
    Supplementary Motor Area (SMA) — rhythm production and motor-beat coupling.

    The SMA is responsible for timing in music and speech. In freestyle rap,
    it fires when the rapper "feels" the beat before they articulate.

    This module scores how well the line's stress pattern matches the target
    flow fingerprint for the current section/arc combination (e.g. triplet
    for [BUILD]/VERSE, melodic for CHORUS/[PEAK]).

    Uses the Flow DNA library of viral trap flows encoded as stress templates.
    """
    name = "flow"

    def evaluate(
        self,
        line: str,
        section: str,
        arc_token: str,
        **kwargs,
    ) -> ModuleOutput:
        diag = flow_diagnose(line, section, arc_token)
        score = diag["score"]

        flags: list[str] = []
        if score > 0.65:
            flags.append(f"FLOW_MATCH:{diag['target_flow']}")
        elif score < 0.35:
            flags.append("FLOW_MISMATCH")

        best_alt = diag["ranking"][0][0] if diag["ranking"] else "unknown"
        if best_alt != diag["target_flow"] and score < 0.50:
            flags.append(f"BETTER_FLOW:{best_alt}")

        return ModuleOutput(
            module_name=self.name,
            score=score,
            confidence=0.78,
            flags=flags,
            reasoning=(
                f"target={diag['target_flow']} ({diag['target_description']}); "
                f"syllables={diag['actual_syllables']}/{diag['target_syllables']}; "
                f"stress_fit={diag['stress_fit']:.2f}; "
                f"key: {diag['key_feature']}"
            ),
            sub_scores={
                "syllable_fit": diag["syllable_fit"],
                "stress_fit":   diag["stress_fit"],
                "density_fit":  diag["density_fit"],
            },
        )


# =============================================================================
#  PART 6: GLOBAL WORKSPACE - the consciousness layer
# =============================================================================

# Default module weights (before self-model adaptation).
# Rebalanced to include surprise (ACC) and flow (SMA) — sum = 1.0
DEFAULT_MODULE_WEIGHTS: dict[str, float] = {
    "phonology": 0.20,    # rhyme is foundational (biggest quality predictor)
    "stress":    0.08,    # rhythm matters but is more forgiving
    "emotion":   0.15,    # emotional fit is what makes songs resonate
    "semantic":  0.10,    # novelty prevents repetition
    "structure": 0.12,    # temporal arc creates coherent songs
    "texture":   0.08,    # phonosemantic is subtle but powerful
    "dopamine":  0.14,    # goosebump moments are what make songs legendary
    "surprise":  0.08,    # ACC: predictive surprise sweet spot
    "flow":      0.05,    # SMA: rhythmic fingerprint match
}


class MetacognitiveWorkspace:
    """
    The Global Workspace - where all module outputs compete for selection
    and the winning candidate is broadcast with full justification.

    This is the GWT consciousness implementation: parallel specialized
    processors -> competition -> broadcast -> all modules update.

    Usage:
        workspace = MetacognitiveWorkspace()
        ranked = workspace.evaluate_candidates(
            candidates=["line 1", "line 2", "line 3"],
            genre="trap", section="verse1", mood="dark",
            target_end_phoneme="EY1", previous_line="last accepted line",
            accepted_lines=["all", "previous", "lines"],
            line_idx=5, tension_state=0.4, target_syllables=10,
        )
        # ranked[0] is the best candidate with its GenerationTrace
    """

    def __init__(self, weights: Optional[dict[str, float]] = None):
        # Initialize the 9 specialized modules
        self.modules = {
            "phonology": PhonologyModule(),
            "stress":    StressModule(),
            "emotion":   EmotionModule(),
            "semantic":  SemanticModule(),
            "structure": StructureModule(),
            "texture":   TextureModule(),
            "dopamine":  DopamineModule(),
            "surprise":  SurpriseModule(),
            "flow":      FlowModule(),
        }
        self.base_weights = weights or DEFAULT_MODULE_WEIGHTS.copy()
        self.self_model = SelfModel()

    def _compute_msv(
        self,
        all_outputs: list[dict[str, ModuleOutput]],
        line_idx: int,
        accepted_lines: list[str],
    ) -> MetacognitiveStateVector:
        """
        Compute the Metacognitive State Vector from current context.
        This determines System 1 vs System 2 mode.
        """
        if not all_outputs:
            return MetacognitiveStateVector()

        # Emotional salience: how emotionally charged is the current context
        # More accepted lines with strong emotion = higher salience
        salience = 0.5
        if accepted_lines:
            recent_emotions = [score_line(l) for l in accepted_lines[-4:]]
            avg_intensity = np.mean([
                abs(e.valence) + e.arousal for e in recent_emotions
            ])
            salience = float(np.clip(avg_intensity / 2.0, 0.0, 1.0))

        # Output confidence: average of all module confidences across all candidates
        all_confidences = []
        for candidate_outputs in all_outputs:
            for mod_out in candidate_outputs.values():
                all_confidences.append(mod_out.confidence)
        confidence = float(np.mean(all_confidences)) if all_confidences else 0.5

        # Experience match: are we in familiar territory?
        # Short lines or very novel vocabulary = less familiar
        experience = 0.6  # default
        if len(accepted_lines) > 8:
            experience = 0.8  # more context = more confidence
        if len(accepted_lines) < 2:
            experience = 0.3  # cold start

        # Conflict level: how much do modules disagree?
        conflicts = []
        for candidate_outputs in all_outputs:
            scores = [m.score for m in candidate_outputs.values()]
            if len(scores) >= 2:
                conflicts.append(float(np.std(scores)))
        conflict = float(np.mean(conflicts)) if conflicts else 0.2

        # Task importance: based on bar position (7-8 are critical)
        position = line_idx % 8
        if position >= 6:
            importance = 0.9  # bars 7-8 are critical resolution points
        elif position >= 4:
            importance = 0.6  # approaching peak
        else:
            importance = 0.4  # generative phase, lower stakes

        return MetacognitiveStateVector(
            emotional_salience=salience,
            output_confidence=confidence,
            experience_match=experience,
            conflict_level=conflict,
            task_importance=importance,
        )

    def _get_effective_weights(self) -> dict[str, float]:
        """
        Get module weights adjusted by the self-model's learning.
        This is TRAP Adaptation: the system adjusts its own strategy.
        """
        adjusted: dict[str, float] = {}
        for mod_name, base_w in self.base_weights.items():
            adaptation = self.self_model.get_weight_adjustment(mod_name)
            adjusted[mod_name] = base_w * adaptation
        # Re-normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted

    def _run_modules(
        self,
        line: str,
        genre: str,
        section: str,
        mood: str,
        target_end_phoneme: Optional[str],
        previous_line: Optional[str],
        accepted_lines: list[str],
        line_idx: int,
        tension_state: float,
        target_syllables: int,
        arc_token: str = "[BUILD]",
    ) -> dict[str, ModuleOutput]:
        """
        Run all 9 specialized modules in parallel on a single candidate line.
        Each module produces a standardized ModuleOutput.

        Brain regions: inferior frontal gyrus, SMA, amygdala, Broca's area,
        DLPFC, planum polare, nucleus accumbens, ACC (surprise), SMA (flow).
        """
        ann = annotate_line(line)
        rhymes_with = False
        if target_end_phoneme and ann.end_phoneme:
            rhymes_with = rhymes(ann.end_phoneme, target_end_phoneme)

        outputs: dict[str, ModuleOutput] = {}
        outputs["phonology"] = self.modules["phonology"].evaluate(
            line, target_end_phoneme, previous_line,
        )
        outputs["stress"] = self.modules["stress"].evaluate(
            line, section, target_syllables=target_syllables,
        )
        outputs["emotion"] = self.modules["emotion"].evaluate(
            line, genre, section, mood,
        )
        outputs["semantic"] = self.modules["semantic"].evaluate(
            line, accepted_lines,
        )
        outputs["structure"] = self.modules["structure"].evaluate(
            line, line_idx, rhymes_with, tension_state,
        )
        outputs["texture"] = self.modules["texture"].evaluate(
            line, mood, section,
        )
        outputs["dopamine"] = self.modules["dopamine"].evaluate(
            line, tension_state, previous_line, mood,
        )
        outputs["surprise"] = self.modules["surprise"].evaluate(
            line, genre=genre, accepted_lines=accepted_lines,
        )
        outputs["flow"] = self.modules["flow"].evaluate(
            line, section=section, arc_token=arc_token,
        )
        return outputs

    def _workspace_select(
        self,
        line: str,
        module_outputs: dict[str, ModuleOutput],
        weights: dict[str, float],
        msv: MetacognitiveStateVector,
    ) -> GenerationTrace:
        """
        The GLOBAL WORKSPACE SELECTION event.

        This is GWT consciousness: all modules have produced their outputs.
        Now ONE integrated representation wins the competition and is
        broadcast to all other modules.

        The selection process:
        1. Compute weighted score across all modules
        2. Identify which modules drove the selection (top contributors)
        3. Identify which modules voted against (flagged issues)
        4. Compute workspace confidence
        5. Produce the generation trace (HOT: thought about the thought)
        6. Make a metacognitive decision: ACCEPT, REVISE, or REGENERATE
        """
        # 1. Weighted score
        total_score = 0.0
        for mod_name, mod_out in module_outputs.items():
            w = weights.get(mod_name, 0.1)
            total_score += w * mod_out.score

        # 2. Identify winning modules (score > 0.6) and losing modules (score < 0.3)
        winning = [name for name, out in module_outputs.items() if out.score > 0.6]
        losing = [name for name, out in module_outputs.items() if out.score < 0.3]

        # 3. Collect all flags
        all_flags = []
        for out in module_outputs.values():
            all_flags.extend(out.flags)

        # 4. Workspace confidence: based on module agreement and MSV
        score_values = [out.score for out in module_outputs.values()]
        agreement = 1.0 - float(np.std(score_values))
        workspace_conf = 0.5 * agreement + 0.3 * msv.output_confidence + 0.2 * (1.0 - msv.conflict_level)

        # 5. Active constraints (which modules are critical at this position)
        active_constraints = []
        position = module_outputs.get("structure", ModuleOutput("structure", 0.0, 0.0)).sub_scores.get("bar_position", 0)
        if position >= 6:
            active_constraints.append("RESOLUTION_REQUIRED")
        if "RHYME_MISS" in all_flags:
            active_constraints.append("RHYME_ENFORCEMENT")
        if any("SYLLABLE_OFF" in f for f in all_flags):
            active_constraints.append("SYLLABLE_ENFORCEMENT")

        # 6. Metacognitive decision
        system_mode = msv.system_mode()
        if total_score > 0.5 and "RHYME_MISS" not in all_flags:
            decision = "ACCEPT"
            decision_reason = f"score {total_score:.3f} > 0.5 threshold, no critical failures"
        elif "RHYME_MISS" in all_flags and position == 7:
            decision = "REGENERATE"
            decision_reason = "bar 8 must resolve rhyme but end-rhyme missed"
        elif total_score < 0.25:
            decision = "REGENERATE"
            decision_reason = f"score {total_score:.3f} below minimum quality threshold"
        elif len(losing) >= 3:
            decision = "REVISE"
            decision_reason = f"{len(losing)} modules scored below 0.3: {losing}"
        else:
            decision = "ACCEPT"
            decision_reason = f"acceptable quality: score {total_score:.3f}"

        return GenerationTrace(
            line=line,
            module_scores={name: out.score for name, out in module_outputs.items()},
            module_reasoning={name: out.reasoning for name, out in module_outputs.items()},
            winning_modules=winning,
            losing_modules=losing,
            all_flags=all_flags,
            active_constraints=active_constraints,
            total_score=total_score,
            workspace_confidence=workspace_conf,
            system_mode=system_mode,
            metacognitive_state=msv,
            decision=decision,
            decision_reason=decision_reason,
        )

    def evaluate_candidates(
        self,
        candidates: list[str],
        genre: str = "hip_hop",
        section: str = "verse1",
        mood: str = "dark",
        target_end_phoneme: Optional[str] = None,
        previous_line: Optional[str] = None,
        accepted_lines: Optional[list[str]] = None,
        line_idx: int = 0,
        tension_state: float = 0.3,
        target_syllables: int = 10,
        arc_token: str = "[BUILD]",
    ) -> list[GenerationTrace]:
        """
        THE MAIN ENTRY POINT - evaluate all candidate lines through the
        full cognitive architecture.

        This is the complete brain process in one call:
        1. Run all 7 modules in parallel on every candidate
        2. Compute the Metacognitive State Vector (System 1/2 decision)
        3. Apply self-model weight adjustments (TRAP Adaptation)
        4. Global workspace selection: rank candidates with justification traces
        5. Return ranked list of GenerationTrace objects

        Returns: list of GenerationTrace, sorted by total_score descending.
                 Each trace contains full per-module reasoning, confidence,
                 which modules drove the selection, and a metacognitive decision.
        """
        if accepted_lines is None:
            accepted_lines = []

        # Step 1: Run all 9 modules on every candidate
        all_outputs: list[dict[str, ModuleOutput]] = []
        for line in candidates:
            outputs = self._run_modules(
                line, genre, section, mood,
                target_end_phoneme, previous_line,
                accepted_lines, line_idx, tension_state, target_syllables,
                arc_token=arc_token,
            )
            all_outputs.append(outputs)

        # Step 2: Compute MSV
        msv = self._compute_msv(all_outputs, line_idx, accepted_lines)

        # Step 3: Get effective weights (with self-model adaptation)
        weights = self._get_effective_weights()

        # Step 4: Workspace selection for each candidate
        traces: list[GenerationTrace] = []
        for line, outputs in zip(candidates, all_outputs):
            trace = self._workspace_select(line, outputs, weights, msv)
            traces.append(trace)

        # Step 5: Sort by total score, best first
        traces.sort(key=lambda t: t.total_score, reverse=True)
        return traces

    def accept_line(self, trace: GenerationTrace):
        """
        After a line is accepted: broadcast the result to the self-model.
        This is the GWT broadcast event - all modules learn from this selection.
        """
        self.self_model.record_evaluation(trace)

    def get_session_report(self) -> dict:
        """
        Full metacognitive session report.
        Shows: per-module reliability, System 1/2 ratio, generation history.
        This is what you show the artist: "Here's how the AI's brain worked."
        """
        report = self.self_model.session_report()

        # Add generation trace summaries
        report["trace_summaries"] = [
            t.summary() for t in self.self_model.session_history[-20:]
        ]

        # Add MSV history
        report["latest_msv"] = None
        if self.self_model.session_history:
            latest = self.self_model.session_history[-1]
            if latest.metacognitive_state:
                report["latest_msv"] = latest.metacognitive_state.describe()

        return report
