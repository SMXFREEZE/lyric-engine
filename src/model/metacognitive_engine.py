"""
Metacognitive workspace controller for lyric generation.

This module layers an explicit "thinking about the thinking" system on top of
the existing scoring stack. The goal is not phenomenal consciousness; it is a
functional creative agent architecture with:

- specialist modules that score different creative constraints in parallel
- a global workspace that integrates those signals into a single broadcast
- working memory that tracks the current songwriting state
- System 1 / System 2 mode switching based on confidence and conflict
- a self-model that adapts module weighting across a session
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Optional

import numpy as np

from src.model.composer_cortex import ComposerCortex, DomainMemory


PLACEHOLDER_MARKERS = {"[no candidate generated]"}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "he", "her", "his", "i", "if", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "they", "this", "to", "us", "was", "we", "with", "you", "your",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return 0.0
    return float(sum(score * weight for score, weight in pairs) / total_weight)


@dataclass
class CandidateEvidence:
    text: str
    base_score: float
    phonetic_score: float
    syllable_ok: bool
    novelty_score: float
    valence_fit: float
    trajectory_fit: float
    texture_alignment: float
    goosebump: float
    hook_dna: float
    polysyllabic_rhyme: float
    internal_rhyme: float
    complexity: float
    temporal_arc: float
    introspection: float
    stress_alignment: float


@dataclass
class WorkspaceContext:
    genre: str
    mood: str
    section: str
    bar_index: int
    rhyme_scheme: str
    target_end_phoneme: Optional[str]
    target_syllables: int
    target_arc_valence: float
    target_arc_arousal: float
    tension_state: float
    accepted_lines: list[str] = field(default_factory=list)


@dataclass
class ModuleObservation:
    module: str
    score: float
    confidence: float
    weight: float
    note: str = ""
    hard_veto: bool = False

    def activation(self) -> float:
        return self.score * self.confidence * self.weight

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "weight": round(self.weight, 4),
            "note": self.note,
            "hard_veto": self.hard_veto,
        }


@dataclass
class WorkingMemoryState:
    section: str
    bar_index: int
    recent_lines: list[str]
    active_constraints: list[str]
    unresolved_tension: float
    task_pressure: float
    remaining_resolution: bool
    domain_memory: DomainMemory

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "bar_index": self.bar_index,
            "recent_lines": self.recent_lines,
            "active_constraints": self.active_constraints,
            "unresolved_tension": round(self.unresolved_tension, 4),
            "task_pressure": round(self.task_pressure, 4),
            "remaining_resolution": self.remaining_resolution,
            "domain_memory": self.domain_memory.to_dict(),
        }


@dataclass
class MetacognitiveState:
    mode: str
    confidence: float
    conflict: float
    novelty_pressure: float
    emotional_salience: float
    task_importance: float
    reasons: list[str] = field(default_factory=list)
    revision_focus: list[str] = field(default_factory=list)
    needs_regeneration: bool = False
    suggested_temperature: float = 0.85

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "confidence": round(self.confidence, 4),
            "conflict": round(self.conflict, 4),
            "novelty_pressure": round(self.novelty_pressure, 4),
            "emotional_salience": round(self.emotional_salience, 4),
            "task_importance": round(self.task_importance, 4),
            "reasons": self.reasons,
            "revision_focus": self.revision_focus,
            "needs_regeneration": self.needs_regeneration,
            "suggested_temperature": round(self.suggested_temperature, 4),
        }


@dataclass
class WorkspaceCandidate:
    evidence: CandidateEvidence
    modules: dict[str, ModuleObservation]
    workspace_score: float
    confidence: float
    conflict: float
    salience: float
    vetoed: bool
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.evidence.text,
            "base_score": round(self.evidence.base_score, 4),
            "workspace_score": round(self.workspace_score, 4),
            "confidence": round(self.confidence, 4),
            "conflict": round(self.conflict, 4),
            "salience": round(self.salience, 4),
            "vetoed": self.vetoed,
            "rationale": self.rationale,
            "modules": {name: obs.to_dict() for name, obs in self.modules.items()},
        }


@dataclass
class SelfModel:
    module_strengths: dict[str, float] = field(default_factory=lambda: {
        "phonology": 1.05,
        "rhythm": 1.00,
        "semantics": 0.95,
        "emotion": 1.00,
        "structure": 0.98,
        "novelty": 0.95,
        "auditory": 1.02,
        "template": 1.01,
    })
    recent_confidence: float = 0.60
    recent_conflict: float = 0.20
    last_mode: str = "system1"

    def weight_for(self, module: str) -> float:
        return self.module_strengths.get(module, 1.0)

    def adapt(self, chosen: WorkspaceCandidate, metacognition: MetacognitiveState):
        for name, obs in chosen.modules.items():
            old = self.module_strengths.get(name, 1.0)
            target = 0.70 + obs.score * 0.60
            if obs.hard_veto:
                target *= 0.70
            blended = old * 0.85 + target * 0.15
            self.module_strengths[name] = _clamp(blended, 0.60, 1.40)
        self.recent_confidence = _clamp(self.recent_confidence * 0.8 + metacognition.confidence * 0.2)
        self.recent_conflict = _clamp(self.recent_conflict * 0.8 + metacognition.conflict * 0.2)
        self.last_mode = metacognition.mode

    def snapshot(self) -> dict:
        return {
            "module_strengths": {
                name: round(value, 4) for name, value in self.module_strengths.items()
            },
            "recent_confidence": round(self.recent_confidence, 4),
            "recent_conflict": round(self.recent_conflict, 4),
            "last_mode": self.last_mode,
        }


@dataclass
class WorkspaceDecision:
    chosen_text: str
    broadcast: str
    metacognition: MetacognitiveState
    working_memory: WorkingMemoryState
    ranked_candidates: list[WorkspaceCandidate]
    self_model_snapshot: dict
    hot_trace: dict

    def to_dict(self) -> dict:
        return {
            "chosen_text": self.chosen_text,
            "broadcast": self.broadcast,
            "metacognition": self.metacognition.to_dict(),
            "working_memory": self.working_memory.to_dict(),
            "ranked_candidates": [candidate.to_dict() for candidate in self.ranked_candidates],
            "self_model_snapshot": self.self_model_snapshot,
            "hot_trace": self.hot_trace,
        }


class MetacognitiveEngine:
    """
    Global workspace controller that turns independent scorer outputs into an
    explicit selection, confidence estimate, and revision strategy.
    """

    def __init__(self):
        self.base_weights = {
            "phonology": 1.20,
            "rhythm": 1.10,
            "semantics": 1.00,
            "emotion": 1.10,
            "structure": 1.05,
            "novelty": 0.95,
            "auditory": 1.08,
            "template": 1.06,
        }
        self.cortex = ComposerCortex()

    def evaluate_candidates(
        self,
        candidates: list[CandidateEvidence],
        context: WorkspaceContext,
        self_model: Optional[SelfModel] = None,
    ) -> WorkspaceDecision:
        self_model = self_model or SelfModel()
        working_memory = self._build_working_memory(context)

        inspected = [
            self._inspect_candidate(candidate, context, working_memory, self_model)
            for candidate in candidates
        ]
        inspected.sort(key=lambda item: item.workspace_score, reverse=True)

        metacognition = self._select_mode(inspected, context, working_memory)
        if metacognition.mode == "system2":
            inspected = self._rerank_system2(inspected, context, metacognition)

        chosen = self._choose_candidate(inspected)
        hot_trace = self.cortex.build_hot_trace(chosen, context, working_memory, metacognition)
        self_model.adapt(chosen, metacognition)

        broadcast = self._build_broadcast(chosen, metacognition, context, hot_trace)
        return WorkspaceDecision(
            chosen_text=chosen.evidence.text,
            broadcast=broadcast,
            metacognition=metacognition,
            working_memory=working_memory,
            ranked_candidates=inspected,
            self_model_snapshot=self_model.snapshot(),
            hot_trace=hot_trace,
        )

    def _build_working_memory(self, context: WorkspaceContext) -> WorkingMemoryState:
        bar_position = context.bar_index % 8
        domain_memory = self.cortex.build_domain_memory(context)
        active_constraints = [
            f"genre={context.genre}",
            f"section={context.section}",
            f"rhyme={context.rhyme_scheme}",
            f"target_syllables={context.target_syllables}",
            f"motifs={','.join(domain_memory.motif_tokens[:3])}" if domain_memory.motif_tokens else "motifs=establish",
        ]
        if context.target_end_phoneme:
            active_constraints.append("rhyme_resolution_due")
        if domain_memory.hook_bias >= 0.75:
            active_constraints.append("hook_memorability")
        if domain_memory.confessional_bias >= 0.55:
            active_constraints.append("confessional_pressure")

        task_pressure = 0.50
        if context.target_end_phoneme:
            task_pressure += 0.10
        if bar_position in {6, 7}:
            task_pressure += 0.15
        if context.section.lower() in {"chorus", "hook", "bridge"}:
            task_pressure += 0.10
        task_pressure += min(context.tension_state * 0.15, 0.15)
        task_pressure += min(domain_memory.hook_bias * 0.08, 0.08)
        task_pressure += min(domain_memory.domain_coherence * 0.05, 0.05)

        remaining_resolution = bool(context.target_end_phoneme or bar_position in {6, 7})
        return WorkingMemoryState(
            section=context.section,
            bar_index=context.bar_index,
            recent_lines=context.accepted_lines[-4:],
            active_constraints=active_constraints,
            unresolved_tension=_clamp(context.tension_state),
            task_pressure=_clamp(task_pressure),
            remaining_resolution=remaining_resolution,
            domain_memory=domain_memory,
        )

    def _inspect_candidate(
        self,
        candidate: CandidateEvidence,
        context: WorkspaceContext,
        working_memory: WorkingMemoryState,
        self_model: SelfModel,
    ) -> WorkspaceCandidate:
        modules = self._build_module_observations(candidate, context, working_memory, self_model)
        module_scores = [obs.score for obs in modules.values()]
        vetoed = any(obs.hard_veto for obs in modules.values())

        spread = float(np.std(module_scores)) if module_scores else 0.0
        coupling_mismatch = (
            abs(modules["emotion"].score - modules["structure"].score) * 0.20
            + abs(modules["phonology"].score - modules["rhythm"].score) * 0.20
            + abs(modules["auditory"].score - modules["template"].score) * 0.16
            + abs(modules["template"].score - modules["semantics"].score) * 0.12
        )
        veto_pressure = 0.18 * sum(1 for obs in modules.values() if obs.hard_veto)
        conflict = _clamp(spread * 0.85 + coupling_mismatch + veto_pressure)

        quality = _weighted_mean([
            (obs.score, obs.weight * obs.confidence) for obs in modules.values()
        ])
        salience = _clamp(_weighted_mean([
            (candidate.goosebump, 1.2),
            (candidate.hook_dna, 1.0),
            (candidate.trajectory_fit, 1.0),
            (candidate.introspection, 0.8),
        ]))
        confidence = _clamp(quality * (1.0 - conflict * 0.55) + candidate.base_score * 0.15)
        workspace_score = _clamp(
            quality * 0.60
            + salience * 0.18
            + candidate.base_score * 0.22
            - conflict * 0.18
            - (0.12 if vetoed else 0.0)
        )

        rationale = [
            f"{name}:{obs.score:.2f}"
            for name, obs in sorted(modules.items(), key=lambda item: item[1].score, reverse=True)[:3]
        ]
        weak_modules = [name for name, obs in modules.items() if obs.score < 0.45 or obs.hard_veto]
        domain_notes = [
            obs.note for obs in modules.values()
            if obs.note in {"motif continuity", "confessional fit", "weak auditory gate", "template drift"}
        ]
        if weak_modules:
            rationale.append("weak=" + ",".join(sorted(weak_modules)))
        if domain_notes:
            rationale.append("cortex=" + ",".join(sorted(dict.fromkeys(domain_notes))))

        return WorkspaceCandidate(
            evidence=candidate,
            modules=modules,
            workspace_score=workspace_score,
            confidence=confidence,
            conflict=conflict,
            salience=salience,
            vetoed=vetoed,
            rationale=rationale,
        )

    def _build_module_observations(
        self,
        candidate: CandidateEvidence,
        context: WorkspaceContext,
        working_memory: WorkingMemoryState,
        self_model: SelfModel,
    ) -> dict[str, ModuleObservation]:
        text = candidate.text.strip()
        placeholder = text.lower() in PLACEHOLDER_MARKERS
        semantic_shape = self._semantic_shape_score(text)
        content_density = self._content_density(text)
        bar_position = context.bar_index % 8
        rhyme_due = context.target_end_phoneme is not None
        cortex = self.cortex.evaluate_candidate(text, context, working_memory.domain_memory)

        phonology_score = _weighted_mean([
            (candidate.phonetic_score, 0.45 if rhyme_due else 0.20),
            (candidate.polysyllabic_rhyme, 0.35),
            (candidate.internal_rhyme, 0.20),
        ])
        rhythm_score = _weighted_mean([
            (1.0 if candidate.syllable_ok else 0.0, 0.45),
            (candidate.stress_alignment, 0.35),
            (candidate.temporal_arc, 0.20),
        ])
        semantics_score = _weighted_mean([
            (semantic_shape, 0.55),
            (content_density, 0.20),
            (candidate.complexity, 0.25),
        ])
        emotion_score = _weighted_mean([
            (candidate.valence_fit, 0.25),
            (candidate.trajectory_fit, 0.35),
            (candidate.goosebump, 0.25),
            (candidate.introspection, 0.15),
        ])
        structure_score = _weighted_mean([
            (candidate.temporal_arc, 0.35),
            (candidate.hook_dna, 0.25),
            (candidate.texture_alignment, 0.20),
            (1.0 - abs(candidate.valence_fit - candidate.trajectory_fit), 0.20),
        ])
        novelty_score = _weighted_mean([
            (candidate.novelty_score, 0.60),
            (candidate.complexity, 0.25),
            (candidate.internal_rhyme, 0.15),
        ])
        auditory_score = _weighted_mean([
            (cortex["auditory_gate"], 0.45),
            (cortex["motor_fit"], 0.30),
            (candidate.texture_alignment, 0.25),
        ])
        template_score = _weighted_mean([
            (cortex["template_match"], 0.35),
            (cortex["confessional_alignment"], 0.20),
            (cortex["motif_continuity"], 0.15),
            (working_memory.domain_memory.domain_coherence, 0.15),
            (semantic_shape, 0.15),
        ])

        phonology_veto = rhyme_due and candidate.phonetic_score < 0.25
        rhythm_veto = (not candidate.syllable_ok) and candidate.stress_alignment < 0.25
        semantics_veto = placeholder or semantic_shape < 0.25
        structure_veto = bar_position in {6, 7} and candidate.temporal_arc < 0.20
        auditory_veto = not placeholder and cortex["auditory_gate"] < 0.18 and candidate.texture_alignment < 0.30
        template_veto = (
            working_memory.domain_memory.domain_coherence > 0.60
            and cortex["template_match"] < 0.18
            and context.section.lower() in {"chorus", "hook", "verse1", "verse2"}
        )

        cortex_note = ""
        if "weak auditory gate" in cortex["notes"]:
            cortex_note = "weak auditory gate"
        elif "template drift" in cortex["notes"]:
            cortex_note = "template drift"
        elif "motif continuity" in cortex["notes"]:
            cortex_note = "motif continuity"
        elif "confessional fit" in cortex["notes"]:
            cortex_note = "confessional fit"

        return {
            "phonology": ModuleObservation(
                module="phonology",
                score=_clamp(phonology_score),
                confidence=0.90 if rhyme_due else 0.72,
                weight=self.base_weights["phonology"] * self_model.weight_for("phonology"),
                note="rhyme slot" if rhyme_due else "free rhyme",
                hard_veto=phonology_veto,
            ),
            "rhythm": ModuleObservation(
                module="rhythm",
                score=_clamp(rhythm_score),
                confidence=0.82,
                weight=self.base_weights["rhythm"] * self_model.weight_for("rhythm"),
                note="motor timing alignment",
                hard_veto=rhythm_veto,
            ),
            "semantics": ModuleObservation(
                module="semantics",
                score=_clamp(semantics_score),
                confidence=0.74,
                weight=self.base_weights["semantics"] * self_model.weight_for("semantics"),
                note="line integrity",
                hard_veto=semantics_veto,
            ),
            "emotion": ModuleObservation(
                module="emotion",
                score=_clamp(emotion_score),
                confidence=0.80,
                weight=self.base_weights["emotion"] * self_model.weight_for("emotion"),
                note=f"mood={context.mood}",
            ),
            "structure": ModuleObservation(
                module="structure",
                score=_clamp(structure_score),
                confidence=0.78,
                weight=self.base_weights["structure"] * self_model.weight_for("structure"),
                note=f"bar={bar_position + 1}/8",
                hard_veto=structure_veto,
            ),
            "novelty": ModuleObservation(
                module="novelty",
                score=_clamp(novelty_score),
                confidence=0.72,
                weight=self.base_weights["novelty"] * self_model.weight_for("novelty"),
                note="anti-repetition pressure",
            ),
            "auditory": ModuleObservation(
                module="auditory",
                score=_clamp(auditory_score),
                confidence=0.80,
                weight=self.base_weights["auditory"] * self_model.weight_for("auditory"),
                note=cortex_note or "auditory hierarchy gate",
                hard_veto=auditory_veto,
            ),
            "template": ModuleObservation(
                module="template",
                score=_clamp(template_score),
                confidence=0.78,
                weight=self.base_weights["template"] * self_model.weight_for("template"),
                note=cortex_note or "broca-style template match",
                hard_veto=template_veto,
            ),
        }

    def _select_mode(
        self,
        candidates: list[WorkspaceCandidate],
        context: WorkspaceContext,
        working_memory: WorkingMemoryState,
    ) -> MetacognitiveState:
        if not candidates:
            return MetacognitiveState(
                mode="system2",
                confidence=0.0,
                conflict=1.0,
                novelty_pressure=0.0,
                emotional_salience=0.0,
                task_importance=working_memory.task_pressure,
                reasons=["no candidates available"],
                revision_focus=["semantics"],
                needs_regeneration=True,
                suggested_temperature=0.60,
            )

        top = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        score_gap = top.workspace_score - (runner_up.workspace_score if runner_up else 0.0)

        reasons: list[str] = []
        revision_focus = [
            name for name, obs in top.modules.items()
            if obs.score < 0.52 or obs.hard_veto
        ]
        novelty_pressure = _clamp(top.modules["novelty"].score)
        emotional_salience = _clamp(top.salience)
        task_importance = working_memory.task_pressure

        mode = "system1"
        if top.vetoed:
            mode = "system2"
            reasons.append("hard constraint conflict")
        if top.confidence < 0.58:
            mode = "system2"
            reasons.append("low confidence")
        if top.conflict > 0.33:
            mode = "system2"
            reasons.append("high inter-module conflict")
        if score_gap < 0.05:
            mode = "system2"
            reasons.append("workspace competition unresolved")
        if working_memory.remaining_resolution and top.modules["structure"].score < 0.55:
            mode = "system2"
            reasons.append("resolution bar needs stronger structure")
        if context.target_end_phoneme and top.modules["phonology"].score < 0.55:
            mode = "system2"
            reasons.append("rhyme target not secure")
        if top.modules["auditory"].hard_veto or top.modules["auditory"].score < 0.40:
            mode = "system2"
            reasons.append("auditory gate rejected the line")
        if top.modules["template"].hard_veto or top.modules["template"].score < 0.42:
            mode = "system2"
            reasons.append("template match drifted from song identity")

        if not reasons:
            reasons.append("top candidate clears active constraints")

        needs_regeneration = (
            mode == "system2"
            and (top.confidence < 0.46 or top.vetoed)
        )
        suggested_temperature = 0.85 if mode == "system1" else 0.64

        return MetacognitiveState(
            mode=mode,
            confidence=top.confidence,
            conflict=top.conflict,
            novelty_pressure=novelty_pressure,
            emotional_salience=emotional_salience,
            task_importance=task_importance,
            reasons=reasons,
            revision_focus=revision_focus or ["structure", "emotion"],
            needs_regeneration=needs_regeneration,
            suggested_temperature=suggested_temperature,
        )

    def _rerank_system2(
        self,
        candidates: list[WorkspaceCandidate],
        context: WorkspaceContext,
        metacognition: MetacognitiveState,
    ) -> list[WorkspaceCandidate]:
        weights = self._system2_weights(context, metacognition.revision_focus)
        reranked: list[WorkspaceCandidate] = []
        for candidate in candidates:
            deliberative_quality = _weighted_mean([
                (candidate.modules[name].score, weight * candidate.modules[name].confidence)
                for name, weight in weights.items()
            ])
            repair_bonus = 0.0
            for focus in metacognition.revision_focus:
                obs = candidate.modules.get(focus)
                if obs and not obs.hard_veto and obs.score >= 0.55:
                    repair_bonus += 0.03
            deliberative_score = _clamp(
                deliberative_quality * 0.68
                + candidate.evidence.base_score * 0.20
                + candidate.salience * 0.12
                + repair_bonus
                - candidate.conflict * 0.22
                - (0.15 if candidate.vetoed else 0.0)
            )
            reranked.append(
                WorkspaceCandidate(
                    evidence=candidate.evidence,
                    modules=candidate.modules,
                    workspace_score=deliberative_score,
                    confidence=_clamp(candidate.confidence + repair_bonus * 0.8),
                    conflict=candidate.conflict,
                    salience=candidate.salience,
                    vetoed=candidate.vetoed,
                    rationale=candidate.rationale + [f"system2:{','.join(sorted(weights))}"],
                )
            )
        reranked.sort(key=lambda item: item.workspace_score, reverse=True)
        return reranked

    def _system2_weights(
        self,
        context: WorkspaceContext,
        revision_focus: list[str],
    ) -> dict[str, float]:
        weights = dict(self.base_weights)
        for focus in revision_focus:
            if focus in weights:
                weights[focus] *= 1.20
        if context.target_end_phoneme:
            weights["phonology"] *= 1.15
        if context.section.lower() in {"chorus", "hook", "bridge"}:
            weights["structure"] *= 1.10
            weights["emotion"] *= 1.08
        if context.bar_index % 8 in {6, 7}:
            weights["structure"] *= 1.12
            weights["rhythm"] *= 1.08
        return weights

    def _choose_candidate(self, candidates: list[WorkspaceCandidate]) -> WorkspaceCandidate:
        for candidate in candidates:
            if not candidate.vetoed:
                return candidate
        return candidates[0]

    def _build_broadcast(
        self,
        chosen: WorkspaceCandidate,
        metacognition: MetacognitiveState,
        context: WorkspaceContext,
        hot_trace: dict,
    ) -> str:
        top_modules = sorted(
            chosen.modules.values(),
            key=lambda obs: obs.score,
            reverse=True,
        )[:2]
        module_summary = ", ".join(f"{obs.module}:{obs.score:.2f}" for obs in top_modules)
        next_focus = ", ".join(hot_trace.get("next_focus", [])[:2]) or "maintain flow"
        return (
            f"{metacognition.mode.upper()} broadcast on bar {context.bar_index + 1}: "
            f"'{chosen.evidence.text}' selected with workspace={chosen.workspace_score:.2f}, "
            f"confidence={chosen.confidence:.2f}, conflict={chosen.conflict:.2f}; "
            f"strongest modules: {module_summary}; next focus: {next_focus}"
        )

    def _semantic_shape_score(self, text: str) -> float:
        clean = text.strip()
        if not clean or clean.lower() in PLACEHOLDER_MARKERS:
            return 0.0
        words = [word for word in re.findall(r"[A-Za-z']+", clean) if word]
        if not words:
            return 0.0

        word_count = len(words)
        alpha_ratio = sum(ch.isalpha() or ch.isspace() or ch in "'?!" for ch in clean) / max(len(clean), 1)
        repeated = len(words) - len(set(word.lower() for word in words))
        repetition_penalty = repeated / max(word_count, 1)

        count_score = math.exp(-((word_count - 9) ** 2) / 24.0)
        ratio_score = _clamp(alpha_ratio)
        repeat_score = _clamp(1.0 - repetition_penalty * 0.8)
        return _clamp(_weighted_mean([
            (count_score, 0.45),
            (ratio_score, 0.30),
            (repeat_score, 0.25),
        ]))

    def _content_density(self, text: str) -> float:
        words = [word.lower() for word in re.findall(r"[A-Za-z']+", text)]
        if not words:
            return 0.0
        content_words = [word for word in words if word not in STOP_WORDS]
        unique_ratio = len(set(words)) / max(len(words), 1)
        density = len(content_words) / max(len(words), 1)
        return _clamp(_weighted_mean([
            (density, 0.55),
            (unique_ratio, 0.45),
        ]))


def summarize_workspace_history(history: list[dict]) -> dict:
    if not history:
        return {
            "mode_counts": {"system1": 0, "system2": 0},
            "avg_confidence": 0.0,
            "avg_conflict": 0.0,
            "common_revision_focus": [],
            "common_hot_focus": [],
            "last_self_model": {},
            "last_hot_trace": {},
        }

    mode_counter: Counter[str] = Counter()
    confidence: list[float] = []
    conflict: list[float] = []
    revision_counter: Counter[str] = Counter()
    hot_focus_counter: Counter[str] = Counter()
    last_self_model: dict = {}
    last_hot_trace: dict = {}

    for item in history:
        meta = item.get("metacognition", {})
        mode_counter[meta.get("mode", "system1")] += 1
        confidence.append(float(meta.get("confidence", 0.0)))
        conflict.append(float(meta.get("conflict", 0.0)))
        revision_counter.update(meta.get("revision_focus", []))
        last_self_model = item.get("self_model_snapshot", last_self_model)
        hot_trace = item.get("hot_trace", {})
        hot_focus_counter.update(hot_trace.get("next_focus", []))
        last_hot_trace = hot_trace or last_hot_trace

    return {
        "mode_counts": dict(mode_counter),
        "avg_confidence": round(float(np.mean(confidence)), 4) if confidence else 0.0,
        "avg_conflict": round(float(np.mean(conflict)), 4) if conflict else 0.0,
        "common_revision_focus": [name for name, _ in revision_counter.most_common(4)],
        "common_hot_focus": [name for name, _ in hot_focus_counter.most_common(4)],
        "last_self_model": last_self_model,
        "last_hot_trace": last_hot_trace,
    }
