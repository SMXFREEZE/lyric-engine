"""
Inference engine: constrained beam search for god-tier lyrics.

Per-line generation flow:
  1. Build context: [genre token + LoRA] + [style prefix] + [section/arc tokens] + [accepted lines]
  2. Generate beam_size=8 candidate lines (DIVERGENT phase — high temperature, unconstrained)
     Mirrors: MPFC-active creative generation phase in professional rappers [PMC3498928]
  3. Post-score each candidate (CONVERGENT phase — full constraint battery):
     ORIGINAL:
       - phonetic constraint: end-rhyme match
       - syllable count: hard filter ±3
       - novelty: lexical overlap with accepted lines
       - valence fit: emotional arc target
     COGNITIVE ENGINE (emotional_geometry + phonosemantic + dopamine_arc):
       - 8D emotional trajectory fit
       - phonosemantic texture alignment (sound matches mood)
       - goosebump predictor (tension × emotional jump × hook DNA)
     RESEARCH-BACKED (research_scoring — 7 signals from peer-reviewed findings):
       - polysyllabic rhyme depth (biggest quality predictor, arXiv:2505.00035)
       - internal rhyme density (doubled in top artists since 2000)
       - quadratic complexity calibrator (optimal ~65th percentile)
       - 8-bar temporal arc weighting (final 2 bars = resolution priority)
       - introspection/confessional bonus (surged 246% in modern hip-hop)
       - vocabulary novelty (recency-weighted TTR)
       - rhythmic stress alignment (motor-rhythm coupling, PMC3498928)
  4. Emit top-1 (auto) or top-3 (co-write mode)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data.phoneme_annotator import annotate_line, LineAnnotation
from src.data.rhyme_labeler import rhymes
from src.data.valence_scorer import score_line, LineEmotion
from src.model.dual_tokenizer import PHONEME_TO_ID, word_to_phoneme_ids
from src.model.emotional_geometry import trajectory_fit_score, compute_song_arc
from src.model.phonosemantic import texture_alignment_score
from src.model.dopamine_arc import (
    goosebump_potential, hook_dna_score, TensionCurve, analyze_song_dopamine,
)
from src.model.metacognitive_engine import (
    CandidateEvidence,
    MetacognitiveEngine,
    SelfModel,
    WorkspaceContext,
    summarize_workspace_history,
)
from src.model.research_scoring import research_score

# ── New human-brain cognitive modules ────────────────────────────────────────
from src.model.felt_state import FeltStateEngine
from src.model.semantic_priming import SemanticPrimingEngine
from src.model.predictive_coder import PredictiveCoder, build_expectation
from src.model.surprise_engine import SurpriseEngine
from src.model.flow_controller import FlowController, RevisionInstinct, InhibitionOfReturn

# Style DNA — used for auto-populating SongMemory defaults and scoring
try:
    from src.data.style_dna import STYLES as _STYLE_LIBRARY, StyleDNA, style_to_prompt_prefix
    _HAS_STYLE_DNA = True
except ImportError:
    _HAS_STYLE_DNA = False
    StyleDNA = None  # type: ignore


_AUTO_DEFAULT = object()


# ── Syllable counter (deterministic, no LLM) ─────────────────────────────────

def count_syllables(line: str) -> int:
    return annotate_line(line).total_syllables


# ── Song memory (KV context of accepted lines) ───────────────────────────────

@dataclass
class SongMemory:
    genre: str
    mood: str = "dark"                      # dark / hype / romantic / chill / sad / epic
    style_vec: Optional[np.ndarray] = None  # (128,)
    sections: list[tuple[str, str]] = field(default_factory=list)  # (arc_token, section_name)
    accepted_lines: list[str] = field(default_factory=list)
    rhyme_scheme: str | object = _AUTO_DEFAULT        # AABB / ABAB / ABCB / free
    target_syllables: int | object = _AUTO_DEFAULT    # target per line
    used_end_phonemes: list[str] = field(default_factory=list)  # to avoid repetition
    tension_curve: TensionCurve = field(default_factory=TensionCurve)
    sections_lines: dict = field(default_factory=dict)  # track lines per section
    self_model: SelfModel = field(default_factory=SelfModel)
    workspace_history: list[dict] = field(default_factory=list)
    last_workspace: Optional[dict] = None
    style_dna: Optional[object] = None      # StyleDNA from style_dna.py

    # ── Human brain cognitive state ───────────────────────────────────────
    # These five systems model the pre-verbal, embodied, associative, predictive,
    # and flow-based processes that drive human lyric writing.
    felt_state_engine:    Optional[FeltStateEngine]     = field(default=None, repr=False)
    semantic_priming:     Optional[SemanticPrimingEngine] = field(default=None, repr=False)
    surprise_engine:      Optional[SurpriseEngine]      = field(default=None, repr=False)
    flow_controller:      Optional[FlowController]      = field(default=None, repr=False)
    inhibition_of_return: Optional[InhibitionOfReturn]  = field(default=None, repr=False)
    revision_instinct:    Optional[RevisionInstinct]    = field(default=None, repr=False)

    def __post_init__(self):
        """Auto-populate defaults from Style DNA when available."""
        explicit_rhyme_scheme = self.rhyme_scheme is not _AUTO_DEFAULT
        explicit_target_syllables = self.target_syllables is not _AUTO_DEFAULT

        if self.style_dna is None and _HAS_STYLE_DNA:
            self.style_dna = _STYLE_LIBRARY.get(self.genre)

        if explicit_target_syllables:
            self.target_syllables = int(self.target_syllables)
        elif self.style_dna is not None and hasattr(self.style_dna, "avg_syllables_per_line"):
            self.target_syllables = round(self.style_dna.avg_syllables_per_line)
        else:
            self.target_syllables = 10

        if explicit_rhyme_scheme:
            self.rhyme_scheme = str(self.rhyme_scheme)
        elif self.style_dna is not None and hasattr(self.style_dna, "rhyme_schemes") and self.style_dna.rhyme_schemes:
            self.rhyme_scheme = self.style_dna.rhyme_schemes[0]
        else:
            self.rhyme_scheme = "AABB"

        # Initialise human-brain cognitive systems
        initial_section = self.sections[-1][1] if self.sections else "intro"
        self.felt_state_engine    = FeltStateEngine(self.genre, initial_section)
        self.semantic_priming     = SemanticPrimingEngine(self.genre)
        self.surprise_engine      = SurpriseEngine()
        self.flow_controller      = FlowController(base_beam_size=8)
        self.inhibition_of_return = InhibitionOfReturn()
        self.revision_instinct    = RevisionInstinct()

    def add_line(self, line: str, section: str = "verse1", quality_score: float = 0.5):
        self.accepted_lines.append(line)
        ann = annotate_line(line)
        if ann.end_phoneme:
            self.used_end_phonemes.append(ann.end_phoneme)
        # Track in tension curve
        target_ep = self.get_target_end_phoneme()
        self.tension_curve.update(line, target_ep)
        # Track per-section
        if section not in self.sections_lines:
            self.sections_lines[section] = []
        self.sections_lines[section].append(line)

        # Update human-brain cognitive systems
        if self.felt_state_engine:
            self.felt_state_engine.contagion_update(line)
        if self.semantic_priming:
            self.semantic_priming.update(line)
        if self.inhibition_of_return:
            self.inhibition_of_return.mark_used(line)
        if self.flow_controller:
            self.flow_controller.update(quality_score)

    def get_target_end_phoneme(self) -> Optional[str]:
        """
        Based on rhyme scheme and lines generated so far,
        return the phoneme the next line should rhyme with (or None if free).
        """
        n = len(self.accepted_lines)
        if self.rhyme_scheme == "free":
            return None

        pairs = {
            "AABB": [(0, 1), (2, 3)],
            "ABAB": [(0, 2), (1, 3)],
            "ABCB": [(1, 3)],
        }
        for a_idx, b_idx in pairs.get(self.rhyme_scheme, []):
            if n % 4 == b_idx and n % 4 > 0:
                # This line should rhyme with line a_idx of this stanza
                stanza_start = (n // 4) * 4
                target_line_idx = stanza_start + a_idx
                if target_line_idx < len(self.accepted_lines):
                    ann = annotate_line(self.accepted_lines[target_line_idx])
                    return ann.end_phoneme
        return None

    def build_prompt(self) -> str:
        """Build the generation prompt in Mistral-instruct format.

        Training wraps data as ``[INST] … [/INST]{completion}``, so inference
        must use the same format.  The model generates everything after
        ``[/INST]``, which is where the genre header and accepted lines live.
        """
        parts: list[str] = []

        # Determine section structure for the instruction prefix
        if self.sections:
            arc, section = self.sections[-1]
            structure_hint = f"[{section}]"
        else:
            structure_hint = "[VERSE]"

        instruction = f"[INST] Write {self.genre} lyrics ({structure_hint}): [/INST]"
        parts.append(instruction)

        # Style DNA prefix for richer prompting
        if self.style_dna is not None and _HAS_STYLE_DNA:
            try:
                parts.append(style_to_prompt_prefix(self.style_dna))
            except Exception:
                pass
        parts.append(f"[GENRE_START] {self.genre} [GENRE_END]")

        # Inject felt state — emotion as generative INPUT not scoring criterion
        if self.felt_state_engine:
            parts.append(self.felt_state_engine.state.to_prompt_token())

        # Inject active semantic field — primed vocabulary cloud
        if self.semantic_priming:
            priming_fragment = self.semantic_priming.get_prompt_fragment(n=6)
            if priming_fragment:
                parts.append(priming_fragment)
        if self.sections:
            arc, section = self.sections[-1]
            parts.append(f"[{section}] {arc}")
        for line in self.accepted_lines[-20:]:  # last 20 lines as context
            parts.append(line)
        return "\n".join(parts) + "\n"


# ── Candidate scorer ──────────────────────────────────────────────────────────

@dataclass
class CandidateScore:
    text:               str
    phonetic_score:     float   # 0-1  end-rhyme match
    syllable_ok:        bool    # within ±3 syllables of target
    novelty_score:      float   # 0-1  lexical novelty
    valence_fit:        float   # 0-1  simple emotional fit
    # Cognitive engine
    trajectory_fit:     float   # 0-1  8D emotional geometry
    texture_alignment:  float   # 0-1  phonosemantic
    goosebump:          float   # 0-1  dopamine potential
    hook_dna:           float   # 0-1  hook pattern strength
    # Research-backed
    polysyllabic_rhyme: float   # 0-1  rhyme depth (arXiv:2505.00035)
    internal_rhyme:     float   # 0-1  within-line rhyme density
    complexity:         float   # 0-1  quadratic complexity target
    temporal_arc:       float   # 0-1  8-bar position alignment (PMC3498928)
    introspection:      float   # 0-1  confessional content signal
    stress_alignment:   float   # 0-1  motor-rhythm pattern
    vocab_novelty:      float   # 0-1  vocabulary freshness
    style_coherence:    float   # 0-1  Style DNA alignment
    base_score:         float
    total_score:        float
    workspace_score:    float = 0.0
    workspace_confidence: float = 0.0
    workspace_conflict: float = 0.0
    decision_mode:      str = "system1"
    decision_trace:     tuple[str, ...] = ()


def score_candidate(
    line: str,
    memory: SongMemory,
    target_arc_valence: float = 0.0,
    target_arc_arousal: float = 0.5,
    section: str = "verse1",
    mood: str = "dark",
    tension_state: float = 0.3,
    line_idx: int = 0,
) -> CandidateScore:
    ann = annotate_line(line)

    # 1. Phonetic (rhyme) score
    target_ep = memory.get_target_end_phoneme()
    if target_ep is None:
        phonetic_score = 1.0
    elif ann.end_phoneme and rhymes(ann.end_phoneme, target_ep):
        phonetic_score = 1.0
    else:
        phonetic_score = 0.0

    # 2. Syllable check
    syllable_ok = abs(ann.total_syllables - memory.target_syllables) <= 3

    # 3. Novelty
    def simple_overlap(a: str, b: str) -> float:
        aw = set(a.lower().split())
        bw = set(b.lower().split())
        return len(aw & bw) / len(aw | bw) if (aw and bw) else 0.0

    max_overlap = max(
        (simple_overlap(line, prev) for prev in memory.accepted_lines),
        default=0.0,
    )
    novelty_score = 1.0 - max_overlap

    # 4. Simple valence fit (legacy, low weight)
    emotion = score_line(line)
    val_diff = abs(emotion.valence - target_arc_valence)
    aro_diff = abs(emotion.arousal - target_arc_arousal)
    valence_fit = 1.0 - (val_diff + aro_diff) / 4.0

    # 5. 8D Emotional geometry trajectory fit
    try:
        traj_fit = trajectory_fit_score(line, memory.genre, section)
    except Exception:
        traj_fit = valence_fit  # graceful fallback

    # 6. Phonosemantic texture alignment
    try:
        tex_align = texture_alignment_score(line, mood, section)
    except Exception:
        tex_align = 0.5

    # 7. Dopamine / goosebump potential
    prev_line = memory.accepted_lines[-1] if memory.accepted_lines else None
    try:
        gbump = goosebump_potential(line, tension_state, prev_line, mood)
    except Exception:
        gbump = 0.0

    # 8. Hook DNA
    try:
        hook = hook_dna_score(line, prev_line)
    except Exception:
        hook = 0.0

    # 9-15. Research-backed scores (7 signals from peer-reviewed papers)
    target_ep = memory.get_target_end_phoneme()
    rhymes_target = bool(
        ann.end_phoneme and target_ep and
        __import__('src.data.rhyme_labeler', fromlist=['rhymes']).rhymes(ann.end_phoneme, target_ep)
    ) if target_ep else True

    try:
        rs = research_score(
            line=line,
            line_idx=line_idx,
            section=section,
            recent_lines=memory.accepted_lines[-8:],
            rhymes_with_target=rhymes_target,
            tension_state=tension_state,
            previous_line=prev_line,
        )
    except Exception:
        rs = {
            "polysyllabic_rhyme": 0.5, "internal_rhyme": 0.0,
            "complexity": 0.5, "temporal_arc": 0.5, "introspection": 0.0,
            "vocab_novelty": 0.5, "stress_alignment": 0.5,
        }

    # 16. Style coherence — does the line match the Style DNA expectations?
    style_coherence = 0.5  # neutral if no Style DNA
    if memory.style_dna is not None and hasattr(memory.style_dna, 'energy'):
        dna = memory.style_dna
        # Speechiness → expect more content words if high speechiness
        word_count = len(line.split())
        expected_words = dna.avg_syllables_per_line * (0.9 if dna.speechiness > 0.5 else 1.1)
        word_fit = max(0.0, 1.0 - abs(word_count - expected_words) / max(expected_words, 1) * 0.6)
        # Repetition ratio — if style is repetitive, penalise novelty less
        rep_tolerance = dna.repetition_ratio
        adjusted_novelty = novelty_score * (1.0 - rep_tolerance * 0.3)
        style_coherence = max(0.0, min(1.0, word_fit * 0.6 + adjusted_novelty * 0.4))

    # ── FINAL WEIGHTED SCORE ──────────────────────────────────────────────────
    # Four-layer scoring:
    #   Layer 1 (Original)  — ensures basic quality: rhyme, syllables, novelty
    #   Layer 2 (Cognitive) — emotional geometry, texture, dopamine prediction
    #   Layer 3 (Research)  — polysyllabic rhyme, internal rhyme, complexity arc
    #   Layer 4 (Style DNA) — genre-specific coherence
    total = (
        # Layer 1: basic quality (27%)
        0.11 * phonetic_score
        + 0.07 * (1.0 if syllable_ok else 0.0)
        + 0.05 * novelty_score
        + 0.04 * valence_fit
        # Layer 2: cognitive music engine (33%)
        + 0.11 * traj_fit
        + 0.07 * tex_align
        + 0.10 * gbump
        + 0.05 * hook
        # Layer 3: research-backed (33%)
        + 0.10 * rs["polysyllabic_rhyme"]
        + 0.06 * rs["internal_rhyme"]
        + 0.05 * rs["complexity"]
        + 0.04 * rs["temporal_arc"]
        + 0.04 * rs["introspection"]
        + 0.03 * rs["vocab_novelty"]
        + 0.01 * rs["stress_alignment"]
        # Layer 4: Style DNA (7%)
        + 0.07 * style_coherence
    )

    return CandidateScore(
        text=line,
        phonetic_score=phonetic_score,
        syllable_ok=syllable_ok,
        novelty_score=novelty_score,
        valence_fit=valence_fit,
        trajectory_fit=traj_fit,
        texture_alignment=tex_align,
        goosebump=gbump,
        hook_dna=hook,
        polysyllabic_rhyme=rs["polysyllabic_rhyme"],
        internal_rhyme=rs["internal_rhyme"],
        complexity=rs["complexity"],
        temporal_arc=rs["temporal_arc"],
        introspection=rs["introspection"],
        stress_alignment=rs["stress_alignment"],
        vocab_novelty=rs["vocab_novelty"],
        style_coherence=style_coherence,
        base_score=float(total),
        total_score=float(total),
    )


# ── Generation engine ─────────────────────────────────────────────────────────

class LyricsEngine:
    @staticmethod
    def _normalize_device_value(value: object) -> str:
        if isinstance(value, int):
            return f"cuda:{value}"
        text = str(value).strip()
        if text.isdigit():
            return f"cuda:{text}"
        return text

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
        beam_size: int = 8,
    ):
        runtime_device = device
        model_runtime_device = self._normalize_device_value(getattr(model, "device", device))
        hf_device_map = (
            getattr(model, "hf_device_map", None)
            or getattr(getattr(model, "base_model", None), "hf_device_map", None)
            or getattr(getattr(getattr(model, "base_model", None), "model", None), "hf_device_map", None)
        )
        if hf_device_map:
            self.model = model
            if model_runtime_device not in {"disk", "meta"}:
                runtime_device = model_runtime_device
            else:
                mapped = None
                for key, target in hf_device_map.items():
                    value = self._normalize_device_value(target)
                    if value in {"cpu", "disk", "meta"}:
                        continue
                    if any(token in key for token in ("embed", "wte", "input")):
                        mapped = value
                        break
                    if mapped is None:
                        mapped = value
                if mapped is not None:
                    runtime_device = mapped
        else:
            self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = runtime_device
        self.beam_size = beam_size
        self.metacognitive_engine = MetacognitiveEngine()

    def _input_device(self) -> str:
        try:
            device = self._normalize_device_value(getattr(self.model, "device", self.device))
            if device not in {"disk", "meta"}:
                return device
        except Exception:
            pass
        try:
            device = self._normalize_device_value(self.model.get_input_embeddings().weight.device)
            if device not in {"cpu", "disk", "meta"}:
                return device
        except Exception:
            pass
        return self._normalize_device_value(self.device)

    def _normalize_section_name(self, section: str) -> str:
        normalized = section.lower().strip("[] ")
        aliases = {
            "verse": "verse1",
            "prechorus": "pre_chorus",
            "pre-chorus": "pre_chorus",
            "pre_chorus": "pre_chorus",
        }
        return aliases.get(normalized, normalized)

    def _score_candidates(
        self,
        candidates: list[str],
        memory: SongMemory,
        target_arc_valence: float,
        target_arc_arousal: float,
        section: str,
        line_idx: int,
    ) -> list[CandidateScore]:
        return [
            score_candidate(
                candidate,
                memory,
                target_arc_valence,
                target_arc_arousal,
                section=section,
                mood=memory.mood,
                tension_state=memory.tension_curve.current,
                line_idx=line_idx,
            )
            for candidate in candidates
        ]

    def _workspace_context(
        self,
        memory: SongMemory,
        section: str,
        line_idx: int,
        target_arc_valence: float,
        target_arc_arousal: float,
    ) -> WorkspaceContext:
        return WorkspaceContext(
            genre=memory.genre,
            mood=memory.mood,
            section=section,
            bar_index=line_idx,
            rhyme_scheme=memory.rhyme_scheme,
            target_end_phoneme=memory.get_target_end_phoneme(),
            target_syllables=memory.target_syllables,
            target_arc_valence=target_arc_valence,
            target_arc_arousal=target_arc_arousal,
            tension_state=memory.tension_curve.current,
            accepted_lines=list(memory.accepted_lines),
        )

    def _rank_with_workspace(
        self,
        scored: list[CandidateScore],
        memory: SongMemory,
        section: str,
        line_idx: int,
        target_arc_valence: float,
        target_arc_arousal: float,
    ) -> tuple[list[CandidateScore], dict]:
        context = self._workspace_context(
            memory,
            section=section,
            line_idx=line_idx,
            target_arc_valence=target_arc_valence,
            target_arc_arousal=target_arc_arousal,
        )
        decision = self.metacognitive_engine.evaluate_candidates(
            [
                CandidateEvidence(
                    text=item.text,
                    base_score=item.base_score,
                    phonetic_score=item.phonetic_score,
                    syllable_ok=item.syllable_ok,
                    novelty_score=item.novelty_score,
                    valence_fit=item.valence_fit,
                    trajectory_fit=item.trajectory_fit,
                    texture_alignment=item.texture_alignment,
                    goosebump=item.goosebump,
                    hook_dna=item.hook_dna,
                    polysyllabic_rhyme=item.polysyllabic_rhyme,
                    internal_rhyme=item.internal_rhyme,
                    complexity=item.complexity,
                    temporal_arc=item.temporal_arc,
                    introspection=item.introspection,
                    stress_alignment=item.stress_alignment,
                )
                for item in scored
            ],
            context=context,
            self_model=memory.self_model,
        )

        by_text: dict[str, CandidateScore] = {}
        for item in scored:
            current = by_text.get(item.text)
            if current is None or item.base_score > current.base_score:
                by_text[item.text] = item

        ranked_scores: list[CandidateScore] = []
        for ranked in decision.ranked_candidates:
            item = by_text.get(ranked.evidence.text)
            if item is None:
                continue
            item.workspace_score = ranked.workspace_score
            item.workspace_confidence = ranked.confidence
            item.workspace_conflict = ranked.conflict
            item.decision_mode = decision.metacognition.mode
            item.decision_trace = tuple(ranked.rationale)
            fused_score = item.base_score * 0.72 + ranked.workspace_score * 0.28
            if ranked.vetoed:
                fused_score *= 0.92
            item.total_score = float(np.clip(fused_score, 0.0, 1.0))
            ranked_scores.append(item)

        if not ranked_scores:
            ranked_scores = sorted(scored, key=lambda item: item.total_score, reverse=True)

        return ranked_scores, decision.to_dict()

    def _merge_candidate_pools(
        self,
        original: list[CandidateScore],
        extra: list[CandidateScore],
    ) -> list[CandidateScore]:
        merged: dict[str, CandidateScore] = {}
        for item in original + extra:
            current = merged.get(item.text)
            if current is None or item.base_score > current.base_score:
                merged[item.text] = item
        return list(merged.values())

    @staticmethod
    def _first_nonempty_line(text: str) -> str:
        for raw in text.splitlines():
            line = raw.strip()
            if line:
                return line
        return text.strip()

    @torch.no_grad()
    def generate_candidates(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.85,
        top_p: float = 0.9,
    ) -> list[str]:
        """
        TWO-PHASE GENERATION — mirrors the MPFC/DLPFC cognitive process
        from freestyle rap neuroscience research [PMC3498928]:

        Phase 1 (DIVERGENT): high temperature, unconstrained, maximum creativity
          → Mirrors MPFC-active state: broad semantic associations, emotional access
          → Generates beam_size × 1.5 candidates with high novelty

        Phase 2 (CONVERGENT): lower temperature, top-k filtering
          → Mirrors DLPFC re-engagement: applies rhythmic/narrative structure
          → Generates beam_size × 0.5 candidates with tighter pattern adherence

        Combined pool is passed to the scorer (which acts as the evaluative phase).
        """
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )
        input_device = self._input_device()
        input_ids = enc["input_ids"].to(input_device)
        attention_mask = enc["attention_mask"].to(input_device)
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        eos = newline_ids[0] if len(newline_ids) == 1 else None

        candidates = []
        generation_errors: list[str] = []
        common_generate_kwargs = {
            "pad_token_id": self.tokenizer.eos_token_id,
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "repetition_penalty": 1.15,      # penalise token repetition in all phases
            "no_repeat_ngram_size": 3,       # block exact n-gram repeats
        }
        if eos is not None:
            common_generate_kwargs["eos_token_id"] = eos

        # ── Phase 1: Divergent (MPFC mode) ────────────────────────────────
        divergent_n = max(1, int(self.beam_size * 1.5))
        try:
            out_divergent = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                min_new_tokens=6,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=min(temperature * 1.3, 1.2),  # hotter = more creative
                top_p=0.98,                                # nearly unconstrained
                num_return_sequences=divergent_n,
                **common_generate_kwargs,
            )
            prompt_len = input_ids.shape[1]
            for out in out_divergent:
                text = self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
                line = self._first_nonempty_line(text)
                if line:
                    candidates.append(line)
        except Exception as exc:
            generation_errors.append(f"divergent:{exc}")

        # ── Phase 2: Convergent (DLPFC mode) ──────────────────────────────
        convergent_n = max(1, self.beam_size // 2)
        try:
            out_convergent = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                min_new_tokens=6,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature * 0.7, 0.5),  # cooler = more structured
                top_k=50,                                   # tighter vocabulary
                top_p=0.85,
                num_return_sequences=convergent_n,
                **common_generate_kwargs,
            )
            prompt_len = input_ids.shape[1]
            for out in out_convergent:
                text = self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
                line = self._first_nonempty_line(text)
                if line:
                    candidates.append(line)
        except Exception as exc:
            generation_errors.append(f"convergent:{exc}")

        if not candidates:
            try:
                fallback_kwargs = {
                    **common_generate_kwargs,
                    "repetition_penalty": 1.08,
                    "no_repeat_ngram_size": 3,
                }
                out_fallback = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    min_new_tokens=8,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=max(2, self.beam_size),
                    num_return_sequences=max(2, min(self.beam_size, 4)),
                    early_stopping=True,
                    **fallback_kwargs,
                )
                prompt_len = input_ids.shape[1]
                for out in out_fallback:
                    text = self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
                    line = self._first_nonempty_line(text)
                    if line:
                        candidates.append(line)
            except Exception as exc:
                generation_errors.append(f"fallback:{exc}")

        # Deduplicate preserving order
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        if unique:
            return unique
        if generation_errors:
            return [f"[generation failed: {generation_errors[0]}]"]
        return ["[no candidate generated]"]

    def generate_line(
        self,
        memory: SongMemory,
        target_arc_valence: float = 0.0,
        target_arc_arousal: float = 0.5,
        top_n: int = 1,
        section: str = "verse1",
    ) -> list[CandidateScore]:
        """
        Generate and rank candidates using the full human-brain cognitive pipeline.

        Generation order:
          1. Flow controller → adaptive temperature + beam size
          2. Felt state → emotion-conditioned prompt
          3. Semantic priming → active vocabulary cloud in prompt
          4. Divergent/convergent generation
          5. ITPRA expectation evaluation
          6. Inhibition of return → novelty suppression
          7. Metacognitive workspace → System 1/2 mode
          8. Surprise engine → violation × resolution final selection
          9. Revision instinct → targeted repair if needed
        """
        prompt = memory.build_prompt()
        section_key = self._normalize_section_name(section)

        # ── Step 1: Flow-adaptive generation parameters ───────────────────
        flow_snap = (
            memory.flow_controller.get_current_snapshot()
            if memory.flow_controller
            else None
        )
        generation_temp = (
            flow_snap.suggested_temperature
            if flow_snap
            else 0.85
        )
        # Felt state also modulates temperature
        if memory.felt_state_engine:
            generation_temp = (
                generation_temp * 0.6
                + memory.felt_state_engine.emotional_temperature() * 0.4
            )
        effective_beam = (
            flow_snap.suggested_beam_size if flow_snap else self.beam_size
        )

        line_idx = len(memory.accepted_lines)

        # ── Step 2: Build expectation (ITPRA — Prediction stage) ──────────
        expectation = build_expectation(
            accepted_lines=memory.accepted_lines,
            genre=memory.genre,
            section=section_key,
            target_end_phoneme=memory.get_target_end_phoneme(),
            target_syllables=memory.target_syllables,
        )

        # ── Step 3: Generate candidates ───────────────────────────────────
        candidates = self.generate_candidates(
            prompt,
            temperature=float(np.clip(generation_temp, 0.6, 1.2)),
        )

        # ── Step 4: Score candidates (existing pipeline) ──────────────────
        scored = self._score_candidates(
            candidates, memory,
            target_arc_valence, target_arc_arousal,
            section=section_key, line_idx=line_idx,
        )

        # ── Step 5: Inhibition of return — discount recently used vocab ───
        if memory.inhibition_of_return:
            for item in scored:
                suppression = memory.inhibition_of_return.suppression_score(item.text)
                # Novelty bonus already in score; apply suppression penalty
                item.novelty_score = float(np.clip(
                    item.novelty_score - suppression * 0.25, 0.0, 1.0
                ))
                item.total_score = float(np.clip(
                    item.total_score - suppression * 0.10, 0.0, 1.0
                ))

        # ── Step 6: Metacognitive workspace ───────────────────────────────
        ranked, workspace = self._rank_with_workspace(
            scored, memory, section=section_key, line_idx=line_idx,
            target_arc_valence=target_arc_valence,
            target_arc_arousal=target_arc_arousal,
        )

        # ── Step 7: Revision instinct — targeted repair if needed ─────────
        meta = workspace.get("metacognition", {})
        if meta.get("needs_regeneration") and memory.revision_instinct and ranked:
            revision_target = memory.revision_instinct.diagnose(
                ranked[0].text,
                ranked[0],
                target_syllables=memory.target_syllables,
                target_phoneme=memory.get_target_end_phoneme(),
            )
            if revision_target:
                repair_temp = float(np.clip(
                    float(meta.get("suggested_temperature", 0.64))
                    + revision_target.temperature_modifier,
                    0.5, 1.0,
                ))
                repair_prompt = memory.revision_instinct.build_repair_prompt(
                    prompt, revision_target
                )
                repair_candidates = self.generate_candidates(repair_prompt, temperature=repair_temp)
                repair_scored = self._score_candidates(
                    repair_candidates, memory,
                    target_arc_valence, target_arc_arousal,
                    section=section_key, line_idx=line_idx,
                )
                combined = self._merge_candidate_pools(scored, repair_scored)
                ranked, workspace = self._rank_with_workspace(
                    combined, memory, section=section_key, line_idx=line_idx,
                    target_arc_valence=target_arc_valence,
                    target_arc_arousal=target_arc_arousal,
                )
            else:
                # Fallback: standard repair without targeted prompt
                repair_candidates = self.generate_candidates(
                    prompt, temperature=float(meta.get("suggested_temperature", 0.64)), top_p=0.82,
                )
                repair_scored = self._score_candidates(
                    repair_candidates, memory,
                    target_arc_valence, target_arc_arousal,
                    section=section_key, line_idx=line_idx,
                )
                combined = self._merge_candidate_pools(scored, repair_scored)
                ranked, workspace = self._rank_with_workspace(
                    combined, memory, section=section_key, line_idx=line_idx,
                    target_arc_valence=target_arc_valence,
                    target_arc_arousal=target_arc_arousal,
                )

        # ── Step 8: Surprise engine — violation × resolution selection ────
        if memory.surprise_engine and ranked:
            previous_line = memory.accepted_lines[-1] if memory.accepted_lines else None
            surprise_decision = memory.surprise_engine.select(
                candidates=[r.text for r in ranked],
                expectation=expectation,
                tension=memory.tension_curve.current,
                section=section_key,
                bar_position=line_idx % 8,
                line_idx=line_idx,
                previous_line=previous_line,
            )
            workspace["surprise_decision"] = {
                "was_surprise": surprise_decision.was_surprise,
                "probability": round(surprise_decision.surprise_probability, 3),
                "violation": round(surprise_decision.violation_score, 3),
                "resolution": round(surprise_decision.resolution_score, 3),
                "reason": surprise_decision.reason,
            }
            # Re-order ranked so the surprise decision's choice is first
            if surprise_decision.was_surprise:
                chosen_text = surprise_decision.chosen_text
                reordered = [r for r in ranked if r.text == chosen_text]
                rest = [r for r in ranked if r.text != chosen_text]
                ranked = reordered + rest

        memory.last_workspace = workspace
        memory.workspace_history.append(workspace)

        ok = [item for item in ranked if item.syllable_ok]
        ranked = ok or ranked
        return ranked[:top_n]

    def generate_verse(
        self,
        memory: SongMemory,
        num_lines: int = 8,
        section: str = "VERSE",
        arc_token: str = "[SETUP]",
        auto_accept: bool = True,
    ) -> list[str]:
        """
        Generate a full verse, accepting top-1 each time and updating memory.
        """
        section_key = self._normalize_section_name(section)
        memory.sections.append((arc_token, section))

        # Estimate arc valence from token
        arc_valence_map = {
            "[SETUP]": (0.0, 0.3),
            "[BUILD]": (0.1, 0.6),
            "[RELEASE]": (0.5, 0.8),
            "[REFRAME]": (-0.1, 0.4),
            "[PEAK]": (0.6, 1.0),
            "[OUTRO]": (-0.2, 0.2),
        }
        target_val, target_aro = arc_valence_map.get(arc_token, (0.0, 0.5))

        memory.tension_curve.reset_for_section(section)
        memory.sections_lines[section_key] = []

        # Transition all cognitive systems to new section
        if memory.felt_state_engine:
            memory.felt_state_engine.section_transition(section_key)
        if memory.semantic_priming:
            memory.semantic_priming.reset_for_section()
        if memory.surprise_engine:
            memory.surprise_engine.reset_for_section(section_key)

        generated_lines = []
        for _ in range(num_lines):
            top = self.generate_line(memory, target_val, target_aro, top_n=1, section=section_key)
            if top:
                best = top[0]
                if auto_accept:
                    memory.add_line(best.text, section=section_key, quality_score=best.total_score)
                generated_lines.append(best.text)

        return generated_lines

    def analyze_song(self, memory: SongMemory) -> dict:
        """
        Full cognitive analysis of a generated song.
        Shows artists:
          - The 8D emotional arc across all sections
          - The tension-release curve
          - Every line's goosebump potential score
          - Hook DNA patterns detected
          - The single strongest moment in the whole song
          - Phonosemantic texture per section

        This is what no other AI music tool provides.
        """
        sections = {
            sec: lines
            for sec, lines in memory.sections_lines.items()
            if lines
        }
        if not sections:
            # Fallback: build from accepted lines
            sections = {"verse1": memory.accepted_lines}

        emotional_arc  = compute_song_arc(sections, memory.genre)
        dopamine_arc   = analyze_song_dopamine(sections, memory.mood)

        # Texture summary per section
        texture_summary = {}
        for sec, lines in sections.items():
            try:
                from src.model.phonosemantic import analyze_line_texture
                profiles = [analyze_line_texture(l) for l in lines]
                avg = np.mean([p.to_array() for p in profiles], axis=0)
                from src.model.phonosemantic import TextureProfile
                avg_profile = TextureProfile.from_array(avg)
                texture_summary[sec] = avg_profile.describe()
            except Exception:
                texture_summary[sec] = "n/a"

        return {
            "emotional_arc":  emotional_arc,
            "dopamine_arc":   dopamine_arc,
            "texture_summary": texture_summary,
            "peak_moment": dopamine_arc.get("peak_moment", {}),
            "metacognition": summarize_workspace_history(memory.workspace_history),
            "last_workspace": memory.last_workspace,
            "genre": memory.genre,
            "mood":  memory.mood,
        }


# ── Co-write session ──────────────────────────────────────────────────────────

class CoWriteSession:
    """
    Interactive co-write mode.
    User can accept, reject, or edit each suggested line.
    Model updates context and generates next line aware of accepted lines.
    """

    def __init__(self, engine: LyricsEngine, genre: str, rhyme_scheme: str = "AABB"):
        self.engine = engine
        self.memory = SongMemory(genre=genre, rhyme_scheme=rhyme_scheme)
        self.history: list[dict] = []  # audit trail

    def start_section(self, section: str = "VERSE", arc_token: str = "[SETUP]"):
        self.memory.sections.append((arc_token, section))
        print(f"\n── {section} {arc_token} ──")

    def suggest(self, n: int = 3) -> list[CandidateScore]:
        """Suggest n candidates for the next line."""
        return self.engine.generate_line(self.memory, top_n=n)

    def accept(self, line: str):
        """Accept a line (user-written or from suggestions)."""
        self.memory.add_line(line)
        self.history.append({"action": "accept", "line": line})
        print(f"  ✓ {line}")

    def reject(self):
        """Skip — regenerate next suggestion."""
        self.history.append({"action": "reject"})

    def get_song(self) -> str:
        return "\n".join(self.memory.accepted_lines)


if __name__ == "__main__":
    # Local smoke test with GPT-2 (no GPU needed)
    from transformers import AutoModelForCausalLM, AutoTokenizer as HFTokenizer

    print("Loading GPT-2 for local smoke test...")
    tok = HFTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained("gpt2")

    engine = LyricsEngine(mdl, tok, device="cpu", beam_size=3)
    memory = SongMemory(genre="hip_hop", rhyme_scheme="AABB", target_syllables=10)
    memory.sections.append(("[SETUP]", "VERSE"))

    print("\nGenerating candidates for first line...")
    candidates = engine.generate_line(memory, top_n=3)
    for i, c in enumerate(candidates):
        print(f"  [{i+1}] (score={c.total_score:.2f}) {c.text}")
