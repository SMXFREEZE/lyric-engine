"""
Composer cortex primitives inspired by domain memory, auditory hierarchy, and
higher-order self-monitoring.

This module does not claim phenomenal consciousness. It implements functional
building blocks that make the generator behave more like a composing agent:

- Domain memory / habitus: what the song has already established
- Auditory gate: whether a candidate feels musically "engageable"
- Template matching: whether a line fits the active genre/section grammar
- HOT trace: an explicit self-representation of why a line was selected
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "he", "her", "his", "i", "if", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "they", "this", "to", "us", "was", "we", "with", "you", "your",
}

FIRST_PERSON = {"i", "im", "i'm", "ive", "i've", "me", "my", "mine", "myself"}
TIME_MARKERS = {
    "now", "then", "before", "after", "forever", "tonight", "today", "yesterday",
    "tomorrow", "memory", "remember", "again", "still", "once",
}

SECTION_PRIORS = {
    "intro": {"target_words": 7.0, "hook_bias": 0.30, "confessional": 0.45},
    "verse1": {"target_words": 10.0, "hook_bias": 0.35, "confessional": 0.55},
    "verse2": {"target_words": 10.0, "hook_bias": 0.40, "confessional": 0.58},
    "chorus": {"target_words": 8.0, "hook_bias": 0.85, "confessional": 0.52},
    "chorus2": {"target_words": 8.0, "hook_bias": 0.88, "confessional": 0.50},
    "hook": {"target_words": 7.0, "hook_bias": 0.92, "confessional": 0.48},
    "bridge": {"target_words": 9.0, "hook_bias": 0.60, "confessional": 0.70},
    "outro": {"target_words": 7.0, "hook_bias": 0.45, "confessional": 0.62},
    "pre_chorus": {"target_words": 8.0, "hook_bias": 0.55, "confessional": 0.50},
}

GENRE_PRIORS = {
    "trap": {"confessional": 0.32, "density": 0.72, "motion": 0.70},
    "hip_hop": {"confessional": 0.52, "density": 0.70, "motion": 0.62},
    "drill": {"confessional": 0.28, "density": 0.74, "motion": 0.76},
    "rnb": {"confessional": 0.68, "density": 0.48, "motion": 0.42},
    "pop": {"confessional": 0.48, "density": 0.52, "motion": 0.50},
    "country": {"confessional": 0.62, "density": 0.54, "motion": 0.45},
    "rock": {"confessional": 0.44, "density": 0.58, "motion": 0.60},
    "afrobeats": {"confessional": 0.40, "density": 0.55, "motion": 0.68},
    "indie": {"confessional": 0.66, "density": 0.44, "motion": 0.38},
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return 0.0
    return float(sum(score * weight for score, weight in pairs) / total_weight)


def _content_words(text: str) -> list[str]:
    return [
        token.lower() for token in re.findall(r"[A-Za-z']+", text)
        if token and token.lower() not in STOP_WORDS
    ]


def _first_person_ratio(text: str) -> float:
    words = [token.lower() for token in re.findall(r"[A-Za-z']+", text)]
    if not words:
        return 0.0
    hits = sum(1 for token in words if token in FIRST_PERSON)
    return hits / max(len(words), 1)


@dataclass
class DomainMemory:
    motif_tokens: list[str] = field(default_factory=list)
    confessional_bias: float = 0.5
    narrative_pressure: float = 0.5
    grammar_richness: float = 0.5
    motor_readiness: float = 0.5
    domain_coherence: float = 0.5
    section_target_words: float = 8.0
    hook_bias: float = 0.5

    def to_dict(self) -> dict:
        return {
            "motif_tokens": self.motif_tokens,
            "confessional_bias": round(self.confessional_bias, 4),
            "narrative_pressure": round(self.narrative_pressure, 4),
            "grammar_richness": round(self.grammar_richness, 4),
            "motor_readiness": round(self.motor_readiness, 4),
            "domain_coherence": round(self.domain_coherence, 4),
            "section_target_words": round(self.section_target_words, 4),
            "hook_bias": round(self.hook_bias, 4),
        }


class ComposerCortex:
    def build_domain_memory(self, context) -> DomainMemory:
        section_prior = SECTION_PRIORS.get(context.section.lower(), SECTION_PRIORS["verse1"])
        genre_prior = GENRE_PRIORS.get(context.genre.lower(), {"confessional": 0.5, "density": 0.55, "motion": 0.5})
        lines = context.accepted_lines[-8:]

        words = [token for line in lines for token in re.findall(r"[A-Za-z']+", line.lower())]
        content = [token for token in words if token not in STOP_WORDS]
        counts = Counter(content)
        motif_tokens = [token for token, _ in counts.most_common(6)]

        if lines:
            mean_len = sum(len(_content_words(line)) for line in lines) / max(len(lines), 1)
            unique_ratio = len(set(content)) / max(len(content), 1) if content else 0.0
            first_person = sum(_first_person_ratio(line) for line in lines) / max(len(lines), 1)
            time_density = sum(
                sum(1 for token in re.findall(r"[A-Za-z']+", line.lower()) if token in TIME_MARKERS)
                for line in lines
            ) / max(len(words), 1)
        else:
            mean_len = section_prior["target_words"]
            unique_ratio = 0.45
            first_person = genre_prior["confessional"]
            time_density = 0.12

        grammar_richness = _clamp(_weighted_mean([
            (unique_ratio, 0.40),
            (math.exp(-((mean_len - section_prior["target_words"]) ** 2) / 18.0), 0.30),
            (genre_prior["density"], 0.30),
        ]))
        motor_readiness = _clamp(_weighted_mean([
            (genre_prior["motion"], 0.45),
            (context.tension_state, 0.25),
            (section_prior["hook_bias"], 0.30),
        ]))
        domain_coherence = _clamp(_weighted_mean([
            (1.0 - abs(unique_ratio - genre_prior["density"]) * 0.8, 0.45),
            (1.0 - min(time_density * 2.0, 1.0), 0.20),
            (0.5 + min(len(motif_tokens), 6) / 12.0, 0.35),
        ]))

        return DomainMemory(
            motif_tokens=motif_tokens,
            confessional_bias=_clamp(first_person * 0.65 + genre_prior["confessional"] * 0.35),
            narrative_pressure=_clamp(time_density * 2.2),
            grammar_richness=grammar_richness,
            motor_readiness=motor_readiness,
            domain_coherence=domain_coherence,
            section_target_words=section_prior["target_words"],
            hook_bias=section_prior["hook_bias"],
        )

    def evaluate_candidate(self, text: str, context, memory: DomainMemory) -> dict:
        words = [token.lower() for token in re.findall(r"[A-Za-z']+", text)]
        content = [token for token in words if token not in STOP_WORDS]
        word_count = len(content)

        target_score = math.exp(-((word_count - memory.section_target_words) ** 2) / 16.0)
        motif_overlap = 0.0
        if memory.motif_tokens and content:
            motif_overlap = len(set(content) & set(memory.motif_tokens)) / max(len(set(memory.motif_tokens)), 1)

        first_person = _first_person_ratio(text)
        confessional_alignment = 1.0 - min(abs(first_person - memory.confessional_bias) * 1.8, 1.0)

        short_long_alternation = 0.0
        if len(content) >= 2:
            lengths = [len(token) for token in content[:10]]
            alternations = sum(
                1 for idx in range(1, len(lengths))
                if (lengths[idx] <= 4) != (lengths[idx - 1] <= 4)
            )
            short_long_alternation = alternations / max(len(lengths) - 1, 1)

        punctuation_energy = min((text.count("!") + text.count("?")) * 0.2, 1.0)
        auditory_gate = _clamp(_weighted_mean([
            (target_score, 0.35),
            (short_long_alternation, 0.20),
            (memory.motor_readiness, 0.25),
            (0.4 + punctuation_energy * 0.6, 0.20),
        ]))

        template_match = _clamp(_weighted_mean([
            (target_score, 0.30),
            (motif_overlap, 0.20),
            (confessional_alignment, 0.20),
            (memory.grammar_richness, 0.15),
            (memory.domain_coherence, 0.15),
        ]))

        notes: list[str] = []
        if motif_overlap > 0:
            notes.append("motif continuity")
        if confessional_alignment > 0.7:
            notes.append("confessional fit")
        if auditory_gate < 0.35:
            notes.append("weak auditory gate")
        if template_match < 0.4:
            notes.append("template drift")

        return {
            "auditory_gate": auditory_gate,
            "template_match": template_match,
            "confessional_alignment": _clamp(confessional_alignment),
            "motif_continuity": _clamp(motif_overlap),
            "motor_fit": _clamp(_weighted_mean([
                (memory.motor_readiness, 0.5),
                (short_long_alternation, 0.3),
                (target_score, 0.2),
            ])),
            "notes": notes,
        }

    def build_hot_trace(self, chosen, context, working_memory, metacognition) -> dict:
        sorted_modules = sorted(
            chosen.modules.values(),
            key=lambda obs: obs.score,
            reverse=True,
        )
        winning_modules = [obs.module for obs in sorted_modules[:3]]
        weak_modules = [obs.module for obs in sorted_modules if obs.score < 0.5 or obs.hard_veto]

        next_focus = list(dict.fromkeys(
            metacognition.revision_focus
            + (["phonology"] if context.target_end_phoneme else [])
            + (["structure"] if working_memory.remaining_resolution else [])
        ))[:4]

        self_statement = (
            f"I selected '{chosen.evidence.text}' because {winning_modules[0]} won the workspace, "
            f"{metacognition.mode} mode judged confidence={chosen.confidence:.2f}, "
            f"and the next focus is {', '.join(next_focus) if next_focus else 'maintain flow'}."
        )

        return {
            "self_statement": self_statement,
            "winning_modules": winning_modules,
            "risk_modules": weak_modules[:4],
            "active_constraints": working_memory.active_constraints,
            "next_focus": next_focus,
        }
