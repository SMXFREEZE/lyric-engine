"""
Cortical Creative Loop (CCL) Training Dataset.

Brain-inspired training format based on:
- Predictive coding: predictions + prediction errors drive learning
- Creative cognition: Default Mode (generate) → Executive Control (evaluate) → Salience (select)
- Contrastive learning: learn from chosen vs rejected comparisons
- Error-driven learning: explicit error signals teach what NOT to do

The cortical loop:
  PERCEIVE → INTENT → PREDICT → ERROR → REVISE → SELECT → MEMORY

Training modes:
1. CCL-SFT: Full loop supervised fine-tuning
2. CCL-DPO: Direct Preference Optimization with chosen/rejected pairs
3. CCL-Curriculum: Progressive complexity (short → long, simple → complex)

This creates a self-supervised curriculum from existing lyrics data by using
scoring modules to synthesize perception, error, and preference signals.
"""

import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from configs.genres import (
    SECTION_TOKENS, ARC_TOKENS, CORTICAL_TOKENS, GENRE_DESCRIPTIONS
)
from src.data.rhyme_labeler import detect_scheme
from src.data.valence_scorer import score_lyrics, score_line, compute_song_arc


# ---------------------------------------------------------------------------
# Perception helpers: extract brain-like context signals
# ---------------------------------------------------------------------------

def _valence_label(val: float) -> str:
    """Categorical valence (maps to emotional tone)."""
    if val < -0.3:
        return "dark"
    elif val < -0.1:
        return "melancholic"
    elif val < 0.1:
        return "neutral"
    elif val < 0.3:
        return "warm"
    elif val < 0.5:
        return "uplifting"
    else:
        return "euphoric"


def _arousal_label(aro: float) -> str:
    """Categorical arousal (maps to energy level)."""
    if aro < 0.25:
        return "calm"
    elif aro < 0.45:
        return "moderate"
    elif aro < 0.65:
        return "energetic"
    else:
        return "intense"


def _rhythm_label(lines: list[str]) -> str:
    """Estimate rhythmic character from line structure."""
    if not lines:
        return "unknown"
    avg_words = sum(len(line.split()) for line in lines) / len(lines)
    avg_chars = sum(len(line) for line in lines) / len(lines)

    if avg_words < 4:
        return "sparse"
    elif avg_words < 7:
        return "moderate"
    elif avg_words < 10:
        return "dense"
    else:
        return "flowing"


def _rhyme_scheme_label(scheme_info: dict) -> str:
    """Convert rhyme scheme to token-friendly label."""
    scheme_type = scheme_info.get("scheme_type", "free")
    return f"[RHYME_{scheme_type.upper()}]" if scheme_type != "free" else "[RHYME_FREE]"


def _novelty_label(lines: list[str]) -> str:
    """Estimate novelty/predictability from vocabulary diversity."""
    if not lines:
        return "unknown"
    words = " ".join(lines).lower().split()
    if not words:
        return "unknown"
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.4:
        return "repetitive"
    elif unique_ratio < 0.6:
        return "moderate"
    elif unique_ratio < 0.8:
        return "varied"
    else:
        return "novel"


# ---------------------------------------------------------------------------
# Error detection: synthesize prediction errors from scoring
# ---------------------------------------------------------------------------

@dataclass
class ErrorSignal:
    """Represents detected errors/weaknesses in a draft."""
    emotion_error: float = 0.0    # Mismatch from target emotion
    rhythm_error: float = 0.0     # Rhythmic weakness
    novelty_error: float = 0.0    # Too predictable
    coherence_error: float = 0.0  # Contextual mismatch
    rhyme_error: float = 0.0      # Rhyme scheme violation

    @property
    def total_error(self) -> float:
        return (
            abs(self.emotion_error) * 0.3 +
            self.rhythm_error * 0.2 +
            self.novelty_error * 0.2 +
            self.coherence_error * 0.2 +
            self.rhyme_error * 0.1
        )

    def to_tokens(self) -> str:
        """Convert errors to cortical tokens."""
        tokens = []
        if abs(self.emotion_error) > 0.3:
            tokens.append("[ERR_EMOTION]")
        if self.rhythm_error > 0.4:
            tokens.append("[ERR_RHYTHM]")
        if self.novelty_error > 0.5:
            tokens.append("[ERR_NOVELTY]")
        if self.coherence_error > 0.4:
            tokens.append("[ERR_COHERENCE]")
        if self.rhyme_error > 0.3:
            tokens.append("[ERR_RHYME]")
        return " ".join(tokens) if tokens else "[NO_ERROR]"

    def to_text(self) -> str:
        """Human-readable error description."""
        parts = []
        if abs(self.emotion_error) > 0.2:
            direction = "too_dark" if self.emotion_error < 0 else "too_bright"
            parts.append(f"emotion={direction}")
        if self.rhythm_error > 0.3:
            parts.append(f"rhythm_weak={self.rhythm_error:.1f}")
        if self.novelty_error > 0.4:
            parts.append(f"predictable={self.novelty_error:.1f}")
        if self.coherence_error > 0.3:
            parts.append(f"incoherent={self.coherence_error:.1f}")
        if self.rhyme_error > 0.2:
            parts.append(f"rhyme_break={self.rhyme_error:.1f}")
        return " ".join(parts) if parts else "none"


def detect_errors(
    lines: list[str],
    target_valence: float,
    target_arousal: float,
    expected_rhyme: str = "free",
) -> ErrorSignal:
    """Detect prediction errors between draft and targets."""
    if not lines:
        return ErrorSignal()

    # Score actual output
    emotions = score_lyrics("\n".join(lines))
    actual_valence = sum(e.valence for e in emotions) / max(len(emotions), 1)
    actual_arousal = sum(e.arousal for e in emotions) / max(len(emotions), 1)

    # Emotion error: difference from target
    emotion_error = actual_valence - target_valence

    # Rhythm error: based on line consistency
    line_lengths = [len(line.split()) for line in lines]
    if len(line_lengths) > 1:
        rhythm_variance = np.std(line_lengths) / max(np.mean(line_lengths), 1)
        rhythm_error = min(1.0, rhythm_variance)
    else:
        rhythm_error = 0.0

    # Novelty error: low vocabulary diversity
    words = " ".join(lines).lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        novelty_error = max(0, 0.6 - unique_ratio)  # Error if < 60% unique
    else:
        novelty_error = 0.5

    # Coherence error: simplified - could use embedding similarity
    coherence_error = 0.0  # Placeholder for semantic coherence check

    # Rhyme error: check scheme
    try:
        scheme_info = detect_scheme(lines)
        actual_scheme = scheme_info.get("scheme_type", "free")
        if expected_rhyme != "free" and actual_scheme != expected_rhyme:
            rhyme_error = 0.5
        else:
            rhyme_error = 0.0
    except Exception:
        rhyme_error = 0.0

    return ErrorSignal(
        emotion_error=emotion_error,
        rhythm_error=rhythm_error,
        novelty_error=novelty_error,
        coherence_error=coherence_error,
        rhyme_error=rhyme_error,
    )


# ---------------------------------------------------------------------------
# Text parsing helpers
# ---------------------------------------------------------------------------

SECTION_HEADER_RE = re.compile(
    r"^\[(verse|chorus|prechorus|bridge|hook|outro|intro)\]",
    re.IGNORECASE
)


def split_into_sections(lyrics: str) -> list[tuple[str, list[str]]]:
    """Split raw lyrics into (section_name, lines) tuples."""
    sections: list[tuple[str, list[str]]] = []
    current_name = "VERSE"
    current_lines: list[str] = []

    for raw_line in lyrics.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = SECTION_HEADER_RE.match(line)
        if m:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = m.group(1).upper()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, current_lines))

    return sections if sections else [("VERSE", [l.strip() for l in lyrics.splitlines() if l.strip()])]


# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

_INST_OPEN = "[INST]"
_INST_CLOSE = "[/INST]"


def _build_instruction(genre: str, section: str, arc: str) -> str:
    """Build Mistral-instruct instruction prefix."""
    genre_desc = GENRE_DESCRIPTIONS.get(genre, genre)
    return f"{_INST_OPEN} Write {genre} lyrics for [{section}] ({arc}): {_INST_CLOSE}"


# ---------------------------------------------------------------------------
# CCL Format: Full cortical loop training example
# ---------------------------------------------------------------------------

def format_ccl_example(
    record: dict,
    include_error_phase: bool = True,
    include_revision: bool = False,  # Set True when we have revision data
    include_memory: bool = True,
) -> tuple[str, int]:
    """
    Format a song record into Cortical Creative Loop training format.

    The full loop:
      [INST]...[/INST]
      [PERCEIVE] context=... emotion=... rhythm=...
      [INTENT] arc=... target_emo=... rhyme=...
      [PREDICT]
      {draft lines}
      [ERROR] {error tokens}
      [REVISE]           # (optional, when revision data available)
      {revised lines}
      [SELECT]
      {final lines}
      [MEMORY] genre=... motif=...

    Returns: (full_text, instruction_char_len)
    """
    genre = record.get("genre", "trap")
    lyrics = record.get("lyrics", "")
    sections = split_into_sections(lyrics)

    if not sections:
        return "", 0

    # Compute arc tokens
    try:
        arc_tokens = compute_song_arc([lines for _, lines in sections])
    except Exception:
        arc_tokens = ["[SETUP]"] * len(sections)

    parts = []
    instruction_char_len = 0

    # Process each section as a cortical loop iteration
    for sec_idx, ((section_name, lines), arc) in enumerate(zip(sections, arc_tokens)):
        if not lines:
            continue

        # Compute perception signals for this section
        emotions = score_lyrics("\n".join(lines))
        avg_valence = sum(e.valence for e in emotions) / max(len(emotions), 1)
        avg_arousal = sum(e.arousal for e in emotions) / max(len(emotions), 1)

        try:
            scheme_info = detect_scheme(lines)
            rhyme_scheme = scheme_info.get("scheme_type", "free")
        except Exception:
            rhyme_scheme = "free"

        # ── INSTRUCTION ──
        section_tok = f"[{section_name}]" if f"[{section_name}]" in SECTION_TOKENS else "[VERSE]"
        instruction = _build_instruction(genre, section_name, arc)

        if sec_idx == 0:
            instruction_char_len = len(instruction)
            parts.append(instruction)
        else:
            # Subsequent sections don't add to instruction length (already counted)
            parts.append(f"\n{instruction}")

        # ── PERCEIVE (bottom-up context encoding) ──
        position = f"{sec_idx + 1}/{len(sections)}"
        perceive = (
            f"[PERCEIVE] [CONTEXT] section={section_name} position={position} "
            f"[EMO_STATE] valence={_valence_label(avg_valence)} arousal={_arousal_label(avg_arousal)} "
            f"[RHYTHM_STATE] flow={_rhythm_label(lines)}"
        )
        parts.append(perceive)

        # ── INTENT (goal formation) ──
        intent = (
            f"[INTENT] {arc} "
            f"[TARGET_EMO] {_valence_label(avg_valence)} "
            f"[TARGET_RHYTHM] {_rhythm_label(lines)} "
            f"[TARGET_NOVELTY] {_novelty_label(lines)}"
        )
        parts.append(intent)

        # ── PREDICT (initial draft - for now, same as final) ──
        parts.append("[PREDICT]")
        parts.extend(lines)

        # ── ERROR (prediction error detection) ──
        if include_error_phase:
            # For training, we synthesize errors from the actual output
            # This teaches the model what error signals look like
            errors = detect_errors(
                lines,
                target_valence=avg_valence,  # Use actual as target (no error)
                target_arousal=avg_arousal,
                expected_rhyme=rhyme_scheme,
            )
            error_tokens = errors.to_tokens()
            parts.append(f"[ERROR] {error_tokens}")

        # ── REVISE (optional - when we have revision pairs) ──
        if include_revision:
            parts.append("[REVISE]")
            parts.extend(lines)  # Same as predict for now

        # ── SELECT (final output) ──
        parts.append("[SELECT]")
        parts.extend(lines)

    # ── MEMORY (consolidation - compact state for continuity) ──
    if include_memory and sections:
        # Extract a potential motif (repeated phrase)
        all_lines = [line for _, lines in sections for line in lines]
        motif = ""
        if len(all_lines) >= 2:
            # Simple motif detection: look for repeated phrases
            from collections import Counter
            words = " ".join(all_lines[:4]).lower().split()
            word_counts = Counter(words)
            common = [w for w, c in word_counts.most_common(3) if c > 1 and len(w) > 3]
            if common:
                motif = common[0]

        memory = f"[MEMORY] genre={genre} sections={len(sections)} emotion={_valence_label(avg_valence)}"
        if motif:
            memory += f" [MOTIF] {motif}"
        parts.append(memory)

    full_text = "\n".join(parts)
    return full_text, instruction_char_len


# ---------------------------------------------------------------------------
# DPO Format: Generate preference pairs for contrastive selection learning
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    """A chosen/rejected pair for DPO training."""
    prompt: str
    chosen: str
    rejected: str
    margin: float  # Score difference (higher = clearer preference)


def generate_preference_pairs(
    record: dict,
    num_pairs: int = 1,
    rng: Optional[random.Random] = None,
) -> list[PreferencePair]:
    """
    Generate DPO preference pairs from a song record.

    Strategy: Create synthetic "rejected" examples by degrading the original:
    - Shuffle lines (breaks coherence)
    - Truncate (incomplete)
    - Add repetition (reduces novelty)

    The original lines are "chosen", degraded versions are "rejected".
    """
    if rng is None:
        rng = random.Random()

    genre = record.get("genre", "trap")
    lyrics = record.get("lyrics", "")
    sections = split_into_sections(lyrics)

    pairs = []

    for section_name, lines in sections:
        if len(lines) < 4:
            continue

        # Build the prompt (perception + intent without output)
        emotions = score_lyrics("\n".join(lines))
        avg_valence = sum(e.valence for e in emotions) / max(len(emotions), 1)
        avg_arousal = sum(e.arousal for e in emotions) / max(len(emotions), 1)

        try:
            scheme_info = detect_scheme(lines)
            rhyme_scheme = scheme_info.get("scheme_type", "free")
        except Exception:
            rhyme_scheme = "free"

        try:
            arc = compute_song_arc([lines])[0]
        except Exception:
            arc = "[SETUP]"

        prompt_parts = [
            f"[PERCEIVE] [CONTEXT] section={section_name} [EMO_STATE] valence={_valence_label(avg_valence)} arousal={_arousal_label(avg_arousal)}",
            f"[INTENT] {arc} [TARGET_EMO] {_valence_label(avg_valence)} rhyme={rhyme_scheme}",
            "[SELECT]",
        ]
        prompt = "\n".join(prompt_parts)

        # Chosen: original lines
        chosen = "\n".join(lines)

        # Generate rejected variants
        rejected_options = []

        # 1. Shuffled lines (breaks coherence/narrative)
        shuffled = lines.copy()
        rng.shuffle(shuffled)
        if shuffled != lines:
            rejected_options.append(("\n".join(shuffled), 0.4))

        # 2. First half only (incomplete)
        if len(lines) > 4:
            truncated = lines[:len(lines) // 2]
            rejected_options.append(("\n".join(truncated), 0.3))

        # 3. Doubled first line (excessive repetition)
        if len(lines) >= 2:
            repeated = [lines[0], lines[0]] + lines[2:]
            rejected_options.append(("\n".join(repeated), 0.35))

        # 4. Reversed lines
        reversed_lines = list(reversed(lines))
        if reversed_lines != lines:
            rejected_options.append(("\n".join(reversed_lines), 0.25))

        # Select pairs
        for i, (rejected, margin) in enumerate(rejected_options[:num_pairs]):
            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                margin=margin,
            ))

    return pairs


# ---------------------------------------------------------------------------
# Dataset Classes
# ---------------------------------------------------------------------------

class CorticalDataset(Dataset):
    """
    Cortical Creative Loop training dataset.

    Supports multiple training modes:
    - mode='ccl': Full cortical loop SFT
    - mode='simple': Basic completion SFT (fallback)
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
        mode: str = "ccl",  # 'ccl' or 'simple'
        include_error_phase: bool = True,
        include_memory: bool = True,
        curriculum_order: bool = True,
        max_per_artist: int = 3,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.include_error_phase = include_error_phase
        self.include_memory = include_memory

        # Load records
        records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if max_samples and len(records) >= max_samples * 2:
                    break

        # Apply curriculum ordering
        if curriculum_order:
            records = self._curriculum_sort(records, max_per_artist, seed)

        if max_samples and len(records) > max_samples:
            records = records[:max_samples]

        self.records = records
        print(f"[cortical_dataset] Loaded {len(self.records)} songs, mode={mode}")

    def _curriculum_sort(
        self,
        records: list[dict],
        max_per_artist: int,
        seed: int,
    ) -> list[dict]:
        """Sort for curriculum: short/simple first, cap per artist."""
        rng = random.Random(seed)

        # Cap per artist
        artist_counts: Counter = Counter()
        filtered = []
        shuffled = list(records)
        rng.shuffle(shuffled)

        for record in shuffled:
            artist = record.get("artist", "unknown")
            if artist_counts[artist] < max_per_artist:
                filtered.append(record)
                artist_counts[artist] += 1

        # Sort by complexity
        def complexity(r: dict) -> int:
            lyrics = r.get("lyrics", "")
            sections = split_into_sections(lyrics)
            total_lines = sum(len(lines) for _, lines in sections)
            return total_lines * 10 + len(sections) * 50

        filtered.sort(key=complexity)

        # Shuffle within bands for variety
        band_size = max(len(filtered) // 5, 1)
        result = []
        for i in range(0, len(filtered), band_size):
            band = filtered[i:i + band_size]
            rng.shuffle(band)
            result.extend(band)

        return result

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        if self.mode == "ccl":
            text, instruction_len = format_ccl_example(
                record,
                include_error_phase=self.include_error_phase,
                include_revision=False,
                include_memory=self.include_memory,
            )
        else:
            # Simple mode fallback
            text, instruction_len = self._format_simple(record)

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask padding
        labels[attention_mask == 0] = -100

        # Mask instruction prefix
        if instruction_len > 0:
            prefix_ids = self.tokenizer(
                text[:instruction_len],
                add_special_tokens=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
            n_prefix = len(prefix_ids)
            if n_prefix < self.max_length:
                labels[:n_prefix] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _format_simple(self, record: dict) -> tuple[str, int]:
        """Simple format fallback."""
        genre = record.get("genre", "trap")
        lyrics = record.get("lyrics", "")
        sections = split_into_sections(lyrics)

        instruction = f"{_INST_OPEN} Write {genre} lyrics: {_INST_CLOSE}"
        instruction_len = len(instruction)

        parts = [instruction]
        for section_name, lines in sections:
            section_tok = f"[{section_name}]" if f"[{section_name}]" in SECTION_TOKENS else "[VERSE]"
            parts.append(section_tok)
            parts.extend(lines)

        return "\n".join(parts), instruction_len


class DPODataset(Dataset):
    """
    Direct Preference Optimization dataset.

    Generates chosen/rejected pairs for contrastive selection learning.
    This teaches the model to prefer better outputs over worse ones.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
        pairs_per_song: int = 2,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = random.Random(seed)

        # Load records and generate preference pairs
        records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Generate preference pairs
        self.pairs: list[PreferencePair] = []
        for record in records:
            pairs = generate_preference_pairs(record, num_pairs=pairs_per_song, rng=self.rng)
            self.pairs.extend(pairs)
            if max_samples and len(self.pairs) >= max_samples:
                break

        if max_samples:
            self.pairs = self.pairs[:max_samples]

        print(f"[dpo_dataset] Generated {len(self.pairs)} preference pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        # Tokenize prompt
        prompt_enc = self.tokenizer(
            pair.prompt,
            max_length=self.max_length // 2,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize chosen
        chosen_enc = self.tokenizer(
            pair.chosen,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize rejected
        rejected_enc = self.tokenizer(
            pair.rejected,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "margin": torch.tensor(pair.margin, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_ccl_dataloaders(
    train_jsonl: str,
    val_jsonl: Optional[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 2,
    max_length: int = 512,
    mode: str = "ccl",
    max_per_artist: int = 3,
    num_workers: int = 0,
):
    """Create training and validation dataloaders."""
    from torch.utils.data import DataLoader

    train_ds = CorticalDataset(
        train_jsonl, tokenizer, max_length,
        mode=mode,
        curriculum_order=True,
        max_per_artist=max_per_artist,
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    val_dl = None
    if val_jsonl and Path(val_jsonl).exists():
        val_ds = CorticalDataset(
            val_jsonl, tokenizer, max_length,
            mode=mode,
            curriculum_order=False,
            max_per_artist=999,
        )
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl
