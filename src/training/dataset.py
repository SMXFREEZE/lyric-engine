"""
Training dataset: reads annotated JSONL, formats examples for the LLM.

Metacognitive Training Format:
  [INST] Write {genre} lyrics ({structure}): [/INST]
  [PLAN] emotion={valence_label} arousal={arousal_label} rhyme={scheme} arc={arc_token}
  [SECTION_GOAL] {section_description}
  [DRAFT]
  {lyrics}
  [FINAL]
  {lyrics again - teaches model to commit after planning}

Curriculum ordering (when enabled):
  1. Short/clean examples first (< 8 lines, single section)
  2. Medium examples (8-16 lines, 2-3 sections)
  3. Full/complex examples

Phoneme IDs are stored as a parallel sequence for the phonetic head.
"""

import json
import os
from pathlib import Path
from typing import Optional
from collections import Counter
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from configs.genres import SECTION_TOKENS, ARC_TOKENS, SPECIAL_TOKENS, GENRE_DESCRIPTIONS
from src.data.phoneme_annotator import annotate_lyrics, annotation_to_dict
from src.data.rhyme_labeler import detect_scheme
from src.data.valence_scorer import score_lyrics, compute_song_arc, score_line
from src.model.dual_tokenizer import word_to_phoneme_ids, PHONEME_TO_ID


SECTION_HEADER_RE = __import__("re").compile(r"^\[(verse|chorus|prechorus|bridge|hook|outro)\]", __import__("re").IGNORECASE)


def _artist_cache_name(artist: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", artist.strip())
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("._")
    return safe or "unknown_artist"


def split_into_sections(lyrics: str) -> list[tuple[str, list[str]]]:
    """
    Split raw lyrics into (section_name, lines) tuples.
    If no section markers, treat as single [VERSE].
    """
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


# Instruction prefix template for Mistral-Instruct.
_INST_OPEN  = "[INST]"
_INST_CLOSE = "[/INST]"


def _valence_label(val: float) -> str:
    """Convert numeric valence to categorical label."""
    if val < -0.3:
        return "dark"
    elif val < 0.0:
        return "melancholic"
    elif val < 0.3:
        return "neutral"
    elif val < 0.6:
        return "uplifting"
    else:
        return "euphoric"


def _arousal_label(aro: float) -> str:
    """Convert numeric arousal to categorical label."""
    if aro < 0.3:
        return "calm"
    elif aro < 0.5:
        return "moderate"
    elif aro < 0.7:
        return "energetic"
    else:
        return "intense"


def _section_goal(genre: str, section_name: str, arc_token: str) -> str:
    """Generate a section goal description based on genre, section, and arc."""
    genre_desc = GENRE_DESCRIPTIONS.get(genre, "lyrical content")

    arc_goals = {
        "[SETUP]": "establish theme and atmosphere",
        "[BUILD]": "increase tension and momentum",
        "[RELEASE]": "deliver emotional payoff",
        "[REFRAME]": "shift perspective or introduce contrast",
        "[PEAK]": "reach maximum intensity",
        "[OUTRO]": "conclude and reflect",
    }
    arc_goal = arc_goals.get(arc_token, "continue the narrative")

    section_roles = {
        "VERSE": "storytelling",
        "CHORUS": "hook and core message",
        "PRECHORUS": "build anticipation",
        "BRIDGE": "contrast and shift",
        "HOOK": "memorable catchphrase",
        "OUTRO": "resolution",
    }
    section_role = section_roles.get(section_name, "development")

    return f"{section_role}: {arc_goal}"


def _build_instruction(genre: str, sections: list[tuple[str, list[str]]]) -> str:
    """Build the [INST] instruction prefix for a training example."""
    structure = " -> ".join(
        f"[{name}]" if f"[{name}]" in SECTION_TOKENS else "[VERSE]"
        for name, _ in sections
    )
    return f"{_INST_OPEN} Write {genre} lyrics ({structure}): {_INST_CLOSE}"


def format_training_example_metacog(
    record: dict,
    style_vec: Optional[np.ndarray] = None,
) -> tuple[str, int]:
    """
    Format a song record into metacognitive training format.

    Format:
      [INST] Write {genre} lyrics ({structure}): [/INST]
      [PLAN] emotion={valence} arousal={arousal} rhyme={scheme} arc={arc}
      [SECTION_GOAL] {goal}
      [DRAFT]
      {lyrics}
      [FINAL]
      {lyrics}

    Returns
    -------
    (full_text, instruction_char_len)
    """
    genre = record.get("genre", "hip_hop")
    lyrics = record.get("lyrics", "")
    sections = split_into_sections(lyrics)

    # Compute arc tokens and emotions per section
    try:
        arc_tokens = compute_song_arc([lines for _, lines in sections])
    except Exception:
        arc_tokens = ["[SETUP]"] * len(sections)

    # Build instruction header
    instruction = _build_instruction(genre, sections)
    instruction_char_len = len(instruction)

    # Build metacognitive completion
    completion_parts = []

    for (section_name, lines), arc in zip(sections, arc_tokens):
        if not lines:
            continue

        # Score emotions for this section
        section_text = "\n".join(lines)
        emotions = score_lyrics(section_text)
        avg_valence = sum(e.valence for e in emotions) / max(len(emotions), 1)
        avg_arousal = sum(e.arousal for e in emotions) / max(len(emotions), 1)

        # Detect rhyme scheme
        try:
            scheme_info = detect_scheme(lines)
            rhyme = scheme_info.get("scheme_type", "free")
        except Exception:
            rhyme = "free"

        # Plan signal
        plan = f"[PLAN] emotion={_valence_label(avg_valence)} arousal={_arousal_label(avg_arousal)} rhyme={rhyme} arc={arc}"

        # Section goal
        goal = f"[SECTION_GOAL] {_section_goal(genre, section_name, arc)}"

        # Section token
        section_tok = f"[{section_name}]" if f"[{section_name}]" in SECTION_TOKENS else "[VERSE]"

        # Build section output
        completion_parts.append(f"{section_tok} {plan}")
        completion_parts.append(goal)
        completion_parts.append("[DRAFT]")
        completion_parts.extend(lines)
        completion_parts.append("[FINAL]")
        completion_parts.extend(lines)  # Repeat for [FINAL] - teaches committing

    completion = "\n".join(completion_parts)
    full_text = f"{instruction}\n{completion}"
    return full_text, instruction_char_len


def format_training_example_simple(
    record: dict,
    style_vec: Optional[np.ndarray] = None,
) -> tuple[str, int]:
    """
    Simple format (original) - for comparison or fallback.

    Format:
      [INST] Write {genre} lyrics ({structure}): [/INST]
      [GENRE_START] {genre} [GENRE_END]
      [SECTION] [ARC]
      {lyrics}
    """
    genre = record.get("genre", "hip_hop")
    lyrics = record.get("lyrics", "")
    sections = split_into_sections(lyrics)

    try:
        arc_tokens = compute_song_arc([lines for _, lines in sections])
    except Exception:
        arc_tokens = ["[SETUP]"] * len(sections)

    instruction = _build_instruction(genre, sections)
    instruction_char_len = len(instruction)

    completion_parts = [f"[GENRE_START] {genre} [GENRE_END]"]
    if style_vec is not None:
        completion_parts.append("[STYLE_START] [STYLE_END]")

    for (section_name, lines), arc in zip(sections, arc_tokens):
        section_tok = f"[{section_name}]" if f"[{section_name}]" in SECTION_TOKENS else "[VERSE]"
        completion_parts.append(f"{section_tok} {arc}")
        completion_parts.extend(lines)

    completion = "\n".join(completion_parts)
    full_text = f"{instruction}\n{completion}"
    return full_text, instruction_char_len


# Legacy alias
format_training_example = format_training_example_simple


def compute_example_complexity(record: dict) -> tuple[int, int, int]:
    """
    Compute complexity metrics for curriculum ordering.

    Returns: (total_lines, num_sections, avg_line_length)
    """
    lyrics = record.get("lyrics", "")
    sections = split_into_sections(lyrics)

    total_lines = sum(len(lines) for _, lines in sections)
    num_sections = len(sections)

    all_lines = [line for _, lines in sections for line in lines]
    avg_len = sum(len(line) for line in all_lines) // max(len(all_lines), 1)

    return (total_lines, num_sections, avg_len)


def curriculum_sort_records(
    records: list[dict],
    max_per_artist: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    Sort records for curriculum learning:
    1. Cap songs per artist to reduce overconcentration
    2. Sort by complexity (short/simple first)
    3. Shuffle within complexity bands for variety

    Returns sorted list of records.
    """
    rng = random.Random(seed)

    # Cap per artist
    artist_counts: Counter = Counter()
    filtered = []

    # Shuffle first to randomize which songs from each artist are kept
    shuffled = list(records)
    rng.shuffle(shuffled)

    for record in shuffled:
        artist = record.get("artist", "unknown")
        if artist_counts[artist] < max_per_artist:
            filtered.append(record)
            artist_counts[artist] += 1

    # Compute complexity for each
    with_complexity = []
    for record in filtered:
        lines, sections, avg_len = compute_example_complexity(record)
        # Complexity score: prioritize short, single-section, moderate line length
        complexity = lines * 10 + sections * 50 + abs(avg_len - 40)
        with_complexity.append((complexity, record))

    # Sort by complexity (simple first)
    with_complexity.sort(key=lambda x: x[0])

    # Group into complexity bands and shuffle within each
    band_size = max(len(with_complexity) // 5, 1)
    result = []
    for i in range(0, len(with_complexity), band_size):
        band = [r for _, r in with_complexity[i:i + band_size]]
        rng.shuffle(band)
        result.extend(band)

    return result


class LyricsDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        style_vec_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        use_metacog_format: bool = True,
        curriculum_order: bool = True,
        max_per_artist: int = 3,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.style_vec_dir = Path(style_vec_dir) if style_vec_dir else None
        self.use_metacog_format = use_metacog_format

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
                    # Load extra for filtering
                    break

        # Apply curriculum ordering if enabled
        if curriculum_order:
            records = curriculum_sort_records(records, max_per_artist=max_per_artist, seed=seed)

        # Limit to max_samples after filtering
        if max_samples and len(records) > max_samples:
            records = records[:max_samples]

        self.records = records
        print(f"[dataset] Loaded {len(self.records)} songs from {jsonl_path}")
        print(f"[dataset] Format: {'metacognitive' if use_metacog_format else 'simple'}, curriculum: {curriculum_order}")

    def __len__(self):
        return len(self.records)

    def _load_style_vec(self, artist: str) -> Optional[np.ndarray]:
        if self.style_vec_dir is None:
            return None
        candidate_paths = [
            self.style_vec_dir / f"{_artist_cache_name(artist)}.npy",
            self.style_vec_dir / f"{artist.replace(' ', '_')}.npy",
        ]
        for path in candidate_paths:
            try:
                if path.exists():
                    return np.load(str(path)).astype(np.float32)
            except OSError:
                continue
        return None

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        style_vec = self._load_style_vec(record.get("artist", ""))

        # Choose format
        if self.use_metacog_format:
            text, instruction_char_len = format_training_example_metacog(record, style_vec)
        else:
            text, instruction_char_len = format_training_example_simple(record, style_vec)

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

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        # Mask instruction prefix tokens - only compute loss on completion
        instruction_prefix = text[:instruction_char_len]
        prefix_ids = self.tokenizer(
            instruction_prefix,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        n_prefix = len(prefix_ids)
        if n_prefix < self.max_length:
            labels[:n_prefix] = -100

        # Build phoneme IDs
        words = text.split()
        ph_ids: list[int] = []
        for w in words:
            ph_ids.extend(word_to_phoneme_ids(w))
        if len(ph_ids) >= self.max_length:
            phoneme_ids = torch.tensor(ph_ids[:self.max_length], dtype=torch.long)
        else:
            pad = [PHONEME_TO_ID["<PAD_PH>"]] * (self.max_length - len(ph_ids))
            phoneme_ids = torch.tensor(ph_ids + pad, dtype=torch.long)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "phoneme_ids": phoneme_ids,
        }

        if style_vec is not None:
            item["style_vec"] = torch.tensor(style_vec, dtype=torch.float32)

        return item


def create_dataloaders(
    train_jsonl: str,
    val_jsonl: Optional[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 1024,
    style_vec_dir: Optional[str] = None,
    num_workers: int = 0,
    use_metacog_format: bool = True,
    curriculum_order: bool = True,
    max_per_artist: int = 3,
):
    from torch.utils.data import DataLoader

    train_ds = LyricsDataset(
        train_jsonl, tokenizer, max_length, style_vec_dir,
        use_metacog_format=use_metacog_format,
        curriculum_order=curriculum_order,
        max_per_artist=max_per_artist,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # shuffle=False because curriculum_sort already ordered the data

    val_dl = None
    if val_jsonl and Path(val_jsonl).exists():
        val_ds = LyricsDataset(
            val_jsonl, tokenizer, max_length, style_vec_dir,
            use_metacog_format=use_metacog_format,
            curriculum_order=False,  # Don't curriculum-sort validation
            max_per_artist=999,
        )
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl
