"""
Training dataset: reads annotated JSONL, formats examples for the LLM.

Each training example format:
  [GENRE_START] trap [GENRE_END] [STYLE_START] <style_prefix> [STYLE_END]
  [VERSE] [SETUP]
  <lyrics line 1>
  <lyrics line 2>
  ...
  [CHORUS] [RELEASE]
  <chorus lines>
  ...

Phoneme IDs are stored as a parallel sequence for the phonetic head.
"""

import json
import os
from pathlib import Path
from typing import Optional
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from configs.genres import SECTION_TOKENS, ARC_TOKENS, SPECIAL_TOKENS
from src.data.phoneme_annotator import annotate_lyrics, annotation_to_dict
from src.data.rhyme_labeler import detect_scheme
from src.data.valence_scorer import score_lyrics, compute_song_arc
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
# Training wraps data in [INST]...[/INST] so the base model's instruction-following
# alignment helps rather than fights fine-tuning.  Labels for instruction tokens
# are set to -100 so loss is only computed on the lyric completion.
_INST_OPEN  = "[INST]"
_INST_CLOSE = "[/INST]"

# Stop words excluded from call_echo overlap ratio to avoid false positives
# on function-word-heavy lines like "I was lost / I was found".
_STOP_WORDS_ECHO = {
    "i", "you", "he", "she", "we", "they", "it", "me", "my", "your",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "have", "had", "do", "did", "not", "no", "so",
    "that", "this", "just", "like", "get", "got", "will", "can",
}


def _build_instruction(genre: str, sections: list[tuple[str, list[str]]]) -> str:
    """Build the [INST] instruction prefix for a training example."""
    structure = " → ".join(
        f"[{name}]" if f"[{name}]" in SECTION_TOKENS else "[VERSE]"
        for name, _ in sections
    )
    return f"{_INST_OPEN} Write {genre} lyrics ({structure}): {_INST_CLOSE}"


def format_training_example(
    record: dict,
    style_vec: Optional[np.ndarray] = None,
) -> tuple[str, int]:
    """
    Format a song record into the Mistral-instruct training format.

    Returns
    -------
    (full_text, instruction_char_len)
        ``instruction_char_len`` is the number of characters in the
        ``[INST]…[/INST]`` prefix so the caller can mask those tokens.
    """
    genre = record.get("genre", "hip_hop")
    lyrics = record.get("lyrics", "")
    sections = split_into_sections(lyrics)

    # Compute arc tokens per section — guarded: malformed lyrics can raise
    try:
        arc_tokens = compute_song_arc([lines for _, lines in sections])
    except Exception:
        arc_tokens = ["[SETUP]"] * len(sections)

    # Build the Mistral-instruct instruction header
    instruction = _build_instruction(genre, sections)
    instruction_char_len = len(instruction)

    # Build the completion (everything the model should learn to generate)
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


class LyricsDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        style_vec_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.style_vec_dir = Path(style_vec_dir) if style_vec_dir else None

        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
                if max_samples and len(self.records) >= max_samples:
                    break

        print(f"[dataset] Loaded {len(self.records)} songs from {jsonl_path}")

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
                # Some raw collaboration names can exceed filesystem limits when
                # translated directly into legacy cache filenames. Just skip the
                # unusable legacy path and continue.
                continue
        return None

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        style_vec = self._load_style_vec(record.get("artist", ""))

        text, instruction_char_len = format_training_example(record, style_vec)
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

        # Mask instruction prefix tokens — only compute loss on the lyric completion.
        # Encode the instruction prefix alone (no padding) to measure its token length.
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
):
    from torch.utils.data import DataLoader

    train_ds = LyricsDataset(train_jsonl, tokenizer, max_length, style_vec_dir)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dl = None
    if val_jsonl and Path(val_jsonl).exists():
        val_ds = LyricsDataset(val_jsonl, tokenizer, max_length, style_vec_dir)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl
