"""
Continuous dataset sync + threshold-based training loop.

This does not magically learn from "all music". It continuously:
  1. reads one or more JSONL lyric datasets
  2. deduplicates and rebuilds data/raw/all_songs.jsonl + per-genre JSONLs
  3. tracks how many new songs arrived since the last training run
  4. triggers stage 1 / stage 2 SFT only when thresholds are met

Typical usage:
  python scripts/continuous_learning.py run-once --train --genres trap,pop
  python scripts/continuous_learning.py run-loop --train --genres trap --interval-minutes 60
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.sft import train_sft

app = typer.Typer(add_completion=False)

AUTOMATION_DIR = ROOT / "data" / "automation"
DEFAULT_STATE_PATH = AUTOMATION_DIR / "continuous_learning_state.json"
DEFAULT_MANIFEST_PATH = AUTOMATION_DIR / "dataset_manifest.json"


@dataclass
class SyncSummary:
    total_songs: int
    genre_counts: dict[str, int]
    source_paths: list[str]
    output_paths: dict[str, str]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "created_at": utc_now_iso(),
        "runs": 0,
        "last_total_songs": 0,
        "last_trained_total": 0,
        "last_genre_counts": {},
        "last_trained_genre_counts": {},
        "checkpoints": {},
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def normalize_record(record: dict[str, Any], default_genre: Optional[str]) -> dict[str, Any] | None:
    artist = str(record.get("artist", "")).strip()
    title = str(record.get("title", "")).strip()
    lyrics = str(record.get("lyrics", "")).strip()
    genre = str(record.get("genre") or default_genre or "").strip().lower()

    if not artist or not title or not lyrics:
        return None
    if not genre:
        genre = "unknown"

    normalized = dict(record)
    normalized["artist"] = artist
    normalized["title"] = title
    normalized["lyrics"] = lyrics
    normalized["genre"] = genre
    return normalized


def dedupe_records(
    source_paths: list[Path],
    default_genre: Optional[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    genre_counts: dict[str, int] = defaultdict(int)

    for path in source_paths:
        for raw_record in iter_jsonl_records(path):
            record = normalize_record(raw_record, default_genre)
            if not record:
                continue
            key = (record["artist"].lower(), record["title"].lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
            genre_counts[record["genre"]] += 1

    deduped.sort(key=lambda row: (row["genre"], row["artist"].lower(), row["title"].lower()))
    return deduped, dict(sorted(genre_counts.items()))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sync_dataset_views(
    source_paths: list[Path],
    raw_dir: Path,
    default_genre: Optional[str],
    manifest_path: Path,
) -> SyncSummary:
    deduped, genre_counts = dedupe_records(source_paths, default_genre)

    raw_dir.mkdir(parents=True, exist_ok=True)
    all_path = raw_dir / "all_songs.jsonl"
    write_jsonl(all_path, deduped)

    per_genre: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in deduped:
        per_genre[record["genre"]].append(record)

    output_paths = {"all": str(all_path)}
    for genre, records in per_genre.items():
        genre_path = raw_dir / f"{genre}.jsonl"
        write_jsonl(genre_path, records)
        output_paths[genre] = str(genre_path)

    manifest = {
        "updated_at": utc_now_iso(),
        "total_songs": len(deduped),
        "genre_counts": genre_counts,
        "source_paths": [str(path) for path in source_paths],
        "output_paths": output_paths,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return SyncSummary(
        total_songs=len(deduped),
        genre_counts=genre_counts,
        source_paths=[str(path) for path in source_paths],
        output_paths=output_paths,
    )


def maybe_train(
    *,
    summary: SyncSummary,
    state: dict[str, Any],
    train_enabled: bool,
    stage1: bool,
    genres: list[str],
    min_new_songs: int,
    min_genre_songs: int,
    min_stage1_songs: int,
    base_model: str,
    output_dir: str,
    batch_size: int,
    grad_accum_steps: int,
    epochs: int,
    lr: float,
    lora_rank: int,
    use_4bit: bool,
    style_vec_dir: Optional[str],
) -> list[str]:
    actions: list[str] = []
    if not train_enabled:
        return actions

    last_trained_total = int(state.get("last_trained_total", 0))
    last_trained_genre_counts = state.get("last_trained_genre_counts", {})
    checkpoints = state.setdefault("checkpoints", {})

    if stage1:
        delta_total = summary.total_songs - last_trained_total
        if summary.total_songs >= min_stage1_songs and delta_total >= min_new_songs:
            actions.append(f"stage1:{summary.total_songs}")
            train_sft(
                stage=1,
                data_path=summary.output_paths["all"],
                base_model=base_model,
                output_dir=output_dir,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                epochs=1,
                lr=lr,
                lora_rank=max(lora_rank, 32),
                use_4bit=use_4bit,
                style_vec_dir=style_vec_dir,
            )
            state["last_trained_total"] = summary.total_songs
            checkpoints["stage1"] = f"{output_dir}/stage1_general/epoch_1"

    target_genres = genres or list(summary.genre_counts.keys())
    for genre in target_genres:
        current_count = int(summary.genre_counts.get(genre, 0))
        previous_count = int(last_trained_genre_counts.get(genre, 0))
        delta = current_count - previous_count
        if current_count < min_genre_songs or delta < min_new_songs:
            continue

        genre_path = summary.output_paths.get(genre)
        if not genre_path:
            continue

        actions.append(f"stage2:{genre}:{current_count}")
        train_sft(
            stage=2,
            genre=genre,
            data_path=genre_path,
            base_model=base_model,
            output_dir=output_dir,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            epochs=epochs,
            lr=lr,
            lora_rank=lora_rank,
            use_4bit=use_4bit,
            style_vec_dir=style_vec_dir,
        )
        state.setdefault("last_trained_genre_counts", {})[genre] = current_count
        checkpoints[genre] = f"{output_dir}/genre_{genre}/epoch_{epochs}"

    return actions


def run_cycle(
    *,
    source: str,
    extra_sources: str,
    raw_dir: str,
    default_genre: Optional[str],
    state_path: str,
    manifest_path: str,
    train: bool,
    stage1: bool,
    genres: str,
    min_new_songs: int,
    min_genre_songs: int,
    min_stage1_songs: int,
    base_model: str,
    output_dir: str,
    batch_size: int,
    grad_accum_steps: int,
    epochs: int,
    lr: float,
    lora_rank: int,
    use_4bit: bool,
    style_vec_dir: Optional[str],
) -> dict[str, Any]:
    state_file = Path(state_path)
    manifest_file = Path(manifest_path)
    source_paths = [Path(source)]
    source_paths.extend(Path(item) for item in parse_csv_list(extra_sources))

    state = load_state(state_file)
    summary = sync_dataset_views(
        source_paths=source_paths,
        raw_dir=Path(raw_dir),
        default_genre=default_genre,
        manifest_path=manifest_file,
    )

    actions = maybe_train(
        summary=summary,
        state=state,
        train_enabled=train,
        stage1=stage1,
        genres=parse_csv_list(genres),
        min_new_songs=min_new_songs,
        min_genre_songs=min_genre_songs,
        min_stage1_songs=min_stage1_songs,
        base_model=base_model,
        output_dir=output_dir,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        epochs=epochs,
        lr=lr,
        lora_rank=lora_rank,
        use_4bit=use_4bit,
        style_vec_dir=style_vec_dir,
    )

    state["runs"] = int(state.get("runs", 0)) + 1
    state["last_total_songs"] = summary.total_songs
    state["last_genre_counts"] = summary.genre_counts
    state["last_sync_at"] = utc_now_iso()
    if actions:
        state["last_train_actions"] = actions
        state["last_train_at"] = utc_now_iso()

    save_state(state_file, state)

    report = {
        "total_songs": summary.total_songs,
        "genre_counts": summary.genre_counts,
        "actions": actions,
        "state_path": str(state_file),
        "manifest_path": str(manifest_file),
    }
    return report


COMMON_OPTIONS = {
    "source": typer.Option("data/crawl/songs.jsonl", help="Primary JSONL source to monitor."),
    "extra_sources": typer.Option("", help="Comma-separated extra JSONL sources to merge."),
    "raw_dir": typer.Option("data/raw", help="Output directory for rebuilt raw datasets."),
    "default_genre": typer.Option(None, help="Fill missing genre values with this."),
    "state_path": typer.Option(str(DEFAULT_STATE_PATH), help="Persistent loop state file."),
    "manifest_path": typer.Option(str(DEFAULT_MANIFEST_PATH), help="Dataset manifest output file."),
    "train": typer.Option(False, help="Trigger training when thresholds are met."),
    "stage1": typer.Option(False, help="Allow general stage 1 retraining on large dataset growth."),
    "genres": typer.Option("", help="Comma-separated genres to auto-train. Empty means all detected genres."),
    "min_new_songs": typer.Option(200, help="Minimum number of new songs before retraining."),
    "min_genre_songs": typer.Option(100, help="Minimum songs required in a genre before stage 2 training."),
    "min_stage1_songs": typer.Option(1000, help="Minimum total songs before stage 1 retraining."),
    "base_model": typer.Option("mistralai/Mistral-7B-Instruct-v0.2", help="Base model for SFT."),
    "output_dir": typer.Option("checkpoints", help="Checkpoint output directory."),
    "batch_size": typer.Option(4, help="Training batch size."),
    "grad_accum_steps": typer.Option(8, help="Gradient accumulation steps."),
    "epochs": typer.Option(3, help="Stage 2 training epochs."),
    "lr": typer.Option(2e-4, help="Learning rate."),
    "lora_rank": typer.Option(64, help="LoRA rank."),
    "use_4bit": typer.Option(False, help="Use 4-bit quantization during training."),
    "style_vec_dir": typer.Option("data/style_vectors", help="Style vector directory."),
}


@app.command("run-once")
def run_once(
    source: str = COMMON_OPTIONS["source"],
    extra_sources: str = COMMON_OPTIONS["extra_sources"],
    raw_dir: str = COMMON_OPTIONS["raw_dir"],
    default_genre: Optional[str] = COMMON_OPTIONS["default_genre"],
    state_path: str = COMMON_OPTIONS["state_path"],
    manifest_path: str = COMMON_OPTIONS["manifest_path"],
    train: bool = COMMON_OPTIONS["train"],
    stage1: bool = COMMON_OPTIONS["stage1"],
    genres: str = COMMON_OPTIONS["genres"],
    min_new_songs: int = COMMON_OPTIONS["min_new_songs"],
    min_genre_songs: int = COMMON_OPTIONS["min_genre_songs"],
    min_stage1_songs: int = COMMON_OPTIONS["min_stage1_songs"],
    base_model: str = COMMON_OPTIONS["base_model"],
    output_dir: str = COMMON_OPTIONS["output_dir"],
    batch_size: int = COMMON_OPTIONS["batch_size"],
    grad_accum_steps: int = COMMON_OPTIONS["grad_accum_steps"],
    epochs: int = COMMON_OPTIONS["epochs"],
    lr: float = COMMON_OPTIONS["lr"],
    lora_rank: int = COMMON_OPTIONS["lora_rank"],
    use_4bit: bool = COMMON_OPTIONS["use_4bit"],
    style_vec_dir: Optional[str] = COMMON_OPTIONS["style_vec_dir"],
):
    report = run_cycle(
        source=source,
        extra_sources=extra_sources,
        raw_dir=raw_dir,
        default_genre=default_genre,
        state_path=state_path,
        manifest_path=manifest_path,
        train=train,
        stage1=stage1,
        genres=genres,
        min_new_songs=min_new_songs,
        min_genre_songs=min_genre_songs,
        min_stage1_songs=min_stage1_songs,
        base_model=base_model,
        output_dir=output_dir,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        epochs=epochs,
        lr=lr,
        lora_rank=lora_rank,
        use_4bit=use_4bit,
        style_vec_dir=style_vec_dir,
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("run-loop")
def run_loop(
    interval_minutes: int = typer.Option(60, help="Minutes between sync/train checks."),
    source: str = COMMON_OPTIONS["source"],
    extra_sources: str = COMMON_OPTIONS["extra_sources"],
    raw_dir: str = COMMON_OPTIONS["raw_dir"],
    default_genre: Optional[str] = COMMON_OPTIONS["default_genre"],
    state_path: str = COMMON_OPTIONS["state_path"],
    manifest_path: str = COMMON_OPTIONS["manifest_path"],
    train: bool = COMMON_OPTIONS["train"],
    stage1: bool = COMMON_OPTIONS["stage1"],
    genres: str = COMMON_OPTIONS["genres"],
    min_new_songs: int = COMMON_OPTIONS["min_new_songs"],
    min_genre_songs: int = COMMON_OPTIONS["min_genre_songs"],
    min_stage1_songs: int = COMMON_OPTIONS["min_stage1_songs"],
    base_model: str = COMMON_OPTIONS["base_model"],
    output_dir: str = COMMON_OPTIONS["output_dir"],
    batch_size: int = COMMON_OPTIONS["batch_size"],
    grad_accum_steps: int = COMMON_OPTIONS["grad_accum_steps"],
    epochs: int = COMMON_OPTIONS["epochs"],
    lr: float = COMMON_OPTIONS["lr"],
    lora_rank: int = COMMON_OPTIONS["lora_rank"],
    use_4bit: bool = COMMON_OPTIONS["use_4bit"],
    style_vec_dir: Optional[str] = COMMON_OPTIONS["style_vec_dir"],
):
    while True:
        report = run_cycle(
            source=source,
            extra_sources=extra_sources,
            raw_dir=raw_dir,
            default_genre=default_genre,
            state_path=state_path,
            manifest_path=manifest_path,
            train=train,
            stage1=stage1,
            genres=genres,
            min_new_songs=min_new_songs,
            min_genre_songs=min_genre_songs,
            min_stage1_songs=min_stage1_songs,
            base_model=base_model,
            output_dir=output_dir,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            epochs=epochs,
            lr=lr,
            lora_rank=lora_rank,
            use_4bit=use_4bit,
            style_vec_dir=style_vec_dir,
        )
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    app()
