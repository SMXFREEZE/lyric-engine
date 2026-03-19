# Engineer Handoff

## Access

- GitHub repo: `https://github.com/SMXFREEZE/lyric-engine`
- Active branch: `codex/emotional-specificity-module`
- Canonical base model: `mistralai/Mistral-7B-Instruct-v0.2`
- Kaggle notebook file: `notebooks/train_kaggle.ipynb`
- Colab notebook file: `notebooks/train_colab.ipynb`

Recommended first commands:

```bash
git clone https://github.com/SMXFREEZE/lyric-engine
cd lyric-engine
git checkout codex/emotional-specificity-module
git pull origin codex/emotional-specificity-module
git log --oneline -10
```

Important recent commits:

- `97a36ad` Use single-GPU inference path in Kaggle notebook
- `3ab4e34` Prefer wrapped model device for generation inputs
- `3728c8a` Normalize numeric shard device ids for inference
- `9dc79c2` Select inference input device from wrapped model
- `128e569` Respect sharded device maps during inference
- `4109504` Harden inference fallback for empty candidate generation
- `c494452` Load Kaggle adapter weights via CPU-first offload
- `4e9138f` Use safe base-model loading in Kaggle generation cell
- `d1ab794` Fix phonetic loss alignment with style prefix
- `eac4c0f` Fix Kaggle T4 OOM during model startup
- `b73b35d` Skip fragile Kaggle audio path in run-all notebook
- `9316a26` Add continuous dataset sync and retraining loop
- `e6e5481` Add multi-provider lyrics retrieval pipeline

## Project Intent

Lyric Engine is a songwriter-oriented generation system, not just plain next-token text generation.
The current branch tries to combine:

- a base causal LM
- LoRA fine-tuning for genre adaptation
- style-vector conditioning at embedding level
- phonetic auxiliary supervision
- research-inspired scoring
- a metacognitive workspace that ranks and explains line choices
- data collection, chart analysis, and continuous dataset sync tooling

It is best understood as four connected subsystems:

1. data ingestion and annotation
2. training and checkpointing
3. inference and metacognitive ranking
4. notebook/orchestration layer for Kaggle and Colab

## Current Working Status

At handoff time, the Kaggle training path is working end to end with manual or refreshed notebook state:

- GPU setup works on Kaggle `T4 x2`
- dataset auto-discovery works from `/kaggle/input`
- style vectors build successfully
- viral conditioning runs successfully
- Stage 2 LoRA training completed successfully on Kaggle
- checkpoint loading for inference works
- generation works after using the latest single-GPU inference cell

User-verified run outcome:

- uploaded dataset rows: `1022`
- `trap` subset rows: `196`
- `trap` artists in current run: `2`
- style vectors built for: `Young Thug`, `Offset`
- final checkpoint path in Kaggle:
  `/kaggle/working/checkpoints/genre_trap/epoch_3`

Example successful generation output:

- `I really don't know a damn thang (I really don't)`
- `I am a boss in the making`
- `When we say "Miami" you know what I mean?`

## Top-Level Architecture

### 1. Data Layer

Main files:

- `src/data/lyrics_providers.py`
- `src/data/scraper.py`
- `scripts/infinite_crawl.py`
- `scripts/continuous_learning.py`
- `src/data/style_extractor.py`
- `src/data/chart_scraper.py`
- `src/data/viral_analyzer.py`
- `src/data/phoneme_annotator.py`
- `src/data/rhyme_labeler.py`
- `src/data/valence_scorer.py`

Responsibilities:

- fetch lyrics from multiple sources
- discover artists and songs
- clean and normalize lyric text
- annotate phonetic and emotional features
- build artist style vectors
- scrape chart data and compute viral conditioning
- sync growing datasets into normalized JSONL views

### 2. Model Layer

Main files:

- `src/model/lyrics_model.py`
- `src/model/dual_tokenizer.py`
- `src/model/phonetic_head.py`
- `src/model/checkpoint_loader.py`
- `src/model/metacognitive_engine.py`
- `src/model/composer_cortex.py`
- `src/model/emotional_geometry.py`
- `src/model/phonosemantic.py`
- `src/model/dopamine_arc.py`
- `src/model/research_scoring.py`

Responsibilities:

- load the base LLM and tokenizer
- inject special tokens
- attach LoRA adapters
- add style projection
- add phonetic auxiliary head
- score and rank candidate lines
- maintain self-model and workspace reasoning state

### 3. Training Layer

Main files:

- `src/training/dataset.py`
- `src/training/sft.py`
- `src/training/rlhf.py`

Responsibilities:

- turn JSONL records into trainable prompt format
- attach style vectors
- derive phoneme token targets
- run Stage 1 and Stage 2 supervised fine-tuning
- save LoRA checkpoints

### 4. Inference/API Layer

Main files:

- `src/inference/engine.py`
- `src/api/server.py`

Responsibilities:

- candidate generation
- candidate scoring and ranking
- co-write session management
- API serving

### 5. Orchestration Layer

Main files:

- `notebooks/train_kaggle.ipynb`
- `notebooks/train_colab.ipynb`
- `scripts/smoke_test.py`

Responsibilities:

- environment setup
- repo sync
- dependency installation
- secrets loading
- dataset discovery
- training orchestration
- smoke validation

## Data Pipeline Details

### Lyrics Providers

`src/data/lyrics_providers.py` implements the multi-source retrieval layer.

Sources currently referenced in the branch:

- Genius
- Vagalume
- LRCLIB
- lyrics.ovh

Observed behavior:

- Genius is unstable in Kaggle and sometimes Colab due to Cloudflare / anti-bot blocking
- Vagalume is useful when reachable
- lyrics.ovh gives a simple fallback
- candidate ranking selects the best lyrics among available provider responses

### Crawling

`scripts/infinite_crawl.py` is the live crawler.

Key behavior:

- artist discovery via Deezer/Genius-compatible paths
- queue expansion through similar artists
- lyrics fetched through the provider layer
- progress persisted in:
  - `data/crawl/seen_artists.txt`
  - `data/crawl/seen_songs.txt`
  - `data/crawl/songs.jsonl`

Important bug fixes already made:

- do not mark songs as seen before a real lyric save
- count lyric misses separately
- allow direct script execution without broken `src` imports

### Continuous Learning

`scripts/continuous_learning.py` exists and works as a sync/retrain utility.

It can:

- ingest one or more JSONL sources
- normalize records
- deduplicate
- rebuild `data/raw/all_songs.jsonl`
- rebuild per-genre files such as `data/raw/trap.jsonl`
- keep state in `data/automation/continuous_learning_state.json`
- optionally retrain once thresholds are reached

It is not yet a true always-on production scheduler.
It is a reusable script, not a deployed orchestration service.

## Training Pipeline Details

### Dataset Formatting

`src/training/dataset.py`

Important functions:

- `split_into_sections()`
- `format_training_example()`
- `LyricsDataset`
- `create_dataloaders()`

Format shape:

- `[GENRE_START] genre [GENRE_END]`
- optional `[STYLE_START] [STYLE_END]`
- section token such as `[VERSE]`
- arc token such as `[SETUP]`
- lyric lines

Style injection note:

- textual placeholders exist in formatting
- actual style conditioning occurs at embedding level through the model path

### SFT

`src/training/sft.py`

There are two conceptual stages:

- Stage 1: general corpus training
- Stage 2: genre-specific adapter training

Current Kaggle workflow usually skips Stage 1 because:

- uploaded dataset sizes are small
- current user run used one narrow genre subset

Stage 2 worked successfully on Kaggle after recent fixes.

### Model Loader

`src/model/lyrics_model.py`

Key details:

- default dev model is `gpt2`
- runtime model path comes from `LYRICS_MODEL_PATH`
- special tokens are injected dynamically
- `resize_token_embeddings(..., mean_resizing=False)` was added to avoid an OOM spike on small GPUs
- loader now supports `device_map_override` to force single-GPU inference in Kaggle notebook

Important note:

- Canonical base model is now `mistralai/Mistral-7B-Instruct-v0.2` across all files
- `src/training/sft.py` default matches
- Kaggle notebook uses the same model
- README documents the correct model
- `gpt2` remains the dev/test fallback when GPU is unavailable

## Inference and Metacognition

### Inference Engine

`src/inference/engine.py`

Core concepts:

- `SongMemory`
- `CandidateScore`
- constrained multi-pass candidate generation
- ranking via metacognitive workspace
- support for `generate_line()` and `generate_verse()`
- co-write session support

Recent inference fixes focused on Kaggle:

- avoid empty first-line extraction
- add fallback pass when candidate generation is empty
- improve device resolution under `PeftModel` + `device_map`
- normalize numeric device ids such as `0` into `cuda:0`
- add single-GPU inference path in notebook to avoid cross-device issues

### Metacognitive Layer

`src/model/metacognitive_engine.py`

Important structures:

- `CandidateEvidence`
- `WorkspaceContext`
- `ModuleObservation`
- `WorkingMemoryState`
- `MetacognitiveState`
- `WorkspaceCandidate`
- `SelfModel`
- `WorkspaceDecision`

Purpose:

- turn scorer outputs into an explicit choice
- keep confidence/conflict state
- decide when regeneration is needed
- generate `hot_trace` explanations

### Composer Cortex

`src/model/composer_cortex.py`

Purpose:

- maintain domain memory about motifs, confessional bias, density, hook pressure, section word targets, and narrative pressure
- score candidates against short-term lyrical context

## Notebook and Runtime Details

### Kaggle Notebook

Primary file:

- `notebooks/train_kaggle.ipynb`

Current design:

- checks GPU and per-device VRAM
- clones or updates repo branch
- installs core dependencies only
- skips optional audio by default on Kaggle
- loads Kaggle secrets when available
- runs `scripts/smoke_test.py`
- discovers uploaded JSONL datasets under `/kaggle/input`
- normalizes into:
  - `data/raw/all_songs.jsonl`
  - `data/raw/<genre>.jsonl`
- builds style vectors
- builds viral conditioning
- verifies pipeline on sample data
- trains Stage 2 LoRA
- runs generation / metacognitive output cell

Very important notebook caveat:

Kaggle imported notebooks are copied into the notebook UI.
Pulling repo updates with `git pull` updates files under `/kaggle/working/lyric-engine`, but it does not rewrite the already-open notebook cells in the UI.

This caused several rounds of confusion.

Implication:

- repo code can be newer than notebook cell text
- module reloads were needed to pick up fixes
- for critical notebook changes, re-importing the notebook is often cleaner than only pulling the repo

### Colab Workflow

Colab was the better environment for collection.

Observed practical split:

- Colab for data collection
- Kaggle for training

Reason:

- Genius scraping was blocked more often on Kaggle
- Colab had better odds of getting lyrics from the live web stack

## API State

`src/api/server.py` has been modernised to use the unified checkpoint loader.

Capabilities:

- loads base model or PEFT checkpoints via `CHECKPOINT_PATH` env var
- uses `src/model/checkpoint_loader.py` for all model loading
- supports `SINGLE_GPU=1` for Kaggle-compatible forced single-device inference
- supports style blending via request body
- returns workspace and metacognition data in suggestion responses
- exposes `/styles` (full Style DNA library), `/analyze` (cognitive analysis), and `/cowrite/analysis/{id}`

The API is aligned with the current training/inference path.

## Audio State

Audio-related modules exist:

- `src/audio/instrumental_generator.py`
- `src/audio/song_assembler.py`
- `src/audio/vocal_generator.py`

But the Kaggle notebook intentionally disables audio because:

- `audiocraft` is unreliable under Kaggle Python 3.12
- optional audio install failures were causing noisy notebook failures

Current status:

- lyric system is primary and working
- audio remains optional and should be tested separately in Colab or local environment

## Known Issues and Uncertainties

### 1. README and Code Drift

The README does not fully match the live branch behavior.

Examples:

- README architecture text still mentions Llama 3.1 8B
- Kaggle notebook uses Mistral 7B Instruct
- current implementation is more metacognitive and more notebook-driven than the README suggests

### 2. API Drift

`src/api/server.py` is behind the notebook path.

Unknowns:

- whether it can cleanly serve latest Kaggle checkpoints
- whether it correctly supports PEFT + style vectors + current inference fixes

### 3. Inference Still Needs Cleanup

Even though generation now works, inference plumbing has been fragile.

Recent issues included:

- sharded-device mismatch
- CPU/GPU adapter loading mismatch
- stale Kaggle modules after repo pulls
- placeholder candidates from empty generation

This area is improved but still not elegant.

### 4. Dataset Breadth Is Still Weak

The current successful user run trained `trap` on:

- 196 songs
- 2 artists

This is enough to validate the pipeline, but not enough for a broad genre model.

Current quality uncertainty:

- generation may overfit to a narrow slice
- candidate quality is constrained by weak genre diversity

### 5. Chart Coverage Is Partial

`src/data/chart_scraper.py` is useful but incomplete.

Observed problems:

- outdated Billboard slugs
- many country charts return `no data`
- Morocco, MENA, and several African regions have weak fallback coverage
- Last.fm country fallback map is incomplete

This affects viral-conditioning richness, not core training correctness.

### 6. Kaggle Notebook Freshness Trap

This is a recurring operational issue:

- notebook cell code in UI gets stale
- repo code under `/kaggle/working/lyric-engine` can be newer
- users may rerun stale notebook cells even after `git pull`

### 7. Base-Model Assumptions — RESOLVED

All files now use `mistralai/Mistral-7B-Instruct-v0.2` as the canonical production model:

- `src/model/lyrics_model.py` PROD_MODEL
- `src/training/sft.py` default and CLI
- README.md
- GPT-2 remains as the dev/test fallback

### 8. Continuous Learning Is Not Yet Automated End to End

`scripts/continuous_learning.py` exists, but there is no deployed scheduler or workflow manager around it.

It is a tool, not yet a production service.

### 9. Provider Reliability Is Environment-Dependent

Genius, Vagalume, and other providers behave differently across:

- Kaggle
- Colab
- local machine

The data collection path is still partly operational rather than fully deterministic.

## Highest-Priority Fixes for the Engineer

### Priority 1: Unified Inference Loading — DONE

Created `src/model/checkpoint_loader.py`:

- `load_for_inference()` — PEFT checkpoint + base model + device resolution
- `load_for_training()` — base model for SFT
- both notebook and API use the same loader

### Priority 2: Base Model Defaults — DONE

All files standardised on `mistralai/Mistral-7B-Instruct-v0.2`.

### Priority 3: Modernise `server.py` — DONE

- uses unified checkpoint loader
- loads real PEFT checkpoints via `CHECKPOINT_PATH`
- returns workspace/metacognition in responses
- new endpoints: `/styles`, `/analyze`, `/cowrite/analysis/{id}`

### Priority 4: Improve Dataset Breadth

Goal:

- make model quality less narrow and less artist-specific

Needed work:

- collect broader trap coverage
- gather more artists per genre
- possibly support mixed-genre or all-songs training mode more explicitly

### Priority 5: Clean Up Kaggle Notebook Operationally

Goal:

- eliminate stale-cell confusion

Needed work:

- minimize cell-local overrides
- centralize logic into repo modules
- reduce the amount of manual cell replacement needed after fixes
- add a clear notebook note explaining that `git pull` does not update already-imported cell text

### Priority 6: Harden Chart Coverage

Goal:

- improve MENA and Africa coverage

Needed work:

- expand Last.fm country mapping
- add region-aware source fallbacks
- fix suspicious Deezer country ids
- update outdated Billboard chart slugs
- log why a country failed, not just `no data`

### Priority 7: Add Better Regression Tests

Current smoke test is helpful but not enough.

Add:

- notebook-path inference regression
- real PEFT checkpoint load regression
- style-prefix plus phonetic-loss regression already partly covered
- API generation regression
- chart scraper fallback tests

## Suggested Immediate Engineering Plan

1. Create a dedicated inference loader module used by notebook and API.
2. Clean up `server.py` to load current checkpoints correctly.
3. Normalize model defaults across README, notebook, and training code.
4. Add inference regression tests for:
   - single GPU
   - sharded device map
   - PEFT adapter load
5. Improve dataset breadth before drawing quality conclusions from generated text.
6. Improve chart scraper coverage and diagnostics.
7. Decide whether continuous learning should become:
   - a local cron workflow
   - a GitHub Action
   - a server-side scheduled job

## Quick File Map for the Engineer

Best entry points:

- `README.md`
- `notebooks/train_kaggle.ipynb`
- `src/model/lyrics_model.py`
- `src/inference/engine.py`
- `src/model/metacognitive_engine.py`
- `src/model/composer_cortex.py`
- `src/training/sft.py`
- `src/training/dataset.py`
- `src/data/lyrics_providers.py`
- `scripts/continuous_learning.py`
- `scripts/infinite_crawl.py`
- `scripts/smoke_test.py`

## Final Notes

- The project is real and functioning, but still in an active integration phase.
- Training works.
- Generation works after the latest inference fixes and single-GPU Kaggle path.
- The biggest risk is not one broken model component. It is inconsistency between modules, notebook copies, and older loading assumptions.
- The engineer should treat this as a system that already has valuable pieces, but needs consolidation and cleanup more than another wave of new features.
