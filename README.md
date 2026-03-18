# Lyric Engine

Lyric Engine is a lyrics generation system built for songwriting, not just text completion.
It pays attention to sound, rhythm, rhyme, and emotional movement while it writes, so the output is shaped more like music and less like generic prose.

## What makes it different

Most lyric generators write a line first and check the rhyme later.
Lyric Engine scores candidates during generation, so rhyme, syllable count, and emotional fit are part of the decision process from the start.

```text
Input:  genre=trap, rhyme_scheme=AABB, section=VERSE, arc=[BUILD]
Output: 8 lines where end rhymes are phonetically checked,
        syllable counts stay close to target, and valence
        follows the requested emotional arc
```

## Architecture

```text
+-----------------------------------------------------------+
|                        LyricsModel                        |
|                                                           |
|  +-------------+    +----------------------------------+  |
|  | StyleVector |--->| Base LLM (Llama 3.1 8B)          |  |
|  | Projector   |    | + Genre LoRA Adapter             |  |
|  | (128d->768d)|    | + General Music Adapter          |  |
|  +-------------+    +------------------+---------------+  |
|                                       | hidden states     |
|                            +----------v-----------+       |
|                            |    Phonetic Head     |       |
|                            |    (2-layer MLP)     |       |
|                            |    -> phoneme logits |       |
|                            +----------------------+       |
+-----------------------------------------------------------+
```

**Dual tokenizer**

- Semantic stream: standard BPE tokens for meaning
- Phoneme stream: ARPAbet tokens from the CMU Pronouncing Dictionary for sound

**Genre LoRA adapters**

Each genre gets a lightweight adapter, roughly 4M parameters, that can be blended at inference time.
That makes it possible to mix styles like `60% trap + 40% R&B` in a single forward pass.

**Artist style encoder**

A 128-dimensional vector captures an artist's statistical fingerprint, including syllable habits, rhyme density, vocabulary spread, metaphor clustering, and emotional profile.
It is injected as a prefix in embedding space instead of copying lyrics directly.

**Emotional arc modeling**

Songs are tagged with section-level arc tokens such as `[SETUP]`, `[BUILD]`, `[RELEASE]`, `[REFRAME]`, and `[PEAK]`.
Valence and arousal are scored line by line, and the arc is treated as a real constraint, not just a prompt hint.

**Constrained beam search**

The generator scores candidates on:

1. phonetic rhyme match
2. syllable count fit
3. novelty against accepted lines
4. emotional fit to the target arc

It emits the top line in auto mode and multiple strong options in co-write mode.

## Project structure

```text
src/
|-- data/
|   |-- scraper.py           # Lyrics collection
|   |-- phoneme_annotator.py # Stress and syllable tagging
|   |-- rhyme_labeler.py     # Rhyme scheme detection
|   |-- valence_scorer.py    # Valence, arousal, and arc tagging
|   `-- style_extractor.py   # Artist style vector extraction
|-- model/
|   |-- dual_tokenizer.py    # BPE + ARPAbet token streams
|   |-- phonetic_head.py     # Auxiliary head for constrained decoding
|   `-- lyrics_model.py      # LLM + LoRA + style projector + phonetic head
|-- training/
|   |-- dataset.py           # Training example formatting and DataLoader
|   |-- sft.py               # General + genre-specific supervised fine-tuning
|   `-- rlhf.py              # Reward model + PPO via TRL
|-- inference/
|   `-- engine.py            # Constrained beam search + co-write session logic
`-- api/
    `-- server.py            # FastAPI endpoints
```

## Supported genres

| Genre | Style |
|---|---|
| Trap | 808 cadence, triplet flow, flex imagery |
| R&B | Melisma-friendly, vulnerable writing, gospel callbacks |
| Indie | Concrete imagery, emotional restraint, less obvious metaphors |
| Pop | Hook-first writing, universal themes, sticky phrasing |
| Drill | UK and Chicago variants, street-specific slang |
| Alt/Emo | Stream of consciousness, distorted metaphors |
| Hip-Hop | Wordplay, storytelling, boom-bap cadence |
| Country | Narrative focus, rural imagery, heartbreak and pride |
| Rock | Anthemic hooks, rebellion |
| Latin | Reggaeton flow, Spanglish, rhythmic wordplay |

## Training pipeline

| Stage | What | Compute |
|---|---|---|
| 1 | General music SFT on a large annotated corpus | 4x A100, about $800 |
| 2 | Per-genre LoRA adapters | 1x A100 per genre, about $150 total |
| 3 | RLHF/PPO with human preference ratings | 1x A100, about $200 |
| 4 | Phonetic head training with frozen base | 1x A100, about $50 |

Free alternative:
Kaggle plus Google Colab.

## Local development

```bash
git clone https://github.com/SMXFREEZE/lyric-engine
cd lyric-engine
pip install -r requirements.txt

# Run smoke tests
python scripts/smoke_test.py

# Start API server
python src/api/server.py
```

API runs at `http://localhost:8000`.

Example request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"genre": "trap", "section": "VERSE", "arc_token": "[BUILD]", "num_lines": 8}'
```

## Co-write mode

```bash
# Start a session
curl -X POST http://localhost:8000/cowrite/start \
  -d '{"genre": "rnb", "rhyme_scheme": "ABAB"}'

# Get suggestions for the next line
curl -X POST http://localhost:8000/cowrite/suggest \
  -d '{"session_id": "abc-123", "n": 3}'

# Accept a line and update context
curl -X POST http://localhost:8000/cowrite/accept \
  -d '{"session_id": "abc-123", "line": "I been moving in silence"}'
```

## Production serving

```text
vLLM on Modal Labs serverless
|-- scale to zero when idle
|-- auto-scale under load
|-- 4-bit quantization with bitsandbytes NF4
|-- speculative decoding with a 1B draft model
`-- target: full verse in under 3 seconds on 1x A10G
```

Estimated cost:
about `$0.002` per generated song.

## Environment variables

```bash
GENIUS_TOKEN=            # Genius API token for collection
LYRICS_MODEL_PATH=gpt2   # Path to trained model for local/dev use
VALENCE_MODEL_PATH=      # Optional emotion model path
BEAM_SIZE=8              # Beam width
HUGGING_FACE_HUB_TOKEN=  # Required to download gated base models
```

## Key technical decisions

**Why not just prompt GPT-4?**

Prompting alone cannot enforce phonetic constraints at the token level.
You can ask a general model to rhyme, but it may still drift.
Lyric Engine checks constraints before a candidate is accepted.

**Why LoRA per genre instead of one giant model?**

It is cheaper, easier to extend, and much more flexible.
You can add a genre without retraining the entire base model, and you can blend adapters at runtime.

**Why style vectors instead of fine-tuning on artist lyrics directly?**

Copyright.
The style vector aims to capture how someone writes, not reproduce what they wrote.
