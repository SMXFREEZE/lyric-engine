# Lyric Engine

Lyric Engine is a lyrics generation system built for songwriting, not just text completion.
It pays attention to sound, rhythm, rhyme, and emotional movement while it writes, so the output is shaped more like music and less like generic prose.

## What makes it different

Most lyric generators write a line first and check the rhyme later.
Lyric Engine scores candidates during generation, so rhyme, syllable count, and emotional fit are part of the decision process from the start.

It combines a **metacognitive workspace** (System 1 / System 2 mode switching, self-model, confidence/conflict tracking) with **research-backed scoring** (polysyllabic rhyme depth, internal rhyme density, motor-rhythm coupling) and **Style DNA conditioning** (genre-specific BPM, flow, vocabulary, and repetition profiles).

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
|  | StyleVector |--→| Base LLM (Mistral 7B Instruct)    |  |
|  | Projector   |    | + Genre LoRA Adapter             |  |
|  | (128d→768d) |    | + General Music Adapter          |  |
|  +-------------+    +------------------+---------------+  |
|                                       | hidden states     |
|                            +----------v-----------+       |
|                            |    Phonetic Head     |       |
|                            |    (2-layer MLP)     |       |
|                            |    → phoneme logits  |       |
|                            +----------------------+       |
+-----------------------------------------------------------+
                              |
              +---------------v-----------------+
              |   Metacognitive Workspace       |
              |   8 specialist modules:         |
              |   phonology · rhythm · semantics|
              |   emotion · structure · novelty |
              |   auditory · template           |
              |   System 1/2 mode switching     |
              |   self-model adaptation         |
              |   hot_trace self-explanation     |
              +---------------+-----------------+
                              |
              +---------------v-----------------+
              |   4-Layer Candidate Scoring      |
              |   L1: rhyme + syllable + novelty |
              |   L2: emotional geometry +       |
              |       phonosemantic + dopamine   |
              |   L3: research-backed (7 signals)|
              |   L4: Style DNA coherence        |
              +---------------------------------+
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

**Style DNA library**

29 styles with full audio + lyrical fingerprints: BPM range, energy, valence, danceability, speechiness, syllable density, rhyme schemes, flow patterns, language weights, and blend compatibility. Styles can be blended at inference time for cross-genre generation.

**Emotional arc modeling**

Songs are tagged with section-level arc tokens such as `[SETUP]`, `[BUILD]`, `[RELEASE]`, `[REFRAME]`, and `[PEAK]`.
Valence and arousal are scored line by line, and the arc is treated as a real constraint, not just a prompt hint.

**Constrained beam search**

The generator scores candidates on:

1. phonetic rhyme match
2. syllable count fit
3. novelty against accepted lines
4. emotional fit to the target arc
5. 8D emotional trajectory geometry
6. phonosemantic texture alignment
7. goosebump / dopamine prediction
8. hook DNA strength
9. polysyllabic rhyme depth (arXiv:2505.00035)
10. internal rhyme density
11. complexity calibration
12. temporal arc weighting
13. introspection/confessional signal
14. vocabulary novelty
15. stress alignment (motor-rhythm coupling)
16. Style DNA coherence

It emits the top line in auto mode and multiple strong options in co-write mode.

## Project structure

```text
src/
|-- data/
|   |-- scraper.py           # Lyrics collection
|   |-- lyrics_providers.py  # Multi-source lyrics retrieval
|   |-- phoneme_annotator.py # Stress and syllable tagging
|   |-- rhyme_labeler.py     # Rhyme scheme detection
|   |-- valence_scorer.py    # Valence, arousal, and arc tagging
|   |-- style_extractor.py   # Artist style vector extraction
|   |-- style_dna.py         # 29-style genre fingerprint library
|   |-- chart_scraper.py     # Billboard, Deezer, Spotify, Last.fm charts
|   `-- viral_analyzer.py    # Viral scoring and conditioning
|-- model/
|   |-- dual_tokenizer.py    # BPE + ARPAbet token streams
|   |-- phonetic_head.py     # Auxiliary head for constrained decoding
|   |-- lyrics_model.py      # LLM + LoRA + style projector + phonetic head
|   |-- metacognitive_engine.py  # GWT workspace + System 1/2 + self-model
|   |-- composer_cortex.py   # Domain memory + auditory gate + template match
|   |-- checkpoint_loader.py # Unified model loading for all environments
|   |-- emotional_geometry.py    # 8D emotional trajectory
|   |-- phonosemantic.py     # Sound-meaning alignment
|   |-- dopamine_arc.py      # Goosebump prediction + tension curves
|   `-- research_scoring.py  # 7 peer-reviewed scoring signals
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

| Genre | Style | Region |
|---|---|---|
| Trap | 808 cadence, triplet flow, flex imagery | Americas |
| Drill | UK/Chicago variants, street-specific slang | Americas/Europe |
| Hip-Hop | Wordplay, storytelling, boom-bap cadence | Americas |
| Boom Bap | Golden-era flow, multisyllabic rhymes | Americas |
| UK Rap | Fast-paced delivery, London slang | Europe |
| Grime | 140 BPM rapid staccato delivery | Europe |
| Latin Trap | Spanglish, triplet and reggaeton flows | Latin America |
| French Rap | French, multisyllabic rhyme depth | Europe |
| R&B | Melisma-friendly, vulnerability, gospel callbacks | Americas |
| Neo Soul | Free-flowing, jazz harmonics, conscious lyrics | Americas |
| Pop | Hook-first writing, universal themes, sticky phrasing | Global |
| Indie Pop | Concrete imagery, conversational melodic delivery | Europe/Americas |
| Afrobeats | Syncopated groove, Pidgin/Yoruba code-switching | Africa |
| Afroswing | UK-African fusion, melodic rap | Africa/Europe |
| Amapiano | Log drum bass, chanted vocals, Zulu/Sesotho | Africa |
| Reggaeton | Dembow riddim, Spanish, club energy | Latin America |
| Bachata | Romantic mellowness, acoustic guitar, heartbreak | Latin America |
| Corrido | Corridos tumbados, narrative, norteño-trap fusion | Latin America |
| K-Pop | Korean/English blend, precise pop structure | Asia |
| J-Pop | Japanese lyrics, anime-influenced melodics | Asia |
| Arabic Pop | Ornamental melodies, Middle Eastern scales | Middle East |
| Mahraganat | Egyptian street rap, rapid Arabic delivery | Middle East |
| Chaabi | Moroccan folk-pop, Arabic/Berber/French mix | Africa |
| Raï | Raï modernised, Arabic/French, improvisation | Africa |
| Dancehall | Jamaican patois, syncopated riddims | Americas |
| Indie | Concrete imagery, non-obvious metaphors | Europe/Americas |
| Alt/Emo | Stream of consciousness, distorted metaphors | Americas/Europe |
| EDM | Drop structure, minimal lyrics, euphoric energy | Global |
| Country | Narrative storytelling, rural imagery | Americas |

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

# Run smoke tests (15 tests, no GPU needed)
python scripts/smoke_test.py

# Start API server
python src/api/server.py
```

API runs at `http://localhost:8000`.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/genres` | List available genres |
| GET | `/styles` | Full Style DNA library |
| POST | `/generate` | Generate a song section |
| POST | `/analyze` | Analyse existing lyrics |
| POST | `/cowrite/start` | Start a co-write session |
| POST | `/cowrite/suggest` | Get line suggestions |
| POST | `/cowrite/accept` | Accept a line |
| GET | `/cowrite/analysis/{id}` | Cognitive analysis of a session |

## Generate example

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"genre": "trap", "section": "VERSE", "arc_token": "[BUILD]", "num_lines": 8}'
```

Response includes lines, phoneme annotations, detected rhyme scheme, workspace decisions, and full cognitive analysis.

## Style blending

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"genre": "trap", "style_blend": {"trap": 0.6, "rnb": 0.4}, "num_lines": 4}'
```

## Co-write mode

```bash
# Start a session
curl -X POST http://localhost:8000/cowrite/start \
  -d '{"genre": "rnb", "rhyme_scheme": "ABAB"}'

# Get suggestions (includes metacognition data)
curl -X POST http://localhost:8000/cowrite/suggest \
  -d '{"session_id": "abc-123", "n": 3}'

# Accept a line
curl -X POST http://localhost:8000/cowrite/accept \
  -d '{"session_id": "abc-123", "line": "I been moving in silence"}'

# Get cognitive analysis of the session
curl http://localhost:8000/cowrite/analysis/abc-123
```

## Environment variables

```bash
LYRICS_MODEL_PATH=gpt2               # Base model (default: gpt2 for dev)
CHECKPOINT_PATH=                      # Path to trained PEFT/LoRA checkpoint
GENIUS_TOKEN=                         # Genius API token for collection
VALENCE_MODEL_PATH=                   # Optional emotion model path
BEAM_SIZE=8                           # Beam width
HUGGING_FACE_HUB_TOKEN=              # Required to download gated base models
SINGLE_GPU=0                          # Force single-GPU inference (1 for Kaggle)
USE_4BIT=1                            # Enable 4-bit quantisation
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

Estimated cost: about `$0.002` per generated song.

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

**Why Mistral 7B Instruct?**

Best balance of quality, speed, and Kaggle/Colab compatibility.
Successfully trained and validated on Kaggle T4 x2.
