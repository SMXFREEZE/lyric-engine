"""
FastAPI server for the lyrics LLM.

Endpoints:
  POST /generate        — auto-generate a full song section
  POST /cowrite/start   — start a co-write session
  POST /cowrite/suggest  — suggest N lines in co-write mode
  POST /cowrite/accept   — accept a line, update session context
  GET  /cowrite/song/{id} — retrieve current lyrics for a session
  GET  /cowrite/analysis/{id} — full cognitive analysis of a session
  DELETE /cowrite/session/{id} — end a session
  GET  /health          — health check
  GET  /genres          — list available genres
  GET  /styles          — list all Style DNA entries
  POST /analyze         — analyse lyrics with cognitive engine

Session state is held in memory (per-request for stateless, or via session_id
for co-write).  For production: swap session dict for Redis.
"""

import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from configs.genres import GENRES, GENRE_DESCRIPTIONS


# ── Lazy model load (loads once on first request) ─────────────────────────────

_engine = None
_sessions: dict[str, object] = {}   # session_id → CoWriteSession


def get_engine():
    """Load the inference engine using the unified checkpoint loader.

    Supports PEFT / LoRA checkpoints via ``CHECKPOINT_PATH`` env var and
    the canonical inference path in ``checkpoint_loader.py``.
    """
    global _engine
    if _engine is None:
        from src.model.checkpoint_loader import load_for_inference
        from src.inference.engine import LyricsEngine

        checkpoint_path = os.getenv("CHECKPOINT_PATH")
        base_model = os.getenv("LYRICS_MODEL_PATH") or None
        use_4bit = os.getenv("USE_4BIT", "1") != "0"
        beam_size = int(os.getenv("BEAM_SIZE", "8"))

        # Allow forcing single-GPU via env var (mirrors Kaggle fix)
        device_map = None
        if os.getenv("SINGLE_GPU", "0") == "1":
            device_map = {"": 0}

        model, tokenizer, device = load_for_inference(
            checkpoint_path=checkpoint_path,
            base_model=base_model,
            use_4bit=use_4bit,
            device_map_override=device_map,
        )

        _engine = LyricsEngine(model, tokenizer, device=device, beam_size=beam_size)
        print(f"[server] Engine ready on {device}")
    return _engine


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Lyric Engine",
    description="Phonetic-aware, genre-native, metacognitive lyrics generation",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    genre: str = Field("hip_hop", description="Genre name")
    section: str = Field("VERSE", description="Song section: VERSE, CHORUS, BRIDGE, etc.")
    arc_token: str = Field("[SETUP]", description="Emotional arc token")
    num_lines: int = Field(8, ge=1, le=32)
    rhyme_scheme: str = Field("AABB", description="AABB, ABAB, ABCB, or free")
    target_syllables: int = Field(10, ge=4, le=20)
    artist_style: Optional[str] = Field(None, description="Artist name for style conditioning")
    seed_lines: list[str] = Field(default_factory=list, description="Fixed anchor lines")
    mood: str = Field("dark", description="Mood: dark / hype / romantic / chill / sad / epic")
    style_blend: Optional[dict[str, float]] = Field(None, description="Style blend weights, e.g. {\"trap\": 0.6, \"rnb\": 0.4}")

class GenerateResponse(BaseModel):
    lines: list[str]
    phoneme_annotations: list[dict]
    rhyme_scheme_detected: str
    workspace: Optional[dict] = None
    analysis: Optional[dict] = None

class CoWriteStartRequest(BaseModel):
    genre: str = "hip_hop"
    rhyme_scheme: str = "AABB"
    target_syllables: int = 10
    mood: str = "dark"

class CoWriteStartResponse(BaseModel):
    session_id: str

class SuggestRequest(BaseModel):
    session_id: str
    n: int = Field(3, ge=1, le=5)

class SuggestResponse(BaseModel):
    suggestions: list[dict]

class AcceptRequest(BaseModel):
    session_id: str
    line: str

class AcceptResponse(BaseModel):
    accepted: str
    total_lines: int

class AnalyzeRequest(BaseModel):
    lines: list[str]
    genre: str = "hip_hop"
    mood: str = "dark"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": os.getenv("LYRICS_MODEL_PATH", "gpt2"),
        "checkpoint": os.getenv("CHECKPOINT_PATH", "none"),
    }


@app.get("/genres")
def list_genres():
    return [
        {"name": g, "description": GENRE_DESCRIPTIONS.get(g, "")}
        for g in GENRES
    ]


@app.get("/styles")
def list_styles():
    """Return the full Style DNA library."""
    from src.data.style_dna import STYLES
    from dataclasses import asdict
    return {
        name: {
            **asdict(dna),
            "bpm_range": list(dna.bpm_range),
        }
        for name, dna in STYLES.items()
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    from src.inference.engine import SongMemory
    from src.data.phoneme_annotator import annotate_line, annotation_to_dict
    from src.data.rhyme_labeler import detect_scheme
    import numpy as np

    engine = get_engine()

    # Load style vector if requested
    style_vec = None
    if req.artist_style:
        from src.data.style_extractor import load_style_vector
        style_vec = load_style_vector(req.artist_style)

    # Load Style DNA if available
    style_dna = None
    try:
        from src.data.style_dna import STYLES, blend_styles
        if req.style_blend:
            style_dna = blend_styles(req.style_blend)
        elif req.genre in STYLES:
            style_dna = STYLES[req.genre]
    except Exception:
        pass

    memory = SongMemory(
        genre=req.genre,
        mood=req.mood,
        style_vec=style_vec,
        rhyme_scheme=req.rhyme_scheme,
        target_syllables=req.target_syllables,
        style_dna=style_dna,
    )

    # Seed with fixed anchor lines
    for line in req.seed_lines:
        memory.add_line(line)

    lines = engine.generate_verse(
        memory,
        num_lines=req.num_lines,
        section=req.section,
        arc_token=req.arc_token,
    )

    annotations = [annotation_to_dict(annotate_line(l)) for l in lines]
    scheme_info = detect_scheme(lines) if lines else {}

    # Build analysis
    analysis = None
    try:
        analysis = engine.analyze_song(memory)
    except Exception:
        pass

    return GenerateResponse(
        lines=lines,
        phoneme_annotations=annotations,
        rhyme_scheme_detected=scheme_info.get("scheme_type", "free"),
        workspace=memory.last_workspace,
        analysis=analysis,
    )


@app.post("/analyze")
def analyze_lyrics(req: AnalyzeRequest):
    """Analyse a set of lyrics with the cognitive engine without generating."""
    from src.inference.engine import SongMemory

    engine = get_engine()
    memory = SongMemory(genre=req.genre, mood=req.mood)
    for line in req.lines:
        memory.add_line(line, section="verse1")

    return engine.analyze_song(memory)


@app.post("/cowrite/start", response_model=CoWriteStartResponse)
def cowrite_start(req: CoWriteStartRequest):
    from src.inference.engine import CoWriteSession

    engine = get_engine()
    session = CoWriteSession(engine, genre=req.genre, rhyme_scheme=req.rhyme_scheme)
    session.memory.target_syllables = req.target_syllables
    session.memory.mood = req.mood

    session_id = str(uuid.uuid4())
    _sessions[session_id] = session
    return CoWriteStartResponse(session_id=session_id)


@app.post("/cowrite/suggest", response_model=SuggestResponse)
def cowrite_suggest(req: SuggestRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    candidates = session.suggest(n=req.n)
    return SuggestResponse(
        suggestions=[
            {
                "text": c.text,
                "phonetic_score": round(c.phonetic_score, 3),
                "syllable_ok": c.syllable_ok,
                "novelty_score": round(c.novelty_score, 3),
                "valence_fit": round(c.valence_fit, 3),
                "total_score": round(c.total_score, 3),
                # Metacognitive workspace data
                "workspace_score": round(c.workspace_score, 3),
                "workspace_confidence": round(c.workspace_confidence, 3),
                "workspace_conflict": round(c.workspace_conflict, 3),
                "decision_mode": c.decision_mode,
                "decision_trace": list(c.decision_trace),
            }
            for c in candidates
        ]
    )


@app.post("/cowrite/accept", response_model=AcceptResponse)
def cowrite_accept(req: AcceptRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    session.accept(req.line)
    return AcceptResponse(
        accepted=req.line,
        total_lines=len(session.memory.accepted_lines),
    )


@app.get("/cowrite/song/{session_id}")
def get_song(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {"lyrics": session.get_song()}


@app.get("/cowrite/analysis/{session_id}")
def get_session_analysis(session_id: str):
    """Full cognitive analysis of an in-progress co-write session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    try:
        from src.inference.engine import LyricsEngine
        engine = get_engine()
        return engine.analyze_song(session.memory)
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")


@app.delete("/cowrite/session/{session_id}")
def end_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"deleted": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
