"""API schema smoke tests via FastAPI TestClient with the model mocked.

The server module itself never imports torch at the top level, so these run
without a working model stack.
"""

import pytest
from fastapi.testclient import TestClient

import src.api.server as server
from src.inference.engine import CandidateScore

client = TestClient(server.app)

AABB_LINES = [
    "I been movin' in silence, they can't feel my weight",
    "Every step I take, yeah I'm moving with fate",
    "They say the game is cold but I turn up the heat",
    "Diamonds on my wrist while I dance to the beat",
]


def _candidate(text: str, score: float = 0.8) -> CandidateScore:
    return CandidateScore(
        text=text,
        phonetic_score=1.0, syllable_ok=True, novelty_score=0.7, valence_fit=0.6,
        trajectory_fit=0.6, texture_alignment=0.5, goosebump=0.4, hook_dna=0.4,
        polysyllabic_rhyme=0.5, internal_rhyme=0.3, complexity=0.5,
        temporal_arc=0.5, introspection=0.3, stress_alignment=0.5,
        total_score=score,
    )


class FakeEngine:
    def generate_verse(self, memory, num_lines=8, section="VERSE",
                       arc_token="[SETUP]", auto_accept=True):
        lines = [AABB_LINES[i % 4] for i in range(num_lines)]
        for line in lines:
            memory.add_line(line, section=section)
        return lines

    def generate_line(self, memory, *args, top_n=1, **kwargs):
        return [_candidate(AABB_LINES[i % 4], 0.9 - 0.1 * i) for i in range(top_n)]


@pytest.fixture()
def fake_engine(monkeypatch):
    monkeypatch.setattr(server, "get_engine", lambda: FakeEngine())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_genres():
    r = client.get("/genres")
    assert r.status_code == 200
    names = [g["name"] for g in r.json()]
    assert "trap" in names and "hip_hop" in names


def test_generate_schema(fake_engine):
    r = client.post("/generate", json={
        "genre": "trap", "section": "VERSE", "arc_token": "[BUILD]",
        "num_lines": 4, "rhyme_scheme": "AABB",
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["lines"]) == 4
    assert len(body["phoneme_annotations"]) == 4
    assert body["rhyme_scheme_detected"] == "AABB"
    ann = body["phoneme_annotations"][0]
    assert {"text", "total_syllables", "end_phoneme", "stress_pattern", "words"} <= set(ann)


def test_generate_rejects_unknown_genre(fake_engine):
    r = client.post("/generate", json={"genre": "yodelcore"})
    assert r.status_code == 400


def test_generate_validates_num_lines(fake_engine):
    r = client.post("/generate", json={"genre": "trap", "num_lines": 999})
    assert r.status_code == 422


def test_cowrite_flow(fake_engine):
    r = client.post("/cowrite/start", json={"genre": "rnb", "rhyme_scheme": "ABAB"})
    assert r.status_code == 200
    sid = r.json()["session_id"]

    r = client.post("/cowrite/suggest", json={"session_id": sid, "n": 3})
    assert r.status_code == 200
    suggestions = r.json()["suggestions"]
    assert len(suggestions) == 3
    assert {"text", "phonetic_score", "syllable_ok", "total_score"} <= set(suggestions[0])

    line = suggestions[0]["text"]
    r = client.post("/cowrite/accept", json={"session_id": sid, "line": line})
    assert r.status_code == 200
    assert r.json() == {"accepted": line, "total_lines": 1}

    r = client.get(f"/cowrite/song/{sid}")
    assert r.status_code == 200
    assert r.json()["lyrics"] == line

    r = client.delete(f"/cowrite/session/{sid}")
    assert r.status_code == 200


def test_cowrite_unknown_session():
    r = client.post("/cowrite/suggest", json={"session_id": "nope", "n": 1})
    assert r.status_code == 404
