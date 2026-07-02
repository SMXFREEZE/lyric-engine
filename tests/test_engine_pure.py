"""Pure-Python tests for the inference engine: SongMemory prompt building,
rhyme-target tracking, style-vector conditioning, and the hard rhyme filter.

No model is loaded — LyricsEngine.generate_candidates is replaced with a
canned generator (conftest stubs torch/transformers if the real stack is
missing or broken).
"""

import numpy as np

from src.inference.engine import LyricsEngine, SongMemory


class _FakeModel:
    def eval(self):
        return self


def _make_engine() -> LyricsEngine:
    return LyricsEngine(_FakeModel(), tokenizer=None, device="cpu", beam_size=2)


# ── SongMemory ────────────────────────────────────────────────────────────────

def test_aabb_rhyme_target():
    m = SongMemory(genre="hip_hop", rhyme_scheme="AABB")
    assert m.get_target_end_phoneme() is None  # line 0 is free
    m.add_line("they can't feel my weight")
    target = m.get_target_end_phoneme()        # line 1 must rhyme with line 0
    assert target == "EY1 T"


def test_free_scheme_has_no_target():
    m = SongMemory(genre="hip_hop", rhyme_scheme="free")
    m.add_line("they can't feel my weight")
    assert m.get_target_end_phoneme() is None


def test_ccl_prompt_structure():
    m = SongMemory(genre="trap")
    m.sections.append(("[BUILD]", "VERSE"))
    prompt = m.build_prompt()
    for marker in ("[INST]", "[PERCEIVE]", "[INTENT]", "[PREDICT]"):
        assert marker in prompt
    assert "flow=standard" in prompt  # target_syllables=10 default


def test_style_vec_conditions_rhythm_label():
    vec = np.zeros(128, dtype=np.float32)
    vec[0] = 14.2  # artist averages dense 14-syllable lines
    m = SongMemory(genre="trap", style_vec=vec)
    m.sections.append(("[BUILD]", "VERSE"))
    assert "flow=dense" in m.build_prompt()


def test_style_vec_out_of_range_falls_back():
    vec = np.zeros(128, dtype=np.float32)
    vec[0] = 99.0  # implausible → ignore, use target_syllables
    m = SongMemory(genre="trap", style_vec=vec)
    m.sections.append(("[BUILD]", "VERSE"))
    assert "flow=standard" in m.build_prompt()


# ── Hard rhyme filter ────────────────────────────────────────────────────────

RHYMER = "every step I take I'm moving with fate"      # EY1 T — rhymes
MISSER = "they say it's cold but I turn up the heat"   # IY1 T — does not


def _memory_needing_rhyme() -> SongMemory:
    m = SongMemory(genre="hip_hop", rhyme_scheme="AABB", target_syllables=10)
    m.sections.append(("[SETUP]", "VERSE"))
    m.add_line("I been movin' in silence, they can't feel my weight")
    return m


def test_rhyme_filter_rejects_non_rhyming_candidates():
    engine = _make_engine()
    engine.generate_candidates = lambda prompt, **kw: [MISSER, RHYMER]
    top = engine.generate_line(_memory_needing_rhyme(), top_n=5)
    assert [c.text for c in top] == [RHYMER]
    assert all("RHYME_MISS" not in t.all_flags for t in engine.last_traces)


def test_rhyme_filter_retries_then_falls_back():
    engine = _make_engine()
    calls = {"n": 0}

    def only_missers(prompt, **kw):
        calls["n"] += 1
        return [MISSER]

    engine.generate_candidates = only_missers
    top = engine.generate_line(_memory_needing_rhyme(), top_n=3)
    assert calls["n"] == 2          # one retry batch was sampled
    assert len(top) >= 1            # still emits output (fallback)
    assert top[0].text == MISSER


def test_rhyme_filter_retry_can_recover():
    engine = _make_engine()
    batches = iter([[MISSER], [RHYMER]])
    engine.generate_candidates = lambda prompt, **kw: next(batches)
    top = engine.generate_line(_memory_needing_rhyme(), top_n=3)
    assert top[0].text == RHYMER


def test_no_filter_when_no_rhyme_required():
    engine = _make_engine()
    engine.generate_candidates = lambda prompt, **kw: [MISSER]
    m = SongMemory(genre="hip_hop", rhyme_scheme="free")
    m.sections.append(("[SETUP]", "VERSE"))
    top = engine.generate_line(m, top_n=1)
    assert top[0].text == MISSER
