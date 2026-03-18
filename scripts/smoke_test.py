"""
Smoke test — runs locally with no GPU, no API keys.
Tests the full pipeline with GPT-2 as the base model.

Run: python scripts/smoke_test.py
"""

import sys
sys.path.insert(0, ".")

def test_phoneme_annotator():
    print("\n-- Phoneme Annotator --")
    from src.data.phoneme_annotator import annotate_line
    ann = annotate_line("I been movin' in silence, they can't feel my weight")
    print(f"  Syllables   : {ann.total_syllables}")
    print(f"  End phoneme : {ann.end_phoneme}")
    print(f"  Stress      : {ann.stress_pattern}")
    assert ann.total_syllables > 0
    print("  PASS")


def test_rhyme_labeler():
    print("\n-- Rhyme Labeler --")
    from src.data.rhyme_labeler import detect_scheme
    lines = [
        "I been movin' in silence, they can't feel my weight",
        "Every step I take, yeah I'm moving with fate",
        "They say the game is cold but I turn up the heat",
        "Diamonds on my wrist while I dance to the beat",
    ]
    result = detect_scheme(lines)
    print(f"  Scheme  : {result['scheme_str']} ({result['scheme_type']})")
    print(f"  Density : {result['rhyme_density']}")
    assert result["rhyme_density"] > 0
    print("  PASS")


def test_valence_scorer():
    print("\n--Valence Scorer --")
    from src.data.valence_scorer import score_line
    em = score_line("I been movin' in silence, they can't feel my weight")
    print(f"  Valence : {em.valence:+.3f}")
    print(f"  Arousal : {em.arousal:.3f}")
    assert -1.0 <= em.valence <= 1.0
    print("  PASS")


def test_dual_tokenizer():
    print("\n--Dual Tokenizer (offline, GPT-2) --")
    from src.model.dual_tokenizer import OfflineDualTokenizer
    tok = OfflineDualTokenizer()
    enc = tok.encode("I been movin' in silence, they can't feel my weight")
    print(f"  Semantic IDs (first 8) : {enc.semantic_ids[:8]}")
    print(f"  Phoneme IDs  (first 8) : {enc.phoneme_ids[:8]}")
    assert len(enc.semantic_ids) == len(enc.phoneme_ids)
    print("  PASS")


def test_phonetic_head():
    print("\n--Phonetic Head --")
    import torch
    from src.model.phonetic_head import PhoneticHead, PhoneticConstraintScorer
    from src.model.dual_tokenizer import PHONEME_TO_ID

    head = PhoneticHead(d_model=64, hidden=32)
    scorer = PhoneticConstraintScorer(head, device="cpu")

    dummy = torch.randn(4, 5, 64)  # batch=4, seq=5, d_model=64
    target_id = PHONEME_TO_ID.get("EY1", 10)
    beam_scores = torch.tensor([-1.0, -1.5, -0.8, -2.0])
    ranked = scorer.rerank_beams(dummy, beam_scores, target_id)
    print(f"  Reranked indices: {ranked}")
    assert len(ranked) == 4
    print("  PASS")


def test_lyrics_provider_ranking():
    print("\n--Lyrics Provider Ranking --")
    from src.data.lyrics_providers import LyricsCandidate, rank_lyrics_candidates

    ranked = rank_lyrics_candidates([
        LyricsCandidate(
            artist="Test Artist",
            title="Test Song",
            lyrics="hello",
            source="lyrics_ovh",
            quality_score=0.78,
        ),
        LyricsCandidate(
            artist="Test Artist",
            title="Test Song",
            lyrics="Line one of the record\nLine two of the record\nLine three of the record",
            source="vagalume",
            quality_score=1.12,
        ),
    ])
    print(f"  Best source : {ranked[0].source}")
    assert ranked[0].source == "vagalume"
    print("  PASS")


def test_metacognitive_engine():
    print("\n--Metacognitive Engine --")
    from src.model.metacognitive_engine import (
        CandidateEvidence,
        MetacognitiveEngine,
        SelfModel,
        WorkspaceContext,
    )

    engine = MetacognitiveEngine()
    self_model = SelfModel()
    context = WorkspaceContext(
        genre="hip_hop",
        mood="dark",
        section="verse1",
        bar_index=1,
        rhyme_scheme="AABB",
        target_end_phoneme="EY1 T",
        target_syllables=10,
        target_arc_valence=0.1,
        target_arc_arousal=0.6,
        tension_state=0.45,
        accepted_lines=["I been movin in silence, they can't feel my weight"],
    )
    decision = engine.evaluate_candidates(
        [
            CandidateEvidence(
                text="Every scar on my heart still glow when I create",
                base_score=0.74,
                phonetic_score=0.95,
                syllable_ok=True,
                novelty_score=0.82,
                valence_fit=0.70,
                trajectory_fit=0.78,
                texture_alignment=0.75,
                goosebump=0.68,
                hook_dna=0.58,
                polysyllabic_rhyme=0.72,
                internal_rhyme=0.44,
                complexity=0.69,
                temporal_arc=0.63,
                introspection=0.60,
                stress_alignment=0.72,
            ),
            CandidateEvidence(
                text="[no candidate generated]",
                base_score=0.20,
                phonetic_score=0.0,
                syllable_ok=False,
                novelty_score=0.0,
                valence_fit=0.2,
                trajectory_fit=0.1,
                texture_alignment=0.1,
                goosebump=0.0,
                hook_dna=0.0,
                polysyllabic_rhyme=0.0,
                internal_rhyme=0.0,
                complexity=0.0,
                temporal_arc=0.0,
                introspection=0.0,
                stress_alignment=0.0,
            ),
        ],
        context=context,
        self_model=self_model,
    )
    print(f"  Mode       : {decision.metacognition.mode}")
    print(f"  Confidence : {decision.metacognition.confidence:.3f}")
    print(f"  Chosen     : {decision.chosen_text}")
    print(f"  HOT trace  : {decision.hot_trace.get('self_statement', '')[:72]}")
    assert decision.chosen_text.startswith("Every scar")
    assert decision.self_model_snapshot["last_mode"] in {"system1", "system2"}
    assert "domain_memory" in decision.working_memory.to_dict()
    assert decision.hot_trace["winning_modules"]
    print("  PASS")


def test_lyrics_model():
    print("\n--Lyrics Model (GPT-2) --")
    import torch
    from src.model.lyrics_model import load_base_model, LyricsModel

    base, tok = load_base_model("gpt2")
    model = LyricsModel(base, d_model=base.config.hidden_size)
    model.eval()

    ids = tok("[VERSE] I been movin in silence", return_tensors="pt")
    with torch.no_grad():
        out = model(
            input_ids=ids["input_ids"],
            attention_mask=ids["attention_mask"],
            style_vec=torch.randn(1, 128),
        )
    print(f"  LM logits shape     : {out['lm_logits'].shape}")
    print(f"  Phoneme logits shape: {out['phoneme_logits'].shape}")
    print("  PASS")


def test_inference_engine():
    print("\n--Inference Engine (GPT-2, 3 beams) --")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.inference.engine import LyricsEngine, SongMemory

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained("gpt2")

    engine = LyricsEngine(mdl, tok, device="cpu", beam_size=3)
    memory = SongMemory(genre="hip_hop", rhyme_scheme="AABB", target_syllables=10)
    memory.sections.append(("[SETUP]", "VERSE"))

    candidates = engine.generate_line(memory, top_n=3)
    print(f"  Got {len(candidates)} candidates:")
    for c in candidates:
        print(f"    [{c.total_score:.2f}] {c.text[:60]}")
    assert len(candidates) > 0
    assert memory.last_workspace is not None
    assert "hot_trace" in memory.last_workspace
    memory.add_line(candidates[0].text, section="verse1")
    analysis = engine.analyze_song(memory)
    assert "common_hot_focus" in analysis["metacognition"]
    print("  PASS")


if __name__ == "__main__":
    import traceback
    tests = [
        test_phoneme_annotator,
        test_rhyme_labeler,
        test_valence_scorer,
        test_dual_tokenizer,
        test_phonetic_head,
        test_lyrics_provider_ranking,
        test_metacognitive_engine,
        test_lyrics_model,
        test_inference_engine,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
