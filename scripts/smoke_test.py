"""
Smoke test — runs locally with no GPU, no API keys.
Pure-Python components always run; model-dependent sections (torch /
transformers / GPT-2 download) SKIP with a clear message when the local
model stack is missing or broken, instead of crashing.

Run: python scripts/smoke_test.py
"""

import sys
sys.path.insert(0, ".")


class SkipTest(Exception):
    """Raised by a test that cannot run in this environment."""


def _skip_unless(what: str, probe):
    """Run `probe`; convert any failure into a SKIP with a clear message.
    Broken installs (e.g. torch/torchvision version mismatches) raise more
    than ImportError, so every exception counts."""
    try:
        probe()
    except Exception as e:
        raise SkipTest(f"{what} ({e.__class__.__name__}: {e})")


def _require_torch(what: str):
    def probe():
        import torch  # noqa: F401
    _skip_unless(f"{what} needs a working torch install", probe)


def _require_tokenizer_stack(what: str):
    def probe():
        from transformers import AutoTokenizer  # noqa: F401
    _skip_unless(f"{what} needs a working transformers install", probe)


def _require_model_stack(what: str):
    def probe():
        import torch  # noqa: F401
        # PreTrainedModel pulls in modeling_utils -> torchvision, which is
        # exactly the path that breaks on mismatched torch/torchvision.
        from transformers import PreTrainedModel, AutoModelForCausalLM  # noqa: F401
    _skip_unless(f"{what} needs a working torch/transformers install", probe)


def test_phoneme_annotator():
    print("\n-- Phoneme Annotator --")
    from src.data.phoneme_annotator import annotate_line
    ann = annotate_line("I been movin' in silence, they can't feel my weight")
    print(f"  Syllables   : {ann.total_syllables}")
    print(f"  End phoneme : {ann.end_phoneme}")
    print(f"  Stress      : {ann.stress_pattern}")
    assert ann.total_syllables > 0
    # Regression: per-syllable stress must come from CMU vowel digits
    assert annotate_line("silence").stress_pattern == "10"
    assert annotate_line("guitar").stress_pattern == "01"
    assert annotate_line("banana").stress_pattern == "010"
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
    print("\n-- Valence Scorer --")
    from src.data.valence_scorer import score_line
    em = score_line("I been movin' in silence, they can't feel my weight")
    print(f"  Valence : {em.valence:+.3f}")
    print(f"  Arousal : {em.arousal:.3f}")
    assert -1.0 <= em.valence <= 1.0
    print("  PASS")


def test_flow_dna():
    print("\n-- Flow DNA --")
    from src.generation.flow_dna import diagnose
    d = diagnose("I been movin' in silence, they can't feel my weight",
                 section="VERSE", arc_token="[BUILD]")
    print(f"  Target flow : {d['target_flow']}")
    print(f"  Score       : {d['score']:.3f}")
    print(f"  Stress      : {d['actual_stress']}")
    assert 0.0 <= d["score"] <= 1.0
    print("  PASS")


def test_dual_tokenizer():
    print("\n-- Dual Tokenizer (offline, GPT-2) --")
    _require_tokenizer_stack("Dual tokenizer (GPT-2 BPE stream)")
    from src.model.dual_tokenizer import OfflineDualTokenizer
    tok = OfflineDualTokenizer()
    enc = tok.encode("I been movin' in silence, they can't feel my weight")
    print(f"  Semantic IDs (first 8) : {enc.semantic_ids[:8]}")
    print(f"  Phoneme IDs  (first 8) : {enc.phoneme_ids[:8]}")
    assert len(enc.semantic_ids) == len(enc.phoneme_ids)
    print("  PASS")


def test_phonetic_head():
    print("\n-- Phonetic Head --")
    _require_torch("Phonetic head (torch MLP)")
    _require_tokenizer_stack("Phonetic head (phoneme vocab)")
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


def test_lyrics_model():
    print("\n-- Lyrics Model (GPT-2) --")
    _require_model_stack("LyricsModel (GPT-2 + LoRA + phonetic head)")
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
    print("\n-- Inference Engine (GPT-2, 3 beams, metacognitive workspace) --")
    _require_model_stack("Inference engine (GPT-2 generation)")
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
    print("  PASS")


def test_metacognitive_engine():
    print("\n-- Metacognitive Engine (GWT + TRAP + HOT + MSV) --")
    from src.model.metacognitive_engine import MetacognitiveWorkspace

    workspace = MetacognitiveWorkspace()

    # Evaluate 3 sample lines
    candidates = [
        "I been movin' in silence, they can't feel my weight",
        "Gold chains and fast lanes, running with the night",
        "Remember when we had nothing but dreams in our pockets",
    ]
    traces = workspace.evaluate_candidates(
        candidates=candidates,
        genre="hip_hop",
        section="verse1",
        mood="dark",
        target_end_phoneme=None,
        previous_line=None,
        accepted_lines=[],
        line_idx=0,
        tension_state=0.3,
        target_syllables=10,
    )

    assert len(traces) == 3, f"Expected 3 traces, got {len(traces)}"
    print(f"  Traces generated   : {len(traces)}")

    best = traces[0]
    print(f"  Best line          : {best.line[:60]}")
    print(f"  Total score        : {best.total_score:.3f}")
    print(f"  System mode        : {best.system_mode}")
    print(f"  Decision           : {best.decision}")
    print(f"  Module scores      : { {k: round(v, 2) for k, v in best.module_scores.items()} }")
    print(f"  Winning modules    : {best.winning_modules}")
    print(f"  Flags              : {best.all_flags[:5]}")

    # Validate all 9 modules produced scores
    expected_modules = {
        "phonology", "stress", "emotion", "semantic", "structure",
        "texture", "dopamine", "surprise", "flow",
    }
    actual_modules = set(best.module_scores.keys())
    assert expected_modules == actual_modules, f"Missing modules: {expected_modules - actual_modules}"
    print(f"  All 9 modules: PASS")

    # Validate per-module reasoning exists
    assert all(best.module_reasoning.get(m) for m in expected_modules), "Missing reasoning"
    print(f"  Module reasoning: PASS")

    # Validate System 2 mode for cold start (no accepted lines = low experience_match)
    # This is correct behavior: unfamiliar territory = DLPFC deliberative mode
    assert best.system_mode == "system2", f"Expected system2 for cold start, got {best.system_mode}"
    print(f"  System 2 for cold start: PASS (correct - unfamiliar territory)")

    # Test self-model learning
    workspace.accept_line(best)
    report = workspace.get_session_report()
    assert report["total_generations"] == 1
    print(f"  Self-model learning: PASS")

    print("  PASS")


if __name__ == "__main__":
    import traceback
    tests = [
        test_phoneme_annotator,
        test_rhyme_labeler,
        test_valence_scorer,
        test_flow_dna,
        test_metacognitive_engine,
        test_dual_tokenizer,
        test_phonetic_head,
        test_lyrics_model,
        test_inference_engine,
    ]
    passed = 0
    failed = 0
    skipped = 0
    for t in tests:
        try:
            t()
            passed += 1
        except SkipTest as e:
            print(f"  SKIP: {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)
