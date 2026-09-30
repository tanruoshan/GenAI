import pandas as pd
import pytest

from blnrepair.bert_data import (assert_no_leakage, build_masked_input, build_pool, draw_training_span,
                                  epoch_examples, example_stats, excerpt_split, freeze_pool, pool_summary)
from blnrepair.data import ROOT, load_config

RAW = ROOT / load_config()["paths"]["raw_dir"]


def test_excerpt_split_covers_every_id_with_no_overlap():
    ids = [f"doc{i}" for i in range(100)]
    usage = excerpt_split(ids, seed=42, val_frac=0.10)
    assert set(usage) == set(ids)
    counts = pd.Series(usage.values()).value_counts()
    assert counts["val"] == 10
    assert counts["train"] == 90


def test_excerpt_split_is_deterministic():
    ids = [f"doc{i}" for i in range(37)]
    assert excerpt_split(ids, seed=42) == excerpt_split(ids, seed=42)


def test_excerpt_split_val_frac_rounds_and_has_at_least_one():
    ids = [f"doc{i}" for i in range(5)]
    usage = excerpt_split(ids, seed=1, val_frac=0.10)
    assert sum(v == "val" for v in usage.values()) == 1


def test_assert_no_leakage_catches_shared_doc_id():
    pool = pd.DataFrame({"doc_id": ["a", "b"], "text": ["x", "y"]})
    sample = [{"doc_id": "b", "gold_text": "z"}]
    with pytest.raises(AssertionError, match="excerpts"):
        assert_no_leakage(pool, sample)


def test_assert_no_leakage_catches_shared_text():
    pool = pd.DataFrame({"doc_id": ["a"], "text": ["Robert was seen near the bridge"]})
    sample = [{"doc_id": "z", "gold_text": "Robert was seen near the bridge"}]
    with pytest.raises(AssertionError, match="sentence"):
        assert_no_leakage(pool, sample)


def test_assert_no_leakage_passes_when_clean():
    pool = pd.DataFrame({"doc_id": ["a"], "text": ["Robert was seen near the bridge"]})
    sample = [{"doc_id": "z", "gold_text": "A different sentence entirely"}]
    assert_no_leakage(pool, sample)  # does not raise


def test_pool_summary_has_a_total_row():
    pool = pd.DataFrame({"usage": ["train", "train", "val"], "id": ["1", "2", "3"],
                         "doc_id": ["a", "a", "b"], "n_words": [20, 25, 30]})
    counts = pool_summary(pool)
    assert counts.loc["total", "sentences"] == 3
    assert counts.loc["total", "excerpts"] == 2
    assert counts.loc["val", "sentences"] == 1


def test_draw_training_span_is_deterministic():
    tokens = "The Court heard that Robert Jones was seen near the old bridge on Monday night".split()
    fact_idx = [1, 4, 5, 12]
    a = draw_training_span("doc-000", len(tokens), fact_idx, seed=42, epoch=0)
    b = draw_training_span("doc-000", len(tokens), fact_idx, seed=42, epoch=0)
    assert a == b


def test_draw_training_span_stays_in_range_and_anchors_on_a_fact():
    tokens = "The Court heard that Robert Jones was seen near the old bridge on Monday night".split()
    n, fact_idx = len(tokens), [1, 4, 5, 12]
    for epoch in range(10):
        span = draw_training_span("doc-000", n, fact_idx, seed=42, epoch=epoch)
        assert 0 <= span["start"] < span["end"] <= n
        assert span["end"] - span["start"] == span["k"]
        assert span["start"] <= span["anchor"] < span["end"]
        assert span["anchor"] in fact_idx


def test_draw_training_span_changes_over_epochs():
    tokens = "The Court heard that Robert Jones was seen near the old bridge on Monday night".split()
    n, fact_idx = len(tokens), [1, 4, 5, 12]
    draws = {tuple(draw_training_span("doc-000", n, fact_idx, seed=42, epoch=e).values()) for e in range(15)}
    assert len(draws) > 1  # at least one epoch differs from the rest


def test_build_masked_input_matches_option_b_pattern():
    tokens = ["The", "prisoner", "was", "seen", "near", "Neckinger-town", "on", "the", "night"]
    span = {"start": 4, "end": 7}
    pieces = {"Neckinger-town": 4}
    tokenize = lambda w: ["p"] * pieces.get(w, 1)
    text, target = build_masked_input(tokens, span, target_pos=5, tokenize=tokenize)
    assert text == "The prisoner was seen near [MASK] [MASK] [MASK] [MASK] [MASK] the night"
    assert target == "Neckinger-town"


def test_build_masked_input_keeps_punctuation_around_the_target():
    tokens = ["He", "met", "Jones,", "of", "the", "Old", "Bailey."]
    span = {"start": 2, "end": 3}
    tokenize = lambda w: ["p", "p"]  # 2 pieces
    text, target = build_masked_input(tokens, span, target_pos=2, tokenize=tokenize)
    assert text == "He met [MASK] [MASK], of the Old Bailey."
    assert target == "Jones"


def test_build_masked_input_punctuation_only_target_uses_the_whole_token():
    tokens = ["He", "said", ",", "and", "left"]
    span = {"start": 2, "end": 3}
    tokenize = lambda w: ["p"]
    text, target = build_masked_input(tokens, span, target_pos=2, tokenize=tokenize)
    assert target == ","
    assert "[MASK]" in text


def test_build_masked_input_pending_words_get_one_mask_regardless_of_piece_count():
    tokens = ["near", "Neckinger-town", "on"]
    span = {"start": 0, "end": 3}
    tokenize = lambda w: ["p"] * (4 if w == "Neckinger-town" else 1)
    text, target = build_masked_input(tokens, span, target_pos=0, tokenize=tokenize)
    assert text == "[MASK] [MASK] [MASK]"  # target "near" (1 piece) + two pending words, one mask each


def test_epoch_examples_pins_validation_span_across_epochs():
    pool = pd.DataFrame([{"id": "docA-000", "doc_id": "docA",
                          "text": "The Court heard that Robert Jones was seen near the old bridge on Monday night",
                          "n_words": 14, "band": "20-29", "fact_idx": [1, 4, 5, 12], "n_facts": 4, "usage": "val"}])
    tokenize = lambda w: ["p"]
    ex0 = list(epoch_examples(pool, seed=42, epoch=0, tokenize=tokenize))
    ex3 = list(epoch_examples(pool, seed=42, epoch=3, tokenize=tokenize))
    assert ex0 == ex3


def test_epoch_examples_uses_the_given_epoch_for_train_rows():
    tokens = "The Court heard that Robert Jones was seen near the old bridge on Monday night".split()
    pool = pd.DataFrame([{"id": "docA-000", "doc_id": "docA", "text": " ".join(tokens), "n_words": len(tokens),
                          "band": "20-29", "fact_idx": [1, 4, 5, 12], "n_facts": 4, "usage": "train"}])
    tokenize = lambda w: ["p"]
    examples = list(epoch_examples(pool, seed=42, epoch=2, tokenize=tokenize))
    span = draw_training_span("docA-000", len(tokens), [1, 4, 5, 12], 42, 2)
    assert {e["target_pos"] for e in examples} == set(range(span["start"], span["end"]))
    assert all(e["level"] == span["level"] for e in examples)
    assert len(examples) == span["k"]


@pytest.mark.skipif(not (RAW / "Ground Truth").exists(), reason="BLN600 not unpacked")
def test_build_pool_on_the_real_corpus_has_no_leakage():
    cfg = load_config()
    pool, sample, counts = build_pool(cfg)
    assert len(pool) > 0
    assert set(pool["doc_id"]).isdisjoint({r["doc_id"] for r in sample})
    assert set(pool["text"]).isdisjoint({r["gold_text"] for r in sample})
    assert counts.loc["total", "sentences"] == len(pool)
    val_share = counts.loc["val", "excerpts"] / counts.loc["total", "excerpts"]
    assert 0.05 < val_share < 0.15


def test_freeze_pool_writes_once_then_checks_and_stops_on_change(tmp_path):
    pool = pd.DataFrame([{"id": "a-000", "text": "one two", "usage": "train"}])
    path, sha = tmp_path / "pool.jsonl", tmp_path / "pool.sha256"
    status, digest = freeze_pool(pool, path, sha)
    assert status == "written" and b"\r\n" not in path.read_bytes()
    assert freeze_pool(pool, path, sha) == ("unchanged", digest)
    assert sha.read_text(encoding="utf-8").split()[0] == digest
    with pytest.raises(RuntimeError):
        freeze_pool(pool.assign(text="one three"), path, sha)


class FakeTokenizer:
    """Pieces = characters; [MASK] is one token with id 0."""
    mask_token, mask_token_id = "[MASK]", 0

    def tokenize(self, word):
        return list(word)

    def __call__(self, text):
        return {"input_ids": [0 if w == "[MASK]" else 1 for w in text.split() for _ in ([w] if w == "[MASK]" else w)]}


def test_example_stats_counts_masks_and_levels():
    examples = [{"usage": "train", "level": "75", "target_word": "ab", "masked_input": "x [MASK] [MASK] [MASK] y"},
                {"usage": "val", "level": "1w", "target_word": "a", "masked_input": "x [MASK] y"}]
    stats = example_stats(examples, FakeTokenizer())
    assert stats["mask_mismatches"] == 0 and stats["examples"] == {"train": 1, "val": 1}
    assert stats["target_pieces (4 = 4 or more)"] == {1: 1, 2: 1}
    assert stats["train_share_by_level"]["75"] == 1.0 and stats["max_length"] == 5
