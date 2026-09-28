"""Part C: sub-slot parts, the text BERT sees, candidate building, and one word per slot."""
import math

import pytest
import torch

from blnrepair.bert_repair import MASK, BertReranker, char_sim, load_repair_config, render, repair_version, split_parts
from blnrepair.data import ROOT
from blnrepair.slots import build_slots


def slot(out, dropped=False, lead="", core=None, trail=""):
    return {"out": out, "dropped": dropped, "lead": lead, "core": out if core is None else core, "trail": trail}


VIEW = {"words": ["At", None, None, "the", None, "court."],
        "slots": [slot("Pol1ce-coart"), slot("", dropped=True, core=""), slot("Grecnwich,", core="Grecnwich", trail=",")]}


def test_split_parts_at_visible_inner_punctuation():
    assert split_parts(slot("Pol1ce-coart")) == (["Pol1ce", "coart"], ["-"])
    assert split_parts(slot("Scndder's")) == (["Scndder", "s"], ["'"])
    assert split_parts(slot("", dropped=True, core="")) == ([""], [])
    assert split_parts(slot("Harris")) == (["Harris"], [])


def test_char_sim_is_plain_normalised_levenshtein():
    assert char_sim("Thomas", "Thomas") == 1.0
    assert char_sim("Thomas", "Bbonias") == pytest.approx(1 - 4 / 7)
    assert char_sim("a", "A") == 0.0  # case-sensitive


def test_render_waiting_slots_show_one_mask_and_their_punctuation():
    fills = [[], [], []]
    assert render(VIEW, fills) == f"At {MASK} {MASK} the {MASK}, court."
    assert render(VIEW, fills, (0, 2)) == f"At {MASK} {MASK}-{MASK} {MASK} the {MASK}, court."
    fills[0] = ["Police"]
    assert render(VIEW, fills, (0, 1)) == f"At Police-{MASK} {MASK} the {MASK}, court."
    fills[0].append("court")
    assert render(VIEW, fills, (1, 3)) == f"At Police-court {MASK} {MASK} {MASK} the {MASK}, court."


class TinyTokenizer:
    """Only what BertReranker.candidates needs: a vocabulary."""
    vocab = ["[PAD]", "[MASK]", "Tho", "##mas", "the", ",", "##s", "Thomas"]

    def __len__(self):
        return len(self.vocab)

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[i] for i in ids]


class NoModel(torch.nn.Module):
    pass


def test_candidates_join_pieces_and_filter_punctuation_for_visible_slots():
    r = BertReranker(NoModel(), TinyTokenizer(), beam=2, top_n=10)
    lp1 = torch.log(torch.tensor([[0, 0, .1, .2, .3, .4, 0, .0]]) + 1e-9)
    lp2 = torch.log(torch.tensor([[0, 0, .6, .1, .1, .1, 0, .1], [0, 0, 0, .7, 0, 0, .3, 0]]) + 1e-9)
    # vocabulary: [PAD] [MASK] Tho ##mas the , ##s Thomas
    visible = r.candidates([lp1, lp2], allow_punct=False)
    assert "," not in visible and "the" in visible
    assert visible["Thomas"] == pytest.approx((math.log(.6) + math.log(.7)) / 2, abs=1e-6)
    assert not any(c.startswith("##") for c in visible)  # a ## piece never starts a word
    assert "," in r.candidates([lp1, lp2], allow_punct=True)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("bert-base-cased", local_files_only=True)
    except OSError:
        pytest.skip("bert-base-cased tokenizer not in the local cache")


class FlatModel(torch.nn.Module):
    """Every piece equally likely: exercises the whole fill loop without downloading a model."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        return type("Out", (), {"logits": torch.zeros(*input_ids.shape, self.vocab_size)})()


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "corrupted_v2.jsonl").exists(), reason="frozen data missing")
def test_one_word_per_slot_on_real_dev_rows(tokenizer):
    from blnrepair.freeze import load_frozen
    from blnrepair.preds import format_ok
    from blnrepair.slots import splice
    rows = [r for r in load_frozen("v2") if r["split"] == "dev" and r["severity"] in ("1w", "25")][:4]
    r = BertReranker(FlatModel(len(tokenizer)), tokenizer, lam=1.0)
    for row in rows:
        out = r(build_slots(row))
        assert format_ok(out["pred_words"], row["k"])
        splice(row, out["pred_words"])  # must not raise


def test_lambda_trades_bert_score_against_letter_similarity(monkeypatch):
    r = BertReranker(NoModel(), TinyTokenizer())
    monkeypatch.setattr(r, "mask_logprobs", lambda texts: [])
    monkeypatch.setattr(r, "candidates", lambda lps, allow_punct: {"court": -5.0, "count": -1.0})
    view = {"id": "x", "severity": "1w", "words": ["the", None], "slots": [slot("coart")]}  # sim 0.8 vs 0.6
    r.lam = 0.0
    assert r(view)["pred_words"] == ["count"]  # context only
    r.lam = 30.0
    assert r(view)["pred_words"] == ["court"]  # letters dominate
    dropped = {"id": "x", "severity": "1w", "words": ["the", None], "slots": [slot("", dropped=True, core="")]}
    assert r(dropped)["pred_words"] == ["count"]  # a dropped slot uses the BERT score only


def test_repair_version_holds_model_settings_and_split():
    cfg = load_repair_config()
    assert repair_version({**cfg, "lam": 1.0, "beam": 5, "top_n": 10}, "dev") == f"{cfg['model_tag']}-l1-b5-n10-dev"
    assert repair_version({**cfg, "lam": 0.5}, "test").endswith("-test")
    assert repair_version({**cfg, "lam": 0.5}, "dev") != repair_version({**cfg, "lam": 2}, "dev")


class WordTokenizer(TinyTokenizer):
    """TinyTokenizer plus word tokenization, for the dictionary candidates."""
    vocab = ["[PAD]", "[MASK]", "[UNK]", "Hu", "##rd", "Hard", "the"]
    unk_token_id = 2
    words = {"Hurd": [3, 4], "Hard": [5], "Longword": [3, 4, 4, 4], "Hürd": [2]}

    def __call__(self, word, add_special_tokens=True):
        return {"input_ids": self.words[word]}


def test_dictionary_candidates_are_scored_from_the_pass_with_their_piece_count():
    r = BertReranker(NoModel(), WordTokenizer(), lex_n=4)
    r.lex = type("Lex", (), {"closest": lambda self, core, n: ["Hurd", "Hard", "Longword", "Hürd"][:n]})()
    lp1, lp2, lp3 = torch.randn(1, 7), torch.randn(2, 7), torch.randn(3, 7)
    out = r.dictionary_candidates([lp1, lp2, lp3], "Hnrd")
    assert out["Hurd"] == pytest.approx(((lp2[0, 3] + lp2[1, 4]) / 2).item())
    assert out["Hard"] == pytest.approx(lp1[0, 5].item())
    assert "Longword" not in out and "Hürd" not in out  # more than 3 pieces, or an unknown piece


def test_part_counts_split_entries_like_slots():
    from collections import Counter
    from blnrepair.bert_repair import part_counts
    assert part_counts(Counter({"Police-court": 2, "court": 1, "prisoner's": 1})) == Counter(
        {"court": 3, "Police": 2, "prisoner": 1, "s": 1})


def test_repair_version_names_the_dictionary_candidates():
    cfg = {"model_tag": "ftv1", "lam": 4, "beam": 5, "top_n": 10}
    assert repair_version(cfg, "dev") == "ftv1-l4-b5-n10-dev"  # unchanged without a word list
    assert repair_version({**cfg, "lex_n": 20}, "dev") == "ftv1-l4-b5-n10-x20-dev"
