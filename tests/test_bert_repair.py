"""Part C: sub-slot parts, the text BERT sees, candidate building, and one word per slot."""
import math

import pytest
import torch

from blnrepair.bert_repair import MASK, BertReranker, char_sim, render, split_parts
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


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "corrupted_v1.jsonl").exists(), reason="frozen data missing")
def test_one_word_per_slot_on_real_dev_rows(tokenizer):
    from blnrepair.freeze import load_frozen
    from blnrepair.preds import format_ok
    from blnrepair.slots import splice
    rows = [r for r in load_frozen("v1") if r["split"] == "dev" and r["severity"] in ("1w", "25")][:4]
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
