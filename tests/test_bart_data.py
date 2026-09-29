"""BART repair: training examples and answer reading (no model needed)."""
import json

import pandas as pd
import pytest

from blnrepair.bart_data import bart_input, bart_target, damaged_record, epoch_examples, read_answer
from blnrepair.corrupt import load_table, settings
from blnrepair.data import ROOT, load_config
from blnrepair.slots import splice

POOL = ROOT / "data" / "processed" / "bert_train_pool.jsonl"
pytestmark = pytest.mark.skipif(not POOL.exists(), reason="needs data/processed/bert_train_pool.jsonl")


@pytest.fixture(scope="module")
def setup():
    calibration = json.loads((ROOT / "reports" / "calibration.json").read_text(encoding="utf-8"))
    pool = pd.read_json(POOL, lines=True).to_dict("records")
    return pool, load_table(calibration), settings(load_config(), calibration)


def test_target_answers_splice_back_to_the_gold_sentence(setup):
    pool, table, cfg = setup
    for row in pool[:200]:
        for level in ["1w", "25", "75"]:
            rec = damaged_record(row, level, "0", table, cfg)
            words = read_answer(bart_target(rec), rec["k"])
            assert splice(rec, words) == " ".join(row["text"].split())


def test_one_bracket_per_slot_and_the_anchor_visible(setup):
    pool, table, cfg = setup
    for row in pool[:200]:
        rec = damaged_record(row, "75", "0", table, cfg)
        source = bart_input(rec)
        assert source.count("{") == rec["k"]
        assert source.count("{ ? }") == sum(p["dropped"] for p in rec["ops_per_word"])
        anchor = next(p for p in rec["ops_per_word"] if p["idx"] == rec["anchor_idx"])
        assert not anchor["dropped"] and f"{{ {anchor['out']} }}" in source


def test_new_damage_every_epoch_and_fixed_validation_draw(setup):
    pool, table, cfg = setup
    sample = pool[:40]
    e0, e1 = epoch_examples(sample, table, cfg, 0), epoch_examples(sample, table, cfg, 1)
    train0 = [e["source"] for e in e0 if e["usage"] == "train"]
    train1 = [e["source"] for e in e1 if e["usage"] == "train"]
    assert train0 != train1 and len(e0) == 5 * len(sample)
    assert [e for e in e0 if e["usage"] == "val"] == [e for e in e1 if e["usage"] == "val"]


def test_read_answer_needs_exactly_k_brackets():
    assert read_answer("the { Harris, } and { ? }", 2) == ["Harris,", "?"]
    assert read_answer("the { Harris, } and", 2) is None
