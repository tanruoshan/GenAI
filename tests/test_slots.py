"""Section 0a: slot view and splice on the frozen data (all 1,020 rows)."""
import pytest

from blnrepair.data import ROOT
from blnrepair.freeze import load_frozen
from blnrepair.slots import DROPPED, build_slots, slot_core, slot_view, splice, split_punct

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "processed" / "corrupted_v2.jsonl").exists(),
                                reason="needs the frozen corrupted_v2.jsonl")


@pytest.fixture(scope="module")
def rows():
    rows = load_frozen("v2")
    assert len(rows) == 1020
    return rows


def plans(row):
    return sorted(row["ops_per_word"], key=lambda p: p["idx"])


def test_gold_words_splice_back_to_gold_text(rows):
    for r in rows:
        assert splice(r, [p["gold"] for p in plans(r)]).split() == r["gold_tokens"], (r["id"], r["severity"])


def test_gold_cores_also_splice_back_to_gold_text(rows):
    # What a model returns for a non-dropped slot is usually the bare word, without punctuation.
    for r in rows:
        preds = [p["gold"] if p["dropped"] else split_punct(p["gold"])[1] for p in plans(r)]
        assert splice(r, preds) == r["gold_text"], (r["id"], r["severity"])


def test_out_forms_splice_back_to_corrupted_text(rows):
    for r in rows:
        assert splice(r, [p["out"] for p in plans(r)]) == r["corrupted_text"], (r["id"], r["severity"])


def test_slot_count_equals_k_and_anchor_is_visible(rows):
    for r in rows:
        view = build_slots(r)
        assert len(view["slots"]) == r["k"] == sum(w is None for w in view["words"])
        anchors = [s for s in view["slots"] if s["anchor"]]
        assert len(anchors) == (1 if r["k"] else 0)
        assert all(not s["dropped"] and s["core"] for s in anchors)


def test_slot_view_has_no_gold_of_damaged_words(rows):
    for r in rows:
        view = build_slots(r)
        assert all(set(s) == {"idx", "out", "dropped", "lead", "core", "trail", "anchor", "fact"} for s in view["slots"])
        text = slot_view(view)
        assert text.count("⟨") == r["k"] and text.count(DROPPED) == sum(p["dropped"] for p in r["ops_per_word"])


def test_level_0_has_no_slots(rows):
    for r in rows:
        if r["severity"] == "0":
            view = build_slots(r)
            assert view["slots"] == [] and slot_view(view) == r["gold_text"] and splice(r, []) == r["gold_text"]


def test_inner_apostrophe_moved_to_the_edge():
    # I'll -> 'll: the model sees ⟨'ll⟩, and a correct answer must splice back as I'll, not 'I'll.
    row = {"id": "x", "severity": "1w", "gold_tokens": ["so", "I'll", "go"], "ops_per_word": [
        {"idx": 1, "gold": "I'll", "out": "'ll", "dropped": False}]}
    assert splice(row, ["I'll"]) == "so I'll go"
    assert splice(row, ["'ll"]) == "so 'll go"
    row["gold_tokens"][1] = "“Who's"
    row["ops_per_word"][0].update(gold="“Who's", out="“Wh'")
    assert splice(row, ["Who's"]) == splice(row, ["“Who's"]) == "so “Who's go"
    assert splice(row, ["“Wh'"]) == "so “Wh' go"


def test_prediction_punctuation_is_not_doubled():
    row = {"id": "x", "severity": "25", "gold_tokens": ["Smith", "and", "Co.,", "whip"], "ops_per_word": [
        {"idx": 2, "gold": "Co.,", "out": "C0.,", "dropped": False}]}
    for pred in ["Co", "Co.,", "Co.", " Co, "]:
        assert splice(row, [pred]) == "Smith and Co., whip"


def test_punctuation_only_gold_word():
    # A lone "," has no letters, so the damage always dropped it; scoring compares the whole token.
    row = {"id": "x", "severity": "10", "gold_tokens": ["100l.", ",", "and"], "ops_per_word": [
        {"idx": 1, "gold": ",", "out": "", "dropped": True}]}
    assert splice(row, [","]) == "100l. , and"
    assert splice(row, [""]) == "100l. and"
    assert slot_core(",") == "," and slot_core(" Harris, ") == "Harris" and slot_core("I'll") == "I'll"


def test_load_frozen_stops_on_hash_mismatch(tmp_path):
    (tmp_path / "corrupted_v1.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "corrupted_v1.sha256").write_text("0" * 64 + "  corrupted_v1.jsonl\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        load_frozen("v1", tmp_path, tmp_path)
