"""Section 0b: prediction store, resume, test-split guard and the slot_exact sanity check."""
import pytest

from blnrepair.data import ROOT
from blnrepair.freeze import load_frozen
from blnrepair.preds import PRED_FIELDS, append_pred, load_preds, pred_path, row_key, run_method, slot_exact
from blnrepair.slots import split_punct

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "processed" / "corrupted_v2.jsonl").exists(),
                                reason="needs the frozen corrupted_v2.jsonl")


@pytest.fixture(scope="module")
def dev_rows():
    return [r for r in load_frozen("v2") if r["split"] == "dev" and r["severity"] in ("0", "1w", "25")][:9]


def gold_predict(row_by_key):
    """A fake method that answers with the gold cores (dropped slots: the gold token)."""
    calls = []

    def predict(view):
        row = row_by_key[(view["id"], view["severity"])]
        plans = sorted(row["ops_per_word"], key=lambda p: p["idx"])
        words = [p["gold"] if p["dropped"] else split_punct(p["gold"])[1] for p in plans]
        calls.append(view["id"])
        return {"raw_output": str(words), "pred_words": words}
    return predict, calls


def test_run_skips_level_0_and_resumes_without_new_calls(dev_rows, tmp_path):
    by_key = {(r["id"], r["severity"]): r for r in dev_rows}
    predict, calls = gold_predict(by_key)
    added = run_method(dev_rows, predict, "fake", "v1", "gold-oracle", preds_dir=tmp_path)
    assert len(added) == len(calls) == sum(1 for r in dev_rows if r["k"]) == 6
    assert all(r["format_ok"] and set(r) == set(PRED_FIELDS) for r in added)
    assert run_method(dev_rows, predict, "fake", "v1", "gold-oracle", preds_dir=tmp_path) == []
    assert len(calls) == 6 and len(load_preds(pred_path("fake", "v1", tmp_path))) == 6


def test_resume_after_a_stop_finishes_only_the_rest(dev_rows, tmp_path):
    by_key = {(r["id"], r["severity"]): r for r in dev_rows}
    predict, calls = gold_predict(by_key)
    run_method(dev_rows[:4], predict, "fake", "v1", "gold-oracle", preds_dir=tmp_path)
    before = len(calls)
    run_method(dev_rows, predict, "fake", "v1", "gold-oracle", preds_dir=tmp_path)
    assert len(calls) - before == 6 - sum(1 for r in dev_rows[:4] if r["k"])


def test_gold_predictions_splice_to_gold_and_score_one(dev_rows, tmp_path):
    by_key = {(r["id"], r["severity"]): r for r in dev_rows}
    predict, _ = gold_predict(by_key)
    for rec in run_method(dev_rows, predict, "fake", "v1", "gold-oracle", preds_dir=tmp_path):
        row = by_key[(rec["id"], rec["severity"])]
        assert rec["spliced_text"] == row["gold_text"]
        assert slot_exact(rec["pred_words"], row) == 1.0


def test_test_split_is_closed_by_default(tmp_path):
    test_row = next(r for r in load_frozen("v2") if r["split"] == "test" and r["k"])
    with pytest.raises(AssertionError, match="test split is closed"):
        run_method([test_row], lambda v: {"pred_words": []}, "fake", "v1", "m", preds_dir=tmp_path)
    assert not pred_path("fake", "v1", tmp_path).exists()


def test_bad_answers_are_stored_as_format_failures_not_fixed(dev_rows, tmp_path):
    row = next(r for r in dev_rows if r["k"] > 1)
    for name, out in {"short": {"raw_output": "[]", "pred_words": ["x"]},
                      "nolist": {"raw_output": "sorry", "pred_words": None, "error": "no array"},
                      "nonstr": {"raw_output": "[1]", "pred_words": [1] * row["k"]}}.items():
        (rec,) = run_method([row], lambda v, out=out: out, name, "v1", "m", preds_dir=tmp_path)
        assert rec["format_ok"] is False and rec["pred_words"] is None and rec["spliced_text"] is None
        assert slot_exact(out["pred_words"], row) == 0.0


def test_a_crash_in_predict_stores_nothing_for_that_row(dev_rows, tmp_path):
    row = next(r for r in dev_rows if r["k"])

    def boom(view):
        raise TimeoutError("network")
    with pytest.raises(TimeoutError):
        run_method([row], boom, "fake", "v1", "m", preds_dir=tmp_path)
    assert load_preds(pred_path("fake", "v1", tmp_path)) == []


def test_duplicate_key_is_refused(dev_rows, tmp_path):
    by_key = {(r["id"], r["severity"]): r for r in dev_rows}
    predict, _ = gold_predict(by_key)
    (rec,) = run_method(dev_rows[1:2], predict, "fake", "v1", "m", preds_dir=tmp_path)
    with pytest.raises(ValueError, match="already stored"):
        append_pred(pred_path("fake", "v1", tmp_path), rec)
    assert row_key(rec) == (rec["id"], rec["severity"], "fake", "v1")


def test_slot_exact_is_case_sensitive_and_counts_partial(dev_rows):
    row = next(r for r in dev_rows if r["k"] >= 3)
    plans = sorted(row["ops_per_word"], key=lambda p: p["idx"])
    words = [p["gold"] if p["dropped"] else split_punct(p["gold"])[1] for p in plans]
    words[0] = "zzz"
    assert slot_exact(words, row) == pytest.approx((len(plans) - 1) / len(plans))
    words[0] = words[0].swapcase()
    assert slot_exact(words, row) == pytest.approx((len(plans) - 1) / len(plans))
    assert slot_exact(words[:-1], row) == 0.0  # wrong length counts as 0
