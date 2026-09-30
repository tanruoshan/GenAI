"""Notebook 4 scoring: row scores on real rows, and the statistics on small hand-made cases."""
import numpy as np
import pytest

from blnrepair.data import ROOT
from blnrepair.evaluation import (bootstrap_ci, holm, mcnemar_exact, resample_index, row_scores, wilcoxon_paired,
                                  wrong_fact_rate)
from blnrepair.freeze import load_frozen
from blnrepair.slots import slot_core

needs_data = pytest.mark.skipif(not (ROOT / "data" / "processed" / "corrupted_v2.jsonl").exists(),
                                reason="needs the frozen corrupted_v2.jsonl")


@pytest.fixture(scope="module")
def rows():
    return [r for r in load_frozen("v2") if r["k"]]


def gold_words(row):
    """The gold answer: the core for a visible slot, the whole token for a dropped slot (its frame is gone)."""
    return [p["gold"] if p["dropped"] else slot_core(p["gold"]) for p in sorted(row["ops_per_word"], key=lambda p: p["idx"])]


@needs_data
def test_gold_answer_scores_perfect_and_damaged_text_scores_zero(rows):
    for row in rows:
        good, bad = row_scores(row, gold_words(row)), row_scores(row, None)
        assert good["exact"] == good["fact recovery"] == 1.0 and good["anchor ok"] and good["edits after"] == 0
        assert good["visible facts ok"] == good["visible facts"] and good["dropped facts ok"] == good["dropped facts"]
        assert bad["exact"] == bad["fact recovery"] == 0.0 and not bad["anchor ok"]
        assert bad["edits after"] == bad["edits before"] > 0 and bad["repaired span"] == bad["damaged span"]


@needs_data
def test_fact_slots_split_into_visible_and_dropped(rows):
    for row in rows:
        s = row_scores(row, gold_words(row))
        assert s["visible facts"] + s["dropped facts"] == len(row["fact_idx_in_span"]) and s["visible facts"] >= 1


@needs_data
def test_only_the_anchor_right():
    row = next(r for r in load_frozen("v2") if r["severity"] == "25" and r["k"] > 2)
    plans = sorted(row["ops_per_word"], key=lambda p: p["idx"])
    words = [slot_core(p["gold"]) if p["idx"] == row["anchor_idx"] else "zzz" for p in plans]
    s = row_scores(row, words)
    assert s["anchor ok"] and s["exact"] == pytest.approx(1 / len(plans))


def test_bootstrap_mean_and_pooled_rate():
    index = resample_index(4, 2000, seed=1)
    point, low, high = bootstrap_ci([1, 0, 1, 1], [1, 1, 1, 1], index)
    assert point == 0.75 and 0 <= low <= point <= high <= 1
    assert bootstrap_ci([2, 2, 2, 2], [4, 4, 4, 4], index) == (0.5, 0.5, 0.5)
    assert np.isnan(bootstrap_ci([0, 0], [0, 0], resample_index(2, 10, seed=1))[0])


def test_same_seed_same_draws():
    assert (resample_index(10, 5, seed=3) == resample_index(10, 5, seed=3)).all()


def test_mcnemar_uses_only_discordant_sentences():
    only_a, only_b, p = mcnemar_exact([1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0])
    assert (only_a, only_b) == (6, 0) and p == pytest.approx(2 / 2 ** 6)
    assert mcnemar_exact([1, 0], [1, 0]) == (0, 0, 1.0)


def test_wilcoxon_counts_and_no_difference():
    higher, lower, p = wilcoxon_paired(np.arange(1, 21) / 10, np.zeros(20))
    assert (higher, lower) == (20, 0) and p < 0.001
    assert wilcoxon_paired([0.5, 0.5], [0.5, 0.5]) == (0, 0, 1.0)


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])  # 2 x 0.8 is capped at 1


def test_wrong_fact_rate_reference_points():
    assert wrong_fact_rate([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 0.0  # the score separates perfectly
    assert wrong_fact_rate([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5  # the score does not see the fact
    assert wrong_fact_rate([0.9, 0.1, 0.5], [1, 0, 0]) == 0.0
    assert wrong_fact_rate([0.4, 0.6], [1, 0]) == 1.0
    assert np.isnan(wrong_fact_rate([0.4, 0.6], [1, 1]))


def fake_scores():
    import pandas as pd
    rows = []
    for method, anchor, bs in [("a", [1, 1, 1, 0], [0.9, 0.8, 0.7, 0.2]), ("b", [0, 0, 1, 0], [0.1, 0.2, 0.7, 0.1])]:
        for i, (ok, s) in enumerate(zip(anchor, bs)):
            rows.append({"method": method, "level": "10", "id": f"s{i}", "anchor ok": ok, "BERTScore span": s,
                         "fact recovery": ok, "visible facts ok": ok, "visible facts": 1, "dropped facts ok": 0,
                         "dropped facts": 0})
    return pd.DataFrame(rows)


def test_ci_long_and_paired_tests_on_a_small_table():
    from blnrepair.evaluation import ci_long, paired_tests
    scores, sentences = fake_scores(), ["s3", "s2", "s1", "s0"]
    table = ci_long(scores, ["anchor recovery", "visible fact slots", "dropped fact slots"], sentences,
                    resample_index(4, 500, seed=1)).set_index(["method", "metric"])
    assert table.loc[("a", "anchor recovery"), "value"] == 0.75 and table.loc[("b", "visible fact slots"), "value"] == 0.25
    assert np.isnan(table.loc[("a", "dropped fact slots"), "value"])
    tests = paired_tests(scores, [("a", "b")], ["10"], sentences).iloc[0]
    assert (tests["anchor: first only"], tests["anchor: second only"]) == (2, 0)
    assert (tests["BERTScore: first higher"], tests["BERTScore: second higher"]) == (3, 0)
