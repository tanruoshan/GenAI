import json
import re

import pytest

from blnrepair.corrupt import (corrupt_sentence, level_sizes, load_table, plan_word, select_span, settings,
                               substitute)
from blnrepair.data import ROOT, load_config
from blnrepair.facts import fact_indices

SENTENCES = ROOT / "data" / "processed" / "sentences.jsonl"
CALIBRATION = ROOT / "reports" / "calibration.json"
pytestmark = pytest.mark.skipif(not (SENTENCES.exists() and CALIBRATION.exists()),
                                reason="needs sections D and E outputs")


@pytest.fixture(scope="module")
def setup():
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in SENTENCES.read_text(encoding="utf-8").splitlines()]
    cfg = settings(load_config(), calibration)
    return rows, load_table(calibration), cfg


@pytest.fixture(scope="module")
def records(setup):
    rows, table, cfg = setup
    return {(r["id"], lvl): corrupt_sentence(r, lvl, table, cfg) for r in rows for lvl in cfg["levels"]}


def damaged_levels(cfg):
    return [lvl for lvl in cfg["levels"] if lvl != "0"]


def test_same_seed_same_records_and_other_seed_differs(setup, records):
    rows, table, cfg = setup
    again = {(r["id"], lvl): corrupt_sentence(r, lvl, table, cfg) for r in rows for lvl in cfg["levels"]}
    assert again == records
    other = dict(cfg, seed=7)
    assert any(corrupt_sentence(r, "75", table, other)["corrupted_text"] != records[(r["id"], "75")]["corrupted_text"]
               for r in rows)


def test_nothing_outside_the_span_changes(records):
    for rec in records.values():
        if rec["k"] == 0:
            continue
        out = {p["idx"] for p in rec["ops_per_word"]}
        assert out == set(range(rec["span_start"], rec["span_end"]))
        kept = [t for i, t in enumerate(rec["gold_tokens"]) if i not in out]
        start = rec["span_start"]
        head = rec["gold_tokens"][:start]
        tail = rec["gold_tokens"][rec["span_end"]:]
        inside = [p["out"] for p in rec["ops_per_word"] if not p["dropped"]]
        assert rec["corrupted_text"].split() == head + inside + tail
        assert kept == head + tail


def test_exactly_k_words_damaged(records):
    for rec in records.values():
        assert len(rec["damaged_idx"]) == rec["k"] == len(rec["ops_per_word"])
        for p in rec["ops_per_word"]:
            assert p["dropped"] or p["out"] != p["gold"]


def test_spans_are_nested_and_each_word_is_corrupted_once(setup, records):
    rows, _, cfg = setup
    levels = damaged_levels(cfg)
    for r in rows:
        for small, large in zip(levels, levels[1:]):
            a, b = records[(r["id"], small)], records[(r["id"], large)]
            assert set(a["damaged_idx"]) <= set(b["damaged_idx"])
            forms = {p["idx"]: p for p in b["ops_per_word"]}
            for p in a["ops_per_word"]:
                assert forms[p["idx"]] == p


def test_anchor_is_a_fact_token_inside_the_span(setup, records):
    rows, _, cfg = setup
    for r in rows:
        facts = fact_indices(r["gold_text"].split())
        for lvl in damaged_levels(cfg):
            rec = records[(r["id"], lvl)]
            assert rec["anchor_idx"] in facts
            assert rec["span_start"] <= rec["anchor_idx"] < rec["span_end"]


def test_k_strictly_increases_for_every_length():
    assert list(level_sizes(25).values()) == [1, 3, 6, 13, 19]
    for n in range(20, 61):
        sizes = list(level_sizes(n).values())
        assert all(a < b for a, b in zip(sizes, sizes[1:])), n
        assert sizes[-1] <= n


def test_span_stays_inside_the_sentence():
    assert select_span(20, 0, 15) == (0, 15)
    assert select_span(20, 19, 15) == (5, 20)
    assert select_span(20, 10, 5) == (8, 13)


def test_level_0_has_no_damage(records):
    for (_, lvl), rec in records.items():
        if lvl == "0":
            assert rec["k"] == 0 and rec["damaged_idx"] == [] and rec["span_start"] is None
            assert rec["corrupted_text"] == rec["gold_text"]


def test_corrupted_text_is_never_empty(records):
    assert all(rec["corrupted_text"].strip() for rec in records.values())


def test_punctuation_is_kept(records):
    # A deleted first letter can expose inner punctuation ("I'll" -> "'ll"), so check that no
    # punctuation is changed, rather than what the word starts with after damage.
    punct = lambda t: [ch for ch in t if not ch.isalnum()]
    for rec in records.values():
        for p in rec["ops_per_word"]:
            if not p["dropped"]:
                g, o = p["gold"], p["out"]
                assert punct(o) == punct(g), p
                assert o.startswith(re.match(r"^\W*", g).group()) and o.endswith(re.search(r"\W*$", g).group()), p
                if "£" in g:
                    assert "£" in o


def test_at_most_two_deletions_and_letters_remain(records):
    for rec in records.values():
        for p in rec["ops_per_word"]:
            if not p["dropped"]:
                assert sum(op[0] == "del" for op in p["ops"]) <= 2
                assert any(ch.isalnum() for ch in p["out"])


def test_intensity_stays_near_the_drawn_range(records, setup):
    _, _, cfg = setup
    lo, hi = cfg["intensity_low"] - 0.10, cfg["intensity_high"] + 0.10  # rounding margin, not the drawn range itself
    for rec in records.values():
        for p in rec["ops_per_word"]:
            if not p["dropped"] and sum(ch.isalnum() for ch in p["gold"]) >= 5:
                assert lo <= p["intensity"] <= hi


def test_substitution_source(setup):
    import random
    rng = random.Random(0)
    toy = {"c": [("o", 5)]}
    assert substitute("c", rng, toy) == ("o", "table")
    assert substitute("d", rng, toy) == ("cl", "fallback_map")
    for digit in "2345679":
        new, source = substitute(digit, rng, toy)
        assert new.isdigit() or source == "fallback_map"
    for ch, cls in (("q", str.islower), ("Q", str.isupper), ("7", str.isdigit)):
        new, source = substitute(ch, rng, toy)
        assert source == "fallback_random" and cls(new) and new != ch


def test_token_without_letters_or_digits_is_dropped(setup):
    _, table, cfg = setup
    for token in ("&", "—", "--"):
        assert plan_word("x", 0, token, table, cfg)["dropped"]


def test_review_examples_cover_levels_and_bands(setup):
    from blnrepair.corrupt import review_examples, review_table
    rows, table, cfg = setup
    examples = review_examples(rows, table, cfg)
    assert len(examples) == 20 and len({r["id"] for r in examples}) == 20
    for level in damaged_levels(cfg):
        assert sum(r["severity"] == level for r in examples) == 4
        assert {r["band"] for r in examples if r["severity"] == level} == {"20-29", "30-44", "45-60"}
    assert examples == review_examples(rows, table, cfg)
    assert list(review_table(examples).columns)[:3] == ["level", "band", "k"]


def test_anchor_is_never_dropped_and_always_changes(setup, records):
    rows, table, cfg = setup
    for r in rows:
        for lvl in damaged_levels(cfg):
            rec = records[(r["id"], lvl)]
            plan = next(p for p in rec["ops_per_word"] if p["idx"] == rec["anchor_idx"])
            assert not plan["dropped"] and plan["out"] != plan["gold"], (r["id"], lvl, plan)


def test_allow_drop_false_ignores_the_roll_and_keeps_the_other_draws(setup):
    _, table, cfg = setup
    always = dict(cfg, drop_prob=1.0)
    assert plan_word("x", 3, "Jones", table, always)["dropped"]
    kept = plan_word("x", 3, "Jones", table, always, allow_drop=False)
    assert not kept["dropped"] and kept["out"] != "Jones"
    never = dict(cfg, drop_prob=0.0)
    assert plan_word("x", 3, "Jones", table, never) == plan_word("x", 3, "Jones", table, never, allow_drop=False)


def test_other_words_keep_the_drop_probability(records, setup):
    _, _, cfg = setup
    plans = [p for (_, lvl), rec in records.items() if lvl == "75"
             for p in rec["ops_per_word"] if p["idx"] != rec["anchor_idx"]]
    share = sum(p["dropped"] for p in plans) / len(plans)
    assert cfg["drop_prob"] - 0.02 < share < cfg["drop_prob"] + 0.04


def test_capital_uses_the_lowercase_entry(setup):
    import random
    rng = random.Random(0)
    toy = {"l": [("1", 5)]}
    assert substitute("L", rng, toy) == ("1", "table_casefold")
    assert substitute("l", rng, toy) == ("1", "table")
    toy = {"m": [("rn", 3)]}
    assert substitute("M", rng, toy) == ("RN", "table_casefold")
    assert substitute("Z", rng, toy)[1] == "fallback_map"


def test_casefold_never_maps_a_capital_to_itself(setup):
    import random
    rng = random.Random(0)
    toy = {"i": [("I", 3)]}
    for _ in range(20):
        new, source = substitute("I", rng, toy)
        assert new != "I" and source != "table_casefold"
