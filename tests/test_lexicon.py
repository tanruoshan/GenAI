"""The dictionary lookup baseline (notebook 2b)."""
import json
from collections import Counter

import pytest

from blnrepair.data import ROOT, load_config, load_gold
from blnrepair.lexicon import LexiconLookup, build_lexicon, lexicon_version, load_lexicon_config
from blnrepair.slots import build_slots

RAW = ROOT / "data" / "raw" / "BLN600"
FROZEN = ROOT / "data" / "processed" / "corrupted_v2.jsonl"


def test_lexicon_keeps_cores_and_skips_excluded_excerpts():
    gold = {"a": ["Mr. Hurd, of Market-place, said £2."], "b": ["Secret words only here."]}
    counts = build_lexicon(gold, exclude_docs={"b"})
    assert counts == Counter({"Mr": 1, "Hurd": 1, "of": 1, "Market-place": 1, "said": 1, "2": 1})


def test_nearest_takes_the_closest_then_the_most_frequent_entry():
    lookup = LexiconLookup(Counter({"Hurd": 2, "Hard": 5, "Harold": 9}))
    assert lookup.nearest("Hnrd") == ("Hard", 1, 2)  # Hurd and Hard are both 1 edit away; Hard is more frequent
    assert lookup.nearest("Harol") == ("Harold", 1, 1)


def test_one_answer_per_slot_and_empty_for_a_dropped_slot():
    view = {"slots": [{"dropped": False, "core": "Hnrd"}, {"dropped": True, "core": ""}]}
    out = LexiconLookup(Counter({"Hurd": 3}))(view)
    assert out["pred_words"] == ["Hurd", ""]
    assert json.loads(out["raw_output"]) == [["Hurd", 1, 1, 3], None]


def test_version_name_and_frozen_flag():
    cfg = load_lexicon_config()
    assert isinstance(cfg["frozen"], bool)
    assert lexicon_version({"version": "v1"}, "dev") == "v1-dev"


@pytest.mark.skipif(not (RAW.exists() and FROZEN.exists()), reason="BLN600 or frozen data missing")
def test_real_lexicon_holds_no_sample_excerpt():
    from blnrepair.freeze import load_frozen
    rows = load_frozen("v2")
    sample_docs = {r["doc_id"] for r in rows}
    gold = load_gold(RAW)
    assert len(sample_docs) == 157 and len(set(gold) - sample_docs) == 443
    counts = build_lexicon(gold, sample_docs)
    assert counts == build_lexicon({d: p for d, p in gold.items() if d not in sample_docs}, set())
    only_in_sample = set(build_lexicon({d: gold[d] for d in sample_docs}, set())) - set(counts)
    assert only_in_sample  # the exclusion bites: words seen only in sample excerpts are not in the list
    row = next(r for r in rows if r["split"] == "dev" and r["severity"] == "75")
    out = LexiconLookup(counts)(build_slots(row))
    assert len(out["pred_words"]) == row["k"]
