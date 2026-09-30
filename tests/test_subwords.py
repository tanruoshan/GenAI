"""Word-piece ceiling helper: parts of a slot and the ceiling table, with a fake tokenizer (no download)."""
from blnrepair.subwords import ceiling_rows, ceiling_table, parts_of


def test_parts_split_at_visible_punctuation_but_dropped_slots_stay_whole():
    assert parts_of("police-court", False) == ["police", "court"]
    assert parts_of("Scudder's-lane", False) == ["Scudder", "s", "lane"]
    assert parts_of("Kentish-town", True) == ["Kentish-town"]
    assert parts_of(",", False) == [","]  # punctuation-only word has no parts, keeps the whole token


def test_ceiling_counts_words_above_three_pieces():
    row = {"id": "x", "severity": "75", "split": "dev", "anchor_idx": 0, "fact_idx_in_span": [0, 2], "ops_per_word": [
        {"idx": 0, "gold": "Neckinger-town,", "dropped": False}, {"idx": 1, "gold": "of", "dropped": False},
        {"idx": 2, "gold": "Kentish-town", "dropped": True}]}
    fake = lambda s: list(s)[:: 3] if s else []  # pieces = every 3rd character, only to have some length
    df = ceiling_rows([row, {**row, "severity": "10"}], fake)
    assert len(df) == 3  # only level 75 is read
    table = ceiling_table(df, max_pieces=2)
    assert table.loc["all slots", "n"] == 3 and table.loc["fact slots", "n"] == 2 and table.loc["anchor slots", "n"] == 1
    assert table.loc["all slots", "split at punctuation"] <= table.loc["all slots", "whole word"]
