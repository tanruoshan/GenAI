import pytest

from blnrepair.data import ROOT, load_config, load_gold, load_metadata, normalize_gold

RAW = ROOT / load_config()["paths"]["raw_dir"]


def test_wrapped_lines_are_joined():
    assert normalize_gold("one two\nthree  four\n") == ["one two three four"]


def test_blank_line_starts_new_paragraph():
    assert normalize_gold("HEADLINE.\n\nBody text\ncontinues.") == ["HEADLINE.", "Body text continues."]


def test_hyphen_at_line_end_is_joined_without_space():
    assert normalize_gold("some-\nthing and Cross-\nexamination") == ["some-thing and Cross-examination"]


def test_no_empty_paragraphs():
    assert normalize_gold("\n\n  \n\nText.\n\n\n") == ["Text."]


@pytest.mark.skipif(not (RAW / "Ground Truth").exists(), reason="BLN600 not unpacked")
def test_full_corpus_loads():
    gold = load_gold(RAW)
    meta = load_metadata(RAW)
    assert len(gold) == 600
    assert set(gold) == set(meta.index)
    assert all(paras for paras in gold.values())
