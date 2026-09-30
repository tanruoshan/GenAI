from blnrepair.calibrate import align_words, block_type, char_blocks, core, length_flag, order_excerpts


def test_core_normalizes_quotes_and_strips_punctuation():
    assert core("“Queen’s,") == "Queen's"
    assert core("—") == ""
    assert core("...") == ""


def test_equal_length_replace_block_is_paired():
    a = align_words("the man was taken to court".split(), "the rnan was takeu to court".split())
    assert a["pairs"] == [("man", "rnan"), ("taken", "takeu")]
    assert a["equal"] == 4 and a["unpaired"] == 0


def test_unequal_replace_block_gives_no_pairs():
    a = align_words("he took the property away".split(), "he took the pro perty away".split())
    assert a["pairs"] == []
    assert a["unpaired"] == 1


def test_block_of_four_words_gives_no_pairs():
    gold = "at the station a b c d was found".split()
    ocr = "at the station w x y z was found".split()
    a = align_words(gold, ocr)
    assert a["pairs"] == [] and a["unpaired"] == 4


def test_isolated_word_drop_is_counted():
    a = align_words("he was then taken away".split(), "he was taken away".split())
    assert a["isolated_drops"] == 1


def test_excerpt_order_is_seeded_and_excludes_sample():
    ids = [f"{i:010d}" for i in range(50)]
    exclude = set(ids[:10])
    first = order_excerpts(ids, exclude, seed=42)
    assert first == order_excerpts(list(reversed(ids)), exclude, seed=42)
    assert not exclude & set(first) and len(first) == 40
    assert first != order_excerpts(ids, exclude, seed=7)


def test_char_blocks_substitution():
    assert char_blocks("them", "tbem") == [("h", "b")]


def test_char_blocks_merges_neighbouring_edits():
    assert char_blocks("modern", "rnodern") == [("m", "rn")]


def test_char_blocks_deletion():
    assert char_blocks("prisoner", "prisner") == [("o", "")]


def test_block_types():
    assert [block_type(*b) for b in [("h", "b"), ("m", "rn"), ("o", ""), ("", "e"), ("abc", "x")]] == [
        "substitution", "substitution", "deletion", "insertion", "other"]


def test_merged_flag():
    assert length_flag("on", "downon") == "merged"
    assert length_flag("on", "onx") is None


def test_split_flag():
    assert length_flag("superficial", "super") == "split"
    assert length_flag("them", "tbm") is None
