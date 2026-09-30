"""BERT fine-tuning helpers: pool hash check, seeded batches, and labels only at the target's masks."""
import pandas as pd
import pytest

from blnrepair.bert_data import build_masked_input, freeze_pool
from blnrepair.bert_train import encode_batch, load_pool_checked, shuffled_batches


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("bert-base-cased", local_files_only=True)
    except OSError:
        pytest.skip("bert-base-cased tokenizer not in the local cache")


def test_load_pool_checked_stops_on_a_changed_pool(tmp_path):
    pool = pd.DataFrame([{"id": "a-000", "text": "one two", "usage": "train"}])
    path, sha = tmp_path / "pool.jsonl", tmp_path / "pool.sha256"
    freeze_pool(pool, path, sha)
    loaded, _ = load_pool_checked(path, sha)
    assert loaded.to_dict("records") == pool.to_dict("records")
    path.write_bytes(path.read_bytes().replace(b"two", b"three"))
    with pytest.raises(RuntimeError):
        load_pool_checked(path, sha)


def test_shuffled_batches_are_seeded_and_cover_every_example():
    examples = [{"n": i} for i in range(37)]
    a, b = shuffled_batches(examples, 16, 42, 0), shuffled_batches(examples, 16, 42, 0)
    assert a == b and [len(x) for x in a] == [16, 16, 5]
    assert sorted(e["n"] for batch in a for e in batch) == list(range(37))
    assert shuffled_batches(examples, 16, 42, 1) != a


def test_labels_only_at_the_target_masks(tokenizer):
    tokens = "He lived at 37 Lowndes street, Belgravia, and was charged by Inspector Smith.".split()
    span = {"start": 4, "end": 9}  # Lowndes street, Belgravia, and was
    examples = []
    for target_pos in (4, 6, 8):
        text, target = build_masked_input(tokens, span, target_pos, tokenizer.tokenize)
        examples.append({"id": "x", "masked_input": text, "target_word": target})
    enc = encode_batch(examples, tokenizer, max_length=192)
    for i, e in enumerate(examples):
        target_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(e["target_word"]))
        labelled = enc["labels"][i][enc["labels"][i] != -100].tolist()
        assert labelled == target_ids
        at = (enc["labels"][i] != -100).nonzero().flatten()
        assert (enc["input_ids"][i][at] == tokenizer.mask_token_id).all()
    # Belgravia (target_pos 6) has later span words pending: they are masked but carry no label
    assert (enc["input_ids"][1] == tokenizer.mask_token_id).sum() > len(tokenizer.tokenize("Belgravia"))


def test_encode_batch_refuses_inputs_longer_than_max_length(tokenizer):
    e = {"id": "x", "masked_input": "a " * 50 + "[MASK]", "target_word": "a"}
    with pytest.raises(AssertionError):
        encode_batch([e], tokenizer, max_length=20)
