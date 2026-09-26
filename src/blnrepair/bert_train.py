"""Notebook 2, Part B: helpers for fine-tuning bert-base-cased on the training pool.

The training loop itself is in scripts/train_bert.py. Loss is computed only at the target word's
masks; the masks of pending span words carry no label (they stand for slots not filled yet).
"""
import hashlib
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from blnrepair.corrupt import seed_int
from blnrepair.freeze import sha256_bytes


def load_pool_checked(path, sha_path):
    """The training pool, after checking its hash against runs/bert_train_pool.sha256 (written by
    notebook 02). Stops on a mismatch, so the model is always trained on the pool the notebook shows."""
    data = Path(path).read_bytes()
    stored = Path(sha_path).read_text(encoding="utf-8").split()[0]
    digest = sha256_bytes(data)
    if digest != stored:
        raise RuntimeError(f"{Path(path).name} hash {digest} does not match the stored {stored}. Rerun notebook 02 first.")
    return pd.read_json(Path(path), lines=True), digest


def shuffled_batches(examples, batch_size, seed, epoch):
    """The examples in a seeded order for this epoch, cut into batches (the last one may be smaller)."""
    order = list(examples)
    random.Random(seed_int(seed, "bert_shuffle", epoch)).shuffle(order)
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def encode_batch(batch, tokenizer, max_length):
    """Token ids with padding, and labels that are -100 everywhere except at the target's masks.
    The target's masks are the first [MASK] tokens of the input: span words before the target are
    gold text, and the masks of pending words all come after it (build_masked_input)."""
    enc = tokenizer([e["masked_input"] for e in batch], padding=True, return_tensors="pt")
    assert enc["input_ids"].shape[1] <= max_length, f"an input is longer than max_length {max_length}"
    labels = torch.full_like(enc["input_ids"], -100)
    for i, e in enumerate(batch):
        target_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(e["target_word"]))
        mask_pos = (enc["input_ids"][i] == tokenizer.mask_token_id).nonzero().flatten()
        assert len(mask_pos) >= len(target_ids), f"{e['id']}: fewer masks than target pieces"
        labels[i, mask_pos[:len(target_ids)]] = torch.tensor(target_ids)
    enc["labels"] = labels
    return enc


def target_loss(model, enc, device, reduction="mean"):
    """Cross-entropy at the labelled positions only. The MLM head runs only on those positions,
    which saves memory: the full output would be batch x length x 28,996 vocabulary scores."""
    enc = {k: v.to(device) for k, v in enc.items()}
    labels = enc.pop("labels")
    hidden = model.bert(**enc).last_hidden_state
    keep = labels != -100
    logits = model.cls(hidden[keep])
    return F.cross_entropy(logits.float(), labels[keep], reduction=reduction), int(keep.sum())


@torch.no_grad()
def validation_loss(model, batches, tokenizer, max_length, device, amp=False):
    """Mean loss per target piece over all validation batches (eval mode, no dropout)."""
    model.eval()
    total, n = 0.0, 0
    for batch in batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss, count = target_loss(model, encode_batch(batch, tokenizer, max_length), device, reduction="sum")
        total, n = total + loss.item(), n + count
    model.train()
    return total / n


def sha256_file(path):
    """Hash of a (possibly large) file, read in chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
