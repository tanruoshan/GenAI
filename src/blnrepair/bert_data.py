"""
Training sentences come only from excerpts outside the 157 sample excerpts (test and dev), so the
fine-tuned model never sees a sample sentence. The 29 calibration excerpts stay in the pool: they
were only used to measure OCR/gold alignment, never sampled as sentences. BERT is trained purely
as a masked language model on clean gold text - it never sees garbled text in training: garbled
letters enter only through the reranking score at inference (same input for all methods).
"""
import json
import random
from collections import Counter
from pathlib import Path

from blnrepair.corrupt import level_sizes, seed_int, select_span
from blnrepair.data import ROOT, build_sentences, load_gold
from blnrepair.freeze import sha256_bytes
from blnrepair.slots import split_punct

LEVELS_FOR_TRAINING = ["1w", "10", "25", "50", "75"]


def assert_no_leakage(pool, sample):
    """Zero shared doc_id and zero exact sentence-text overlap between the training pool and the
    sampled sentences (test and dev)."""
    shared_docs = set(pool["doc_id"]) & {r["doc_id"] for r in sample}
    assert not shared_docs, f"training pool shares excerpts with the sample: {shared_docs}"
    shared_text = set(pool["text"]) & {r["gold_text"] for r in sample}
    assert not shared_text, f"training pool repeats {len(shared_text)} sampled sentence(s) verbatim"


def excerpt_split(doc_ids, seed, val_frac=0.10):
    """Assign each training excerpt to 'train' or 'val', seeded, holding out about val_frac of
    excerpts for validation loss. Split by excerpt, not sentence, so no excerpt has sentences on
    both sides. Validation is only for checkpoint choice during fine-tuning; it is never the dev
    split used later for prompt or lambda tuning."""
    ordered = sorted(doc_ids, key=lambda d: seed_int(seed, "bert_val_split", d))
    n_val = max(1, round(len(ordered) * val_frac))
    val_ids = set(ordered[:n_val])
    return {d: ("val" if d in val_ids else "train") for d in doc_ids}


def pool_summary(pool):
    """Sentence, excerpt and word counts by usage (train, val), plus a total row."""
    counts = pool.groupby("usage").agg(sentences=("id", "count"), excerpts=("doc_id", "nunique"), words=("n_words", "sum"))
    counts.loc["total"] = counts.sum()
    return counts


def build_pool(cfg, raw_dir=None, processed_dir=None, min_facts=1, val_frac=0.10):
    """The full training pool: excerpts outside the sample, sentences of 20 to 60 words with at
    least min_facts fact tokens (default 1; the eval sample uses 2 with a fallback to 1 -
    training does not need the same fact density, and a lower bar means more training data). No
    per-excerpt cap: the eval sample caps at 3 sentences per excerpt to spread the sample across
    excerpts, but that reason does not hold for training data, where more is better. Returns the
    pool (with a 'usage' column, 'train' or 'val'), the sample rows, and a summary table."""
    raw_dir = Path(raw_dir) if raw_dir else ROOT / cfg["paths"]["raw_dir"]
    processed_dir = Path(processed_dir) if processed_dir else ROOT / cfg["paths"]["processed_dir"]
    with (processed_dir / "sentences.jsonl").open(encoding="utf-8") as f:
        sample = [json.loads(line) for line in f]
    gold_all = load_gold(raw_dir)
    train_ids = sorted(set(gold_all) - {r["doc_id"] for r in sample})
    sentences = build_sentences({d: gold_all[d] for d in train_ids}, cfg["sentence_pool"]["length_bands"])
    pool = sentences[sentences["band"].notna() & (sentences["n_facts"] >= min_facts)].reset_index(drop=True)
    assert_no_leakage(pool, sample)
    usage = excerpt_split(sorted(pool["doc_id"].unique()), cfg["seed"], val_frac)
    pool = pool.assign(usage=pool["doc_id"].map(usage))
    return pool, sample, pool_summary(pool)


def pool_bytes(pool):
    """The pool as JSON lines, UTF-8, LF line endings: the same bytes on Windows and on Colab."""
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool.to_dict("records")).encode("utf-8")


def save_pool(pool, path):
    """Write the training pool as JSON lines (UTF-8), one sentence per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(pool_bytes(pool))


def freeze_pool(pool, path, sha_path):
    """Write the pool and its sha256 (read by scripts/train_bert.py before training). If the file
    already exists, the rebuilt pool must have the same bytes; otherwise stop, because a model may
    already have been trained on the old pool. Returns ("written" or "unchanged", sha256)."""
    data = pool_bytes(pool)
    digest = sha256_bytes(data)
    path, sha_path = Path(path), Path(sha_path)
    if path.exists():
        if sha256_bytes(path.read_bytes()) != digest:
            raise RuntimeError(f"the rebuilt pool differs from {path.name}. Stop and check before overwriting.")
        status = "unchanged"
    else:
        save_pool(pool, path)
        status = "written"
    sha_path.parent.mkdir(parents=True, exist_ok=True)
    sha_path.write_bytes(f"{digest}  {path.name}\n".encode("utf-8"))
    return status, digest


def draw_training_span(sent_id, n_words, fact_idx, seed, epoch):
    """One seeded span draw for this sentence at this epoch: a random evaluation level and its k
    (the evaluation's clamp rule: corrupt.level_sizes and corrupt.select_span), anchored on a random
    fact token. epoch enters the seed, so the draw differs every epoch ("new draw every epoch")."""
    level = random.Random(seed_int(seed, "bert_level", sent_id, epoch)).choice(LEVELS_FOR_TRAINING)
    k = level_sizes(n_words)[level]
    anchor = random.Random(seed_int(seed, "bert_anchor", sent_id, epoch)).choice(fact_idx)
    start, end = select_span(n_words, anchor, k)
    return {"level": level, "k": k, "anchor": anchor, "start": start, "end": end}


def build_masked_input(tokens, span, target_pos, tokenize, mask_token="[MASK]"):
    """One training example in the inference-matched format. Span words
    before target_pos are gold text (already "filled", matching the left-to-right fill order).
    The target word is masked at its true WordPiece piece count; loss is computed only there.
    Span words after target_pos are one [MASK] each, whatever their real piece count - a pending
    slot shows one mask until its turn, exactly as at inference. Context outside the span, and
    every word's leading and trailing punctuation, is never masked (punctuation is never damaged).
    A gold word that is punctuation only (its core is empty) is masked as the whole token,
    matching the scoring rule that compares the whole token in that case.

    A word with an internal hyphen or apostrophe is masked here as one target with all its pieces
    at once. This differs from inference, where such a word is split into punctuation
    sub-slots and filled one part at a time - a deliberate simplification: it affects under 1% of
    slots (notebook 2, word-piece ceiling), and training does not share inference's 3-mask cap that the split exists for.

    Returns (the sentence as space-joined text with masks, the target's gold core)."""
    start, end = span["start"], span["end"]
    words, target_core = [], None
    for i, tok in enumerate(tokens):
        if not (start <= i < end) or i < target_pos:
            words.append(tok)
        else:
            lead, core, trail = split_punct(tok)
            if i == target_pos:
                target_core = core or tok
                n_pieces = max(1, len(tokenize(target_core)))
                words.append(lead + " ".join([mask_token] * n_pieces) + trail)
            else:
                words.append(lead + mask_token + trail)
    return " ".join(words), target_core


def epoch_examples(pool, seed, epoch, tokenize, mask_token="[MASK]"):
    """Every training example for one epoch: one per span word, for every pool sentence. Rows
    with usage 'train' get a fresh span draw each epoch ("new draw every epoch"), so the model
    does not memorize one fixed masking. Rows with usage 'val' always use a fixed draw (epoch
    "val"), the same at every call, so validation loss is comparable from one epoch to the next
    and early stopping is not chasing a moving target. The caller (scripts/train_bert.py) filters
    by usage: 'train' rows are the training batches, 'val' rows measure validation loss for the
    checkpoint choice and the early-stopping patience count. tokenize is any word -> list[str]
    callable (e.g. a Hugging Face tokenizer's .tokenize method); this module never imports
    transformers itself, the same pattern subwords.py uses."""
    for row in pool.itertuples():
        tokens = row.text.split()
        draw_epoch = epoch if row.usage == "train" else "val"
        span = draw_training_span(row.id, row.n_words, row.fact_idx, seed, draw_epoch)
        for target_pos in range(span["start"], span["end"]):
            text, target = build_masked_input(tokens, span, target_pos, tokenize, mask_token)
            yield {"id": row.id, "usage": row.usage, "level": span["level"], "target_pos": target_pos,
                   "target_word": target, "masked_input": text}


def example_stats(examples, tokenizer, max_length=192):
    """Checks on one epoch of examples with the real tokenizer: the encoded length (no truncation
    at max_length), the mask count (target pieces + one per pending span word), the target piece
    counts, and the share of training examples per level. tokenizer is a Hugging Face tokenizer."""
    lengths, mismatches, pieces, levels, usage = [], 0, Counter(), Counter(), Counter()
    for e in examples:
        ids = tokenizer(e["masked_input"])["input_ids"]
        n_target = len(tokenizer.tokenize(e["target_word"]))
        n_pending = e["masked_input"].count(tokenizer.mask_token) - n_target
        mismatches += ids.count(tokenizer.mask_token_id) != n_target + n_pending
        lengths.append(len(ids))
        pieces[min(n_target, 4)] += 1
        usage[e["usage"]] += 1
        if e["usage"] == "train":
            levels[e["level"]] += 1
    return {"examples": dict(usage), "max_length": max(lengths), "over_max_length": sum(n > max_length for n in lengths),
            "mask_mismatches": mismatches, "target_pieces (4 = 4 or more)": dict(sorted(pieces.items())),
            "train_share_by_level": {lv: round(levels[lv] / usage["train"], 3) for lv in LEVELS_FOR_TRAINING}}
