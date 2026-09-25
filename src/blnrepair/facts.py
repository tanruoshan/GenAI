"""Fact tokens: the words a later repair must get right (numbers, money, names)."""
import re
from collections import Counter

import pandas as pd

PUNCT = ".,;:!?\"'“”‘’()[]—–-"
NOT_FACT_WORDS = {"mr", "mrs", "dr", "st", "the", "miss", "sir", "rev", "messrs"}  # compared without case; "I" exactly


def is_fact(token, idx, prev=""):
    """A word with a digit or a pound sign, or a capitalized word that is not sentence-initial.
    A capitalized word is not a fact if it is I, Mr, Mrs, Dr, St, The, Miss, Sir, Rev or Messrs,
    or if it follows a token ending in a colon."""
    if any(c.isdigit() for c in token) or "£" in token:
        return True
    core = re.sub(r"^\W+|\W+$", "", token)  # letters only, no quotes or punctuation around
    if idx == 0 or not core[:1].isupper():
        return False
    return not (core == "I" or core.lower() in NOT_FACT_WORDS or prev.endswith(":"))


def fact_indices(tokens):
    """Positions of the fact tokens in a list of whitespace tokens."""
    return [i for i, tok in enumerate(tokens) if is_fact(tok, i, tokens[i - 1] if i else "")]


def mark_words(text, idx):
    """Show the words at positions idx in [brackets]."""
    idx = set(idx)
    return " ".join(f"[{tok}]" if i in idx else tok for i, tok in enumerate(text.split()))


def top_fact_tokens(pool, n=20):
    """The most frequent fact tokens in the pool, with their share of all fact tokens."""
    counts = Counter()
    for text, idx in zip(pool["text"], pool["fact_idx"]):
        words = text.split()
        counts.update(words[i].strip(PUNCT) for i in idx)
    total = sum(counts.values())
    top = pd.DataFrame(counts.most_common(n), columns=["fact token", "count"])
    top["share"] = (top["count"] / total).round(3)
    return top
