"""Section 0c / CHECKPOINT 1: how many WordPiece pieces the gold words of the damaged slots need.

BERT gets 1 to 3 masks per slot, so a gold word needing more than 3 pieces cannot be produced exactly.
Only the tokenizer is used here, no model.
"""
import re

import pandas as pd

from blnrepair.slots import slot_core


def parts_of(core, dropped):
    """The pieces of a slot filled separately if slots are split at visible internal punctuation
    (hyphens, apostrophes): the runs of letters and digits. A dropped slot has nothing visible, so
    it stays whole."""
    return [core] if dropped else [p for p in re.split(r"[^0-9A-Za-zÀ-ɏ]+", core) if p] or [core]


def ceiling_rows(rows, tokenize):
    """One record per damaged gold word (level 75 holds every damaged word of a sentence, and the
    smaller levels are nested inside it). whole = pieces of the whole gold core; split = the most
    pieces any part needs when the word is split at its internal punctuation."""
    out = []
    for r in rows:
        if r["severity"] != "75":
            continue
        facts = set(r["fact_idx_in_span"])
        for p in r["ops_per_word"]:
            core = slot_core(p["gold"])
            out.append({"id": r["id"], "split": r["split"], "gold_core": core, "dropped": p["dropped"],
                        "fact": p["idx"] in facts, "anchor": p["idx"] == r["anchor_idx"],
                        "whole": len(tokenize(core)),
                        "split_max": max(len(tokenize(x)) for x in parts_of(core, p["dropped"]))})
    return pd.DataFrame(out)


def ceiling_table(df, max_pieces=3):
    """Share of slots whose gold core needs more than max_pieces pieces, for all slots, fact slots
    and anchor slots (the only slot at level 1w), whole and split at internal punctuation."""
    groups = {"all slots": df, "fact slots": df[df["fact"]], "anchor slots": df[df["anchor"]]}
    return pd.DataFrame({name: {"n": len(g), "whole word": (g["whole"] > max_pieces).mean(),
                                "split at punctuation": (g["split_max"] > max_pieces).mean()}
                         for name, g in groups.items()}).T
