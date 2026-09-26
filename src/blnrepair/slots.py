"""Section 0a: the slot view shared by both repair tracks, and the splice that writes predictions back.

Model inputs are built from gold_tokens + ops_per_word, never from corrupted_text: a dropped word
disappears from corrupted_text, so slots would be lost.
"""
from collections import Counter

DROPPED = "⟨?⟩"


def split_punct(word):
    """(leading punctuation, core, trailing punctuation). Punctuation is any character that is not a
    letter or digit, the same rule the damage code used: it altered only letters and digits."""
    i = 0
    while i < len(word) and not word[i].isalnum():
        i += 1
    j = len(word)
    while j > i and not word[j - 1].isalnum():
        j -= 1
    return word[:i], word[i:j], word[j:]


def build_slots(row):
    """The shared model input for one row. words: every gold position, with None where a slot is.
    slots: one per damaged gold word, in order. A slot holds only what a model may see (the damaged
    form and its visible punctuation) plus flags for reporting. No gold text of damaged words."""
    plans = sorted(row["ops_per_word"], key=lambda p: p["idx"])
    facts = set(row["fact_idx_in_span"])
    slots = []
    for p in plans:
        lead, core, trail = ("", "", "") if p["dropped"] else split_punct(p["out"])
        slots.append({"idx": p["idx"], "out": p["out"], "dropped": p["dropped"], "lead": lead, "core": core,
                      "trail": trail, "anchor": p["idx"] == row["anchor_idx"], "fact": p["idx"] in facts})
    damaged = {s["idx"] for s in slots}
    words = [None if i in damaged else tok for i, tok in enumerate(row["gold_tokens"])]
    return {"id": row["id"], "severity": row["severity"], "split": row["split"], "k": row["k"],
            "words": words, "slots": slots}


def slot_view(view):
    """The sentence as text: context words unchanged, each slot as ⟨damaged form⟩, a dropped word as ⟨?⟩."""
    slots = iter(view["slots"])
    parts = []
    for w in view["words"]:
        if w is None:
            s = next(slots)
            parts.append(DROPPED if s["dropped"] else f"⟨{s['out']}⟩")
        else:
            parts.append(w)
    return " ".join(parts)


def _reframe(pred, lead, trail):
    """pred inside the gold word's edge punctuation. On a side where the gold word has punctuation,
    the prediction's own punctuation on that side is replaced by it (so "Harris," in a "," frame stays
    "Harris,"). On a side without punctuation, the prediction is left as it is."""
    p_lead, p_core, p_trail = split_punct(pred)
    if not p_core:
        return lead + pred + trail
    body = pred
    if lead:
        body = body[len(p_lead):]
    if trail:
        body = body[:len(body) - len(p_trail)]
    return lead + body + trail


def splice(row, preds):
    """The repaired sentence: one prediction per slot, context words never rewritten.
    Non-dropped slot: gold word's leading punctuation + prediction core + gold word's trailing punctuation.
    The frame comes from the gold word, not the damaged form: in 2 words a deleted letter left an inner
    apostrophe at the edge (I'll -> 'll), and a correct answer must still splice back correctly. For all
    other words the two frames are identical. Dropped slot: the prediction as returned, stripped.
    An empty result removes the word."""
    plans = sorted(row["ops_per_word"], key=lambda p: p["idx"])
    assert len(preds) == len(plans), f"{row['id']} {row['severity']}: {len(preds)} predictions for {len(plans)} slots"
    tokens = list(row["gold_tokens"])
    for p, pred in zip(plans, preds):
        pred = pred.strip()
        if p["dropped"]:
            tokens[p["idx"]] = pred
        else:
            lead, _, trail = split_punct(p["gold"])
            tokens[p["idx"]] = _reframe(pred, lead, trail)
    return " ".join(t for t in tokens if t)


def slot_core(word):
    """What scoring compares: the core without edge punctuation, or the whole stripped token when it
    is punctuation only (a lone "," is a gold word too)."""
    word = word.strip()
    return split_punct(word)[1] or word


def row_counts(rows):
    """Rows by split and severity, and rows with slots (level 0 has none and is skipped by both tracks)."""
    table = Counter((r["split"], r["severity"]) for r in rows)
    with_slots = Counter(r["split"] for r in rows if r["k"])
    return table, with_slots
