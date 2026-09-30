"""Synthetic OCR-like damage for one sentence at one level (notebook 1)."""
import hashlib
import random
import string
from collections import Counter

import pandas as pd
from rapidfuzz.distance import Levenshtein

from blnrepair.facts import fact_indices

SOURCES = ("table", "table_casefold", "fallback_map", "fallback_random")
FALLBACK_MAP = {"O": "0", "0": "O", "l": "1", "1": "l", "I": "l", "S": "5", "5": "S", "B": "8", "8": "B",
                "G": "6", "Z": "2", "m": "rn", "d": "cl", "e": "c", "c": "e", "o": "e", "a": "o",
                "u": "n", "n": "u", "h": "b", "r": "t", "t": "f"}


def seed_int(seed, purpose, sent_id, idx=""):
    """A stable integer seed from sha256 (never Python's hash(), which changes between runs)."""
    h = hashlib.sha256(f"{seed}:{purpose}:{sent_id}:{idx}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def settings(config, calibration):
    """Corruption settings: the config block, the seed, and p_deletion from the calibration."""
    mix = calibration["op_mix"]
    c = config["corruption"]
    return {"seed": config["seed"], "drop_prob": c["drop_prob"], "max_deletions": c["max_deletions"],
            "intensity_low": c["intensity_low"], "intensity_high": c["intensity_high"],
            "levels": ["0"] + [str(level) for level in c["levels"]],
            "p_deletion": round(mix["deletion"] / (mix["substitution"] + mix["deletion"]), 2)}


def load_table(calibration):
    """{gold character: [(ocr part, count), ...]} from the substitution entries of the calibration."""
    table = {}
    for entry in calibration["confusions"]:
        if entry["ocr"]:
            assert len(entry["gold"]) == 1, f"gold part longer than one character: {entry}"
            table.setdefault(entry["gold"], []).append((entry["ocr"], entry["count"]))
    return table


def level_sizes(n):
    """Damaged words per level: 1, then 10/25/50/75% of n with integer half-up rounding."""
    return {"1w": 1, **{str(pct): (pct * n + 50) // 100 for pct in (10, 25, 50, 75)}}


def choose_anchor(tokens, sent_id, seed):
    """One anchor per sentence, drawn among its fact tokens."""
    return random.Random(seed_int(seed, "anchor", sent_id)).choice(fact_indices(tokens))


def select_span(n, anchor, k):
    """The damaged span [start, start + k) around the anchor, kept inside the sentence.
    For a fixed anchor, a smaller span always lies inside a larger one."""
    start = min(max(anchor - k // 2, 0), n - k)
    return start, start + k


def substitute(c, rng, table):
    """OCR-like replacement for character c. Order: (a) the calibrated table; (b) for a capital
    without an entry, the entry of its lowercase letter, written as a capital ("table_casefold");
    (c) a short look-alike map; (d) a random different character of the same class."""
    if c in table:
        parts, counts = zip(*table[c])
        return rng.choices(parts, weights=counts)[0], "table"
    if c.isupper():
        entries = [(o.upper(), n) for o, n in table.get(c.lower(), []) if o.upper() != c]  # "i" -> "I" is no change for "I"
        if entries:
            parts, counts = zip(*entries)
            return rng.choices(parts, weights=counts)[0], "table_casefold"
    if c in FALLBACK_MAP:
        return FALLBACK_MAP[c], "fallback_map"
    pool = string.ascii_lowercase if c.islower() else string.ascii_uppercase if c.isupper() else string.digits
    return rng.choice([x for x in pool if x != c]), "fallback_random"


def plan_word(sent_id, idx, token, table, cfg, allow_drop=True):
    """The damage plan for one word. It depends only on its arguments, so every level that
    damages this word gets the same result. Steps, drawn from one seeded Random in this order:
    1. no letters or digits: drop; 2. drop the word with drop_prob (the roll is always drawn, but
    allow_drop=False, used for the anchor, ignores it); 3. draw the intensity;
    4. choose the characters to alter (punctuation is never touched); 5. delete or substitute
    each, at most max_deletions deletions; 6. substitute from the table or a fallback."""
    rng = random.Random(seed_int(cfg["seed"], "word", sent_id, idx))
    plan = {"idx": idx, "gold": token, "out": "", "ops": [], "intensity": None, "dropped": True}
    alnum = [i for i, ch in enumerate(token) if ch.isalnum()]
    drop_roll = rng.random() < cfg["drop_prob"]
    if not alnum or (drop_roll and allow_drop):
        return plan
    n_alnum = len(alnum)
    p = rng.uniform(cfg["intensity_low"], cfg["intensity_high"])
    m = min(n_alnum, max(1, int(p * n_alnum + 0.5)))
    positions = sorted(rng.sample(alnum, m))
    kinds = ["del" if rng.random() < cfg["p_deletion"] else "sub" for _ in positions]
    cap = min(cfg["max_deletions"], n_alnum - 1, m)
    kinds = [k if k == "sub" or kinds[:j].count("del") < cap else "sub" for j, k in enumerate(kinds)]
    replace = {}
    for pos, kind in zip(positions, kinds):
        c = token[pos]
        if kind == "del":
            plan["ops"].append(["del", c])
            replace[pos] = ""
        else:
            new, source = substitute(c, rng, table)
            plan["ops"].append(["sub", c, new, source])
            replace[pos] = new
    out = "".join(replace.get(i, ch) for i, ch in enumerate(token))
    if out == token:
        raise ValueError(f"damage left the word unchanged: {sent_id} {idx} {token!r}")
    plan.update(out=out, intensity=m / n_alnum, dropped=False)
    return plan


def corrupt_sentence(row, level, table, cfg):
    """One output record for a sentence at a level ("0", "1w", "10", "25", "50", "75")."""
    tokens = row["gold_text"].split()
    n = len(tokens)
    anchor = choose_anchor(tokens, row["id"], cfg["seed"])
    k = 0 if level == "0" else level_sizes(n)[level]
    start, end = select_span(n, anchor, k) if k else (None, None)
    damaged = list(range(start, end)) if k else []
    plans = [plan_word(row["id"], i, tokens[i], table, cfg, allow_drop=(i != anchor)) for i in damaged]
    assert not k or not next(p for p in plans if p["idx"] == anchor)["dropped"], f"anchor dropped: {row['id']}"
    words = tokens[:start] + [p["out"] for p in plans if not p["dropped"]] + tokens[end:] if k else tokens
    return {"id": row["id"], "doc_id": row["doc_id"], "gold_text": row["gold_text"], "gold_tokens": tokens,
            "band": row["band"], "severity": level, "k": k, "frac": k / n, "anchor_idx": anchor,
            "span_start": start, "span_end": end, "damaged_idx": damaged, "ops_per_word": plans,
            "corrupted_text": " ".join(words), "mask_positions": damaged,
            "fact_idx_in_span": [i for i in fact_indices(tokens) if k and start <= i < end],
            "seed": cfg["seed"], "split": row["split"]}


def mark_damage(record):
    """The damaged sentence with each damaged word in [[...]] and a dropped word as [[]]."""
    plans = {p["idx"]: p for p in record["ops_per_word"]}
    return " ".join(f"[[{plans[i]['out']}]]" if i in plans else tok for i, tok in enumerate(record["gold_tokens"]))


def corruption_stats(records):
    """What the damage looks like over the given records (use level 75: it holds every damaged word)."""
    plans = [p for r in records for p in r["ops_per_word"]]
    ops = [op for p in plans for op in p["ops"]]
    subs = Counter(op[3] for op in ops if op[0] == "sub")
    alnum = lambda w: sum(ch.isalnum() for ch in w)
    kept = [p for p in plans if not p["dropped"]]
    bins = {"1-3": (1, 3), "4-6": (4, 6), "7+": (7, 10**6)}
    return {"damaged_words": len(plans),
            "share_dropped": sum(p["dropped"] for p in plans) / len(plans),
            "deletion_share_of_altered_chars": sum(op[0] == "del" for op in ops) / len(ops),
            "substitution_source_share": {s: subs[s] / sum(subs.values()) for s in SOURCES},
            "mean_intensity_by_length": {b: sum(p["intensity"] for p in kept if lo <= alnum(p["gold"]) <= hi)
                                         / max(1, sum(lo <= alnum(p["gold"]) <= hi for p in kept))
                                         for b, (lo, hi) in bins.items()},
            "drop_share": {"anchor": _share(p["dropped"] for r in records for p in r["ops_per_word"] if p["idx"] == r["anchor_idx"]),
                           "other_words": _share(p["dropped"] for r in records for p in r["ops_per_word"] if p["idx"] != r["anchor_idx"])},
            "digit_words": sum(any(ch.isdigit() for ch in p["gold"]) for p in plans),
            "no_alnum_tokens": sum(alnum(p["gold"]) == 0 for p in plans)}


def review_examples(rows, table, cfg, per_level=4):
    """About 20 (sentence, level) examples for the review: per_level sentences at each damaged level,
    with the three length bands rotated so every level shows every band. Each sentence is used once;
    the order is seeded (sha256), so the same examples come back every time."""
    bands = sorted({r["band"] for r in rows})
    order = sorted(rows, key=lambda r: seed_int(cfg["seed"], "review", r["id"]))
    used, records = set(), []
    for li, level in enumerate(cfg["levels"][1:]):
        for slot in range(per_level):
            band = bands[(li + slot) % len(bands)]
            row = next(r for r in order if r["band"] == band and r["id"] not in used)
            used.add(row["id"])
            records.append(corrupt_sentence(row, level, table, cfg))
    return records


def review_table(records):
    """One row per example: level, band, the fact words inside the damaged span, the gold
    sentence, and the damaged sentence with damaged words in [[...]]."""
    facts = lambda r: ", ".join(r["gold_tokens"][i] for i in r["fact_idx_in_span"])
    return pd.DataFrame({"level": [r["severity"] for r in records], "band": [r["band"] for r in records],
                         "k": [r["k"] for r in records], "fact words in span": [facts(r) for r in records],
                         "gold sentence": [r["gold_text"] for r in records],
                         "damaged (marked)": [mark_damage(r) for r in records]}).set_index(pd.Index([r["id"] for r in records], name="id"))


def _share(flags):
    flags = list(flags)
    return sum(flags) / len(flags)


def cer_by_level(rows, table, cfg):
    """Sentence-level character error rate of the damaged text against the gold text, per level:
    Levenshtein distance / gold length, summarized over the sentences."""
    out = {}
    for level in cfg["levels"]:
        cer = pd.Series([Levenshtein.distance(r["gold_text"], rec["corrupted_text"]) / len(r["gold_text"])
                         for r in rows for rec in [corrupt_sentence(r, level, table, cfg)]])
        out[level] = {"mean": cer.mean(), "median": cer.median(), "q1": cer.quantile(0.25), "q3": cer.quantile(0.75)}
    return out
