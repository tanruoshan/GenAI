"""Calibration: real OCR errors (gold to OCR) as context for the synthetic damage (notebook 0)."""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from blnrepair.data import normalize_gold

TYPOGRAPHIC = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def sample_doc_ids(sentences_path):
    """Excerpt ids used by the test and dev sample."""
    with Path(sentences_path).open(encoding="utf-8") as f:
        return {json.loads(line)["doc_id"] for line in f}


def order_excerpts(all_ids, exclude, seed):
    """Ids not in exclude, in a seeded order: sha256 of (seed, "calibration", id)."""
    rest = [d for d in all_ids if d not in set(exclude)]
    return sorted(rest, key=lambda d: hashlib.sha256(f"{seed}:calibration:{d}".encode()).hexdigest())


def load_pair(raw_dir, doc_id):
    """Gold and OCR text of one excerpt, cleaned the same way as for the sentence pool (notebook 0), as one line each."""
    read = lambda folder: (Path(raw_dir) / folder / f"{doc_id}.txt").read_text(encoding="utf-8")
    return " ".join(normalize_gold(read("Ground Truth"))), " ".join(normalize_gold(read("OCR Text")))


def excerpt_cer(gold, ocr):
    """Whole-excerpt character error rate."""
    return Levenshtein.distance(gold, ocr) / len(gold)


def core(token):
    """The alphanumeric core of a token: NFC, typographic quotes and dashes mapped, outer punctuation stripped."""
    token = unicodedata.normalize("NFC", token).translate(TYPOGRAPHIC)
    return re.sub(r"^\W+|\W+$", "", token)


def cores(text):
    return [c for c in (core(t) for t in text.split()) if c]


def align_words(gold, ocr, max_block=3):
    """Align gold and OCR word cores with difflib. Equal-length replace blocks of at most max_block
    words are paired by position; every other non-equal block leaves its gold words unpaired."""
    ops = SequenceMatcher(None, gold, ocr, autojunk=False).get_opcodes()
    equal, pairs, unpaired, drops = [], [], 0, 0
    for n, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal":
            equal += gold[i1:i2]
        elif tag == "replace" and i2 - i1 == j2 - j1 <= max_block:
            pairs += list(zip(gold[i1:i2], ocr[j1:j2]))
        else:
            unpaired += i2 - i1
            isolated = 0 < n < len(ops) - 1 and ops[n - 1][0] == ops[n + 1][0] == "equal"
            drops += tag == "delete" and i2 - i1 <= 2 and isolated
    return {"gold_words": len(gold), "equal": len(equal), "paired": len(pairs), "unpaired": unpaired,
            "equal_words": equal, "pairs": pairs, "isolated_drops": drops}


def align_excerpts(raw_dir, ordered_ids, n=30, min_keep=20, min_coverage=0.60):
    """Align the first n excerpts; drop those with coverage below min_coverage (misordered columns,
    missing text); take further excerpts from the order until min_keep are kept.
    Returns (table with one row per excerpt looked at, {doc_id: alignment} for the kept ones)."""
    rows, kept = [], {}
    for pos, doc_id in enumerate(ordered_ids):
        if pos >= n and len(kept) >= min_keep:
            break
        gold, ocr = load_pair(raw_dir, doc_id)
        a = align_words(cores(gold), cores(ocr))
        coverage = (a["equal"] + a["paired"]) / a["gold_words"]
        keep = coverage >= min_coverage
        rows.append({"doc_id": doc_id, "cer": round(excerpt_cer(gold, ocr), 3), "gold_words": a["gold_words"],
                     "equal": a["equal"], "paired": a["paired"], "unpaired": a["unpaired"],
                     "coverage": round(coverage, 3), "used": keep})
        if keep:
            kept[doc_id] = a
    return pd.DataFrame(rows).set_index("doc_id"), kept


def coverage_totals(table):
    """Word totals over the excerpts used, and the word error rate: paired / (equal + paired)."""
    tot = table[table["used"]][["gold_words", "equal", "paired", "unpaired"]].sum()
    return tot, tot["paired"] / (tot["equal"] + tot["paired"])


def random_pairs(kept, n, seed):
    """n aligned word pairs, drawn with a seeded shuffle, shown as 'gold -> ocr'."""
    pairs = [p for a in kept.values() for p in a["pairs"]]
    pairs.sort(key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
    return pd.DataFrame(pairs[:n], columns=["gold", "ocr"])


def char_blocks(g, o):
    """Character edits from gold word g to OCR word o. Neighbouring edits are merged into one
    block, so "m" -> "rn" is one confusion, not a replace plus an insert."""
    blocks, cur = [], None
    for op in Levenshtein.opcodes(g, o):
        if op.tag == "equal":
            if cur:
                blocks.append(cur)
                cur = None
        elif cur is None:
            cur = [op.src_start, op.src_end, op.dest_start, op.dest_end]
        else:
            cur[1], cur[3] = op.src_end, op.dest_end
    if cur:
        blocks.append(cur)
    return [(g[a:b], o[c:d]) for a, b, c, d in blocks]


def block_type(gold_part, ocr_part):
    if len(gold_part) > 2 or len(ocr_part) > 2:
        return "other"
    if gold_part and ocr_part:
        return "substitution"
    return "deletion" if gold_part else "insertion"


def all_pairs(kept):
    return [p for a in kept.values() for p in a["pairs"]]


def length_flag(g, o, gap=3):
    """'merged' if the OCR word is at least `gap` characters longer (a neighbour glued on),
    'split' if it is at least `gap` characters shorter (part of the word lost), else None."""
    if len(o) >= len(g) + gap:
        return "merged"
    if len(g) >= len(o) + gap:
        return "split"
    return None


def tier_a(kept, floor=50, length_flags=True):
    """Clean pairs: gold and OCR word share at least `floor` percent (rapidfuzz ratio) and,
    with length_flags, are neither merged nor split."""
    return [(g, o) for g, o in all_pairs(kept)
            if fuzz.ratio(g, o) >= floor and not (length_flags and length_flag(g, o))]


def flagged_pairs(kept, floor=50):
    """The pairs that the length flags move out of Tier A, by flag."""
    out = {"merged": [], "split": []}
    for g, o in all_pairs(kept):
        if fuzz.ratio(g, o) >= floor and length_flag(g, o):
            out[length_flag(g, o)].append((g, o))
    return out


def char_tables(kept, pairs_a, min_count=3):
    """Confusion table, operation mix and character deletion rate from the Tier A pairs.
    A confusion's rate is its count divided by how often its gold part occurs in all aligned
    gold words (equal words plus Tier A gold words). OCR part "" means a deletion."""
    aligned_gold = " ".join([w for a in kept.values() for w in a["equal_words"]] + [g for g, _ in pairs_a])
    types, confusions, deleted = Counter(), Counter(), 0
    for g, o in pairs_a:
        for gold_part, ocr_part in char_blocks(g, o):
            kind = block_type(gold_part, ocr_part)
            types[kind] += 1
            if kind in ("substitution", "deletion"):
                confusions[(gold_part, ocr_part)] += 1
            if kind == "deletion":
                deleted += len(gold_part)
    op_mix = {k: types[k] / sum(types.values()) for k in ("substitution", "deletion", "insertion", "other")}
    table = pd.DataFrame([{"gold": gp, "ocr": op, "count": c, "rate": c / aligned_gold.count(gp)}
                          for (gp, op), c in confusions.most_common() if c >= min_count])
    return table, op_mix, deleted / len(aligned_gold.replace(" ", ""))


def error_stats(pairs, min_len=1):
    """Per-word error rate min(1, edit distance / gold length), summarized."""
    r = pd.Series([min(1, Levenshtein.distance(g, o) / len(g)) for g, o in pairs if len(g) >= min_len])
    return {"n": len(r), "mean": r.mean(), "median": r.median(), "q1": r.quantile(0.25),
            "q3": r.quantile(0.75), "share_ge_0.40": (r >= 0.40).mean()}


def per_word_error(pairs_a, pairs_b):
    """Error-rate stats for Tier A (lower bound) and Tier B (upper bound), for words of at least
    4 characters and for all words."""
    return {tier: {"min4": error_stats(p, 4), "all": error_stats(p)}
            for tier, p in (("tier_a", pairs_a), ("tier_b", pairs_b))}


def per_word_table(stats):
    rows = {(tier, words): s for tier, d in stats.items() for words, s in d.items()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["pairs", "words"]).round(3)


def isolated_drop_rate(kept):
    """Word drops of 1 to 2 words with matching words on both sides, per gold word."""
    return sum(a["isolated_drops"] for a in kept.values()) / sum(a["gold_words"] for a in kept.values())


def _round(x):
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return round(float(x), 4) if isinstance(x, float) else x


def before_after(kept, floor=50):
    """Tier A without and with the length flags: pair count, top-5 confusions, edit mix and
    per-word error (median, mean, share at 0.40 or above)."""
    cols = {}
    for name, flags in (("before (ratio only)", False), ("after (ratio, no merged/split)", True)):
        pairs = tier_a(kept, floor, length_flags=flags)
        conf, mix, dele = char_tables(kept, pairs)
        col = {"Tier A pairs": len(pairs),
               "top-5 confusions": ", ".join(f"{r.gold}>{r.ocr or '∅'} {r.count}" for r in conf.head(5).itertuples())}
        col.update({f"edit mix: {k}": f"{v:.3f}" for k, v in mix.items()})
        col["deleted characters"] = f"{dele:.4f}"
        for words, s in per_word_error(pairs, all_pairs(kept))["tier_a"].items():
            col[f"per-word error, Tier A, {words}"] = f"median {s['median']:.3f}, mean {s['mean']:.3f}, >=0.40 {s['share_ge_0.40']:.3f}"
        cols[name] = col
    return pd.DataFrame(cols)


def save_calibration(path, seed, align_table, kept, pairs_a, confusions, op_mix, deletion_rate, per_word, drop_rate,
                     flags=None):
    """Write reports/calibration.json in the shape the damage pipeline (notebook 1) reads."""
    dropped = align_table[~align_table["used"]]
    totals, wer = coverage_totals(align_table)
    out = {"seed": seed, "excerpts_used": list(kept),
           "excerpts_dropped": [{"id": d, "reason": f"coverage {c:.2f} below 0.60"} for d, c in dropped["coverage"].items()],
           "cer_per_excerpt": align_table["cer"].to_dict(),
           "coverage": {k: int(v) for k, v in totals.items()},
           "word_error_rate_overall": wer,
           "pairs": {"tier_a": len(pairs_a), "tier_b": len(all_pairs(kept)),
                     **{k: len(v) for k, v in (flags or {}).items()}},
           "op_mix": op_mix, "char_deletion_rate": deletion_rate,
           "confusions": confusions.to_dict("records"),
           "per_word_error": per_word, "isolated_word_drop_rate": drop_rate}
    Path(path).write_text(json.dumps(_round(out), ensure_ascii=False, indent=1), encoding="utf-8")
    return out
