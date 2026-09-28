"""Notebook 4: scoring of the stored repairs, confidence intervals and paired tests.

Every score is computed over the damaged span of one row (one sentence at one level). The unit of
resampling and of the paired tests is the sentence: the slots of a sentence share its context and are
filled one after another, so they are not independent draws.
"""
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from scipy.stats import binomtest, wilcoxon

from blnrepair.preds import slot_exact
from blnrepair.slots import slot_core, splice


def span_row(row):
    """The row cut to its damaged span, so that splice builds the span text only."""
    s, e = row["span_start"], row["span_end"]
    return {"gold_tokens": row["gold_tokens"][s:e],
            "ops_per_word": [{**p, "idx": p["idx"] - s} for p in row["ops_per_word"]]}


def row_scores(row, pred_words):
    """All scores of one repair. pred_words None = a format failure, scored as no repair: its slots count
    as wrong and its text is the damaged text. Fact counts are kept per slot type, so that recovery can be
    pooled over slots (visible and dropped fact slots) as well as averaged per row."""
    plans = sorted(row["ops_per_word"], key=lambda p: p["idx"])
    ok = pred_words is not None
    words = pred_words if ok else [p["out"] for p in plans]
    right = [ok and slot_core(w) == slot_core(p["gold"]) for w, p in zip(words, plans)]
    facts = [(r, p["dropped"]) for r, p in zip(right, plans) if p["idx"] in row["fact_idx_in_span"]]
    gold_span = " ".join(row["gold_tokens"][row["span_start"]:row["span_end"]])
    damaged_span, repaired_span = splice(span_row(row), [p["out"] for p in plans]), splice(span_row(row), words)
    edits_before, edits_after = (Levenshtein.distance(t, gold_span) for t in (damaged_span, repaired_span))
    return {"format_ok": ok, "gold span": gold_span, "damaged span": damaged_span, "repaired span": repaired_span,
            "repaired sentence": splice(row, words), "exact": slot_exact(pred_words, row),
            "fact recovery": sum(r for r, _ in facts) / len(facts),
            "anchor ok": any(r for r, p in zip(right, plans) if p["idx"] == row["anchor_idx"]),
            "visible facts": sum(not d for _, d in facts), "visible facts ok": sum(r and not d for r, d in facts),
            "dropped facts": sum(d for _, d in facts), "dropped facts ok": sum(r and d for r, d in facts),
            "gold chars": len(gold_span), "edits before": edits_before, "edits after": edits_after,
            "edits saved": edits_before - edits_after,
            "CER before": edits_before / len(gold_span), "CER after": edits_after / len(gold_span)}


def sentence_matrix(scores, column, sentences):
    """One value per sentence, in the order of sentences (the same order for every method and level)."""
    return scores.set_index("id").loc[sentences, column].to_numpy(dtype=float)


def resample_index(n_sentences, n_boot, seed):
    """The bootstrap draws: n_boot rows of sentence positions, drawn with replacement. The same seed gives
    the same draws for every method, so their intervals come from the same resampled test sets."""
    return np.random.default_rng(seed).integers(0, n_sentences, size=(n_boot, n_sentences))


def bootstrap_ci(num, den, index, alpha=0.05):
    """Point estimate and percentile interval of sum(num) / sum(den) over resampled sentences.
    A per-sentence mean is den = 1 for every sentence; a pooled rate (pooled CER, recovery of dropped fact
    slots) is a sum of counts over a sum of counts. Resamples with den 0 are left out."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    top, bottom = num[index].sum(axis=1), den[index].sum(axis=1)
    stats = top[bottom > 0] / bottom[bottom > 0]
    point = num.sum() / den.sum() if den.sum() > 0 else np.nan
    if len(stats) == 0:
        return point, np.nan, np.nan
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, low, high


def mcnemar_exact(a_ok, b_ok):
    """Exact McNemar test for two methods on the same sentences (a binary outcome per sentence).
    Only sentences where exactly one method is right carry information; under the null each of them is
    a fair coin. Returns (a right only, b right only, two-sided p)."""
    a_ok, b_ok = np.asarray(a_ok, dtype=bool), np.asarray(b_ok, dtype=bool)
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def wilcoxon_paired(a, b):
    """Wilcoxon signed-rank test on paired scores (scipy defaults: sentences with equal scores are left
    out). Returns (sentences with a higher, with b higher, two-sided p)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    higher, lower = int((diff > 0).sum()), int((diff < 0).sum())
    p = wilcoxon(diff).pvalue if higher + lower else 1.0
    return higher, lower, p


def holm(pvalues):
    """Holm-adjusted p-values (family-wise error control without assuming independent tests)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def wrong_fact_rate(score, fact_ok):
    """SQ3: the chance that a repair with a wrong fact gets at least as high a score as a repair with the
    right fact (all pairs of one wrong and one right repair; equal scores count half). 0 = the score
    always ranks the right fact higher; 0.5 = the score does not see the fact at all. Equals 1 - AUC.
    NaN when one of the two groups is empty."""
    score, fact_ok = np.asarray(score, dtype=float), np.asarray(fact_ok, dtype=bool)
    wrong, right = score[~fact_ok], score[fact_ok]
    if len(wrong) == 0 or len(right) == 0:
        return np.nan
    diff = wrong[:, None] - right[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


# name: (numerator column, denominator column). None = a mean over sentences, where every sentence counts
# the same; a column = a rate pooled over slots or characters, where every slot or character counts the same.
METRICS = {"BERTScore span": ("BERTScore span", None), "Fact Recovery Rate": ("fact recovery", None),
           "anchor recovery": ("anchor ok", None), "visible fact slots": ("visible facts ok", "visible facts"),
           "dropped fact slots": ("dropped facts ok", "dropped facts"), "exact words": ("exact", None),
           "CER before (pooled)": ("edits before", "gold chars"), "CER after (pooled)": ("edits after", "gold chars"),
           "repair gain (pooled)": ("edits saved", "gold chars"), "BERTScore sentence": ("BERTScore sentence", None)}


def ci_long(scores, metrics, sentences, index, alpha=0.05):
    """Point estimate and bootstrap interval for every method, level and metric. scores holds one row per
    method, level and sentence; sentences fixes the order that index (resample_index) refers to."""
    out = []
    for (method, level), group in scores.groupby(["method", "level"], sort=False):
        for name in metrics:
            num_col, den_col = METRICS[name]
            num = sentence_matrix(group, num_col, sentences)
            den = sentence_matrix(group, den_col, sentences) if den_col else np.ones(len(sentences))
            value, low, high = bootstrap_ci(num, den, index, alpha)
            out.append({"method": method, "level": level, "metric": name, "value": value, "low": low, "high": high})
    return pd.DataFrame(out)


def paired_tests(scores, pairs, levels, sentences):
    """Per pair and level: exact McNemar on anchor recovery and Wilcoxon signed-rank on span BERTScore,
    first method against second, on the same sentences. Each test family (all McNemar p-values, all
    Wilcoxon p-values) is Holm-adjusted over all pairs and levels together."""
    out = []
    for a, b in pairs:
        for level in levels:
            at = lambda m, col: sentence_matrix(scores[(scores["method"] == m) & (scores["level"] == level)], col, sentences)
            only_a, only_b, p_m = mcnemar_exact(at(a, "anchor ok"), at(b, "anchor ok"))
            higher, lower, p_w = wilcoxon_paired(at(a, "BERTScore span"), at(b, "BERTScore span"))
            out.append({"first": a, "second": b, "level": level, "anchor: first only": only_a,
                        "anchor: second only": only_b, "McNemar p": p_m, "BERTScore: first higher": higher,
                        "BERTScore: second higher": lower, "Wilcoxon p": p_w})
    table = pd.DataFrame(out)
    table["McNemar p (Holm)"] = holm(table["McNemar p"])
    table["Wilcoxon p (Holm)"] = holm(table["Wilcoxon p"])
    return table
