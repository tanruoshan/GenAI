"""Load BLN600 gold text and metadata (raw data is read-only)."""
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pysbd
import yaml

from blnrepair.facts import fact_indices

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_gold(raw):
    """Join wrapped lines and return a list of paragraphs (blank line = paragraph break)."""
    raw = raw.replace("\xa0", " ")
    raw = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "-", raw)  # hyphen at a line end: join, keep hyphen
    paragraphs = re.split(r"\n\s*\n", raw)
    return [" ".join(p.split()) for p in paragraphs if p.strip()]


def load_gold(raw_dir, folder="Ground Truth"):
    """Return {doc_id: [paragraphs]} for every text file in a BLN600 folder."""
    files = sorted((Path(raw_dir) / folder).glob("*.txt"))
    return {f.stem: normalize_gold(f.read_text(encoding="utf-8")) for f in files}


def load_metadata(raw_dir):
    """Return one row per excerpt: doc_id, date, year, publication, article_count."""
    with (Path(raw_dir) / "metadata.json").open(encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    df["publication"] = df["publication"].map(html.unescape)
    df["year"] = df["date"].str[:4].astype(int)
    return df.rename(columns={"short_id": "doc_id", "doc_id": "gale_id"}).set_index("doc_id")


def excerpt_stats(gold):
    """Paragraph and word counts per excerpt (words are whitespace tokens)."""
    rows = {d: {"paragraphs": len(p), "words": sum(len(x.split()) for x in p)} for d, p in gold.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


TITLES = {"Mr.", "Mrs.", "Messrs.", "Dr.", "Wm.", "Thos.", "Capt.", "Rev.", "Col."}


def _ends_with_title(segment):
    last = re.sub(r"^\W+", "", segment.split()[-1])
    return last in TITLES or re.fullmatch(r"[A-Z]\.", last) is not None  # "Mr." or an initial "J."


AMOUNT_END = re.compile(r"[\d½¼¾][sdl]\.[)”’\"]*$")  # "5s." "6d." "½d." "100l."
LEADING_DASH = re.compile(r"^(?:[–—]|-{2,})+\s*")  # a dash at the start of a sentence, alone or attached ("—Jules")


def _ends_with_amount(segment):
    return AMOUNT_END.search(segment.split()[-1]) is not None


def split_sentences(paragraph, segmenter):
    """pysbd split, then re-join three kinds of false split: after a title or initial
    ("—Mr." with a dash attached is missed by pysbd), before a lowercase start (after
    "6d.", "inst.", "100l."), and between two parts of an amount ("£2 5s." then "10 ½d.").
    A dash at the start of a sentence, alone or attached to the first word, is removed."""
    merged = []
    for seg in (s.strip() for s in segmenter.segment(paragraph)):
        if not seg:
            continue
        starts_lower = re.sub(r"^\W+", "", seg)[:1].islower()
        amount_continues = re.match(r"[\d½¼¾]", seg) and merged and _ends_with_amount(merged[-1])
        if merged and (starts_lower or amount_continues or _ends_with_title(merged[-1])):
            merged[-1] += " " + seg
        else:
            merged.append(seg)
    return [text for text in (LEADING_DASH.sub("", m) for m in merged) if text]


def band_of(n_words, bands):
    """Name of the length band ("20-29") that contains n_words, or None."""
    for lo, hi in bands:
        if lo <= n_words <= hi:
            return f"{lo}-{hi}"
    return None


def build_sentences(gold, bands):
    """Return all sentences of all excerpts, with word count, length band and fact-token positions."""
    seg = pysbd.Segmenter(language="en", clean=False)
    rows = []
    for doc_id, paragraphs in gold.items():
        texts = [s for p in paragraphs for s in split_sentences(p, seg)]
        for i, text in enumerate(texts):
            tokens = text.split()
            facts = fact_indices(tokens)
            rows.append({"id": f"{doc_id}-{i:03d}", "doc_id": doc_id, "text": text, "n_words": len(tokens),
                         "band": band_of(len(tokens), bands), "fact_idx": facts, "n_facts": len(facts)})
    return pd.DataFrame(rows)


def make_pool(sentences, pool_cfg):
    """Sentences of 20-60 words with enough fact tokens. Relaxes to the fallback fact
    count only if the strict pool is smaller than the threshold. Returns (pool, min_facts)."""
    in_range = sentences[sentences["band"].notna()]
    min_facts = pool_cfg["min_fact_tokens"]
    pool = in_range[in_range["n_facts"] >= min_facts]
    if len(pool) < pool_cfg["fallback_pool_threshold"]:
        min_facts = pool_cfg["min_fact_tokens_fallback"]
        pool = in_range[in_range["n_facts"] >= min_facts]
    return pool, min_facts


def pool_funnel(sentences, pool_cfg):
    """How many sentences (and excerpts) survive each filter, in order."""
    in_range = sentences[sentences["band"].notna()]
    steps = {"all sentences": sentences,
             f"{pool_cfg['min_words']} to {pool_cfg['max_words']} words": in_range,
             "... and at least 1 fact token": in_range[in_range["n_facts"] >= 1],
             "... and at least 2 fact tokens": in_range[in_range["n_facts"] >= 2]}
    rows = {name: {"sentences": len(df), "excerpts": df["doc_id"].nunique()} for name, df in steps.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def pool_table(pool):
    """Sentence counts by length band (rows) and number of fact tokens (columns: 2, 3, 4+)."""
    group = pool["n_facts"].clip(upper=4).map(lambda n: "4+" if n == 4 else str(n))
    table = pd.crosstab(pool["band"], group)
    table["total"] = table.sum(axis=1)
    table["share"] = (table["total"] / table["total"].sum()).round(3)
    return table


LENGTH_BINS = [(0, 19, "under 20"), (20, 29, "20-29"), (30, 44, "30-44"), (45, 60, "45-60"),
               (61, 80, "61-80"), (81, 10**9, "81 and more")]


def length_coverage(sentences):
    """Sentences and words per length bin, over all sentences (before the 20 to 60 filter)."""
    label = sentences["n_words"].map(lambda n: next(name for lo, hi, name in LENGTH_BINS if lo <= n <= hi))
    table = pd.DataFrame({"sentences": label.value_counts(), "words": sentences.groupby(label)["n_words"].sum()})
    table = table.reindex([name for _, _, name in LENGTH_BINS])
    table["share of sentences"] = (table["sentences"] / table["sentences"].sum()).round(3)
    table["share of words"] = (table["words"] / table["words"].sum()).round(3)
    table.index.name = "words per sentence"
    return table[["sentences", "share of sentences", "words", "share of words"]]


def long_examples(sentences, lo, hi, n=3, seed=42, chars=200):
    """A few sentences with lo to hi words: id, word count and the first characters."""
    pick = sentences[sentences["n_words"].between(lo, hi)].sample(n, random_state=seed)
    return pd.DataFrame({"n_words": pick["n_words"], "start of sentence": pick["text"].str[:chars]}).set_index(pick["id"])


def band_quotas(pool, n):
    """Split n sentences over the length bands in proportion to the pool (largest remainder)."""
    counts = pool["band"].value_counts().sort_index()
    raw = counts / counts.sum() * n
    quota = raw.astype(int)
    quota[(raw - quota).sort_values(ascending=False, kind="stable").index[: n - quota.sum()]] += 1
    return quota


def draw_sample(pool, sample_cfg, max_per_excerpt, seed):
    """Seeded random sample with band quotas. Test is drawn first, then dev from the excerpts that
    test does not use. At most max_per_excerpt sentences come from one excerpt. The random order is
    a sha256 of (seed, sentence id), so it does not depend on the order of the pool."""
    keys = [hashlib.sha256(f"{seed}:{i}".encode()).hexdigest() for i in pool["id"]]
    order = pool.assign(key=keys).sort_values("key")
    used_excerpts, picks = set(), []
    for split, n in [("test", sample_cfg["n_test"]), ("dev", sample_cfg["n_dev"])]:
        left, per_excerpt = band_quotas(pool, n).to_dict(), Counter()
        for row in order.itertuples():
            if row.doc_id in used_excerpts or left[row.band] == 0 or per_excerpt[row.doc_id] >= max_per_excerpt:
                continue
            per_excerpt[row.doc_id] += 1
            left[row.band] -= 1
            picks.append((row.Index, split))
            if not any(left.values()):
                break
        assert not any(left.values()), f"could not fill the {split} quotas: {left}"
        used_excerpts |= set(per_excerpt)
    sample = pool.loc[[i for i, _ in picks]].assign(split=[s for _, s in picks])
    return sample.rename(columns={"text": "gold_text"}).reset_index(drop=True)


def sample_tables(sample):
    """Counts by split and band, and how the sample spreads over excerpts."""
    by_band = pd.crosstab(sample["split"], sample["band"], margins=True, margins_name="total")
    per_excerpt = sample.groupby("doc_id").size()
    shared = sample.groupby("doc_id")["split"].nunique().gt(1).sum()
    spread = pd.Series({"sentences": len(sample), "excerpts": len(per_excerpt), "most sentences from one excerpt": per_excerpt.max(),
                        "excerpts with both test and dev sentences": shared}, name="sample")
    return by_band, spread


def save_sentences(sample, path):
    """Write the sample as JSON lines (UTF-8), one sentence per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in sample.to_dict("records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
