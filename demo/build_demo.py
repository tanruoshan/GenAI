"""Build the interactive defence demo: one self-contained, offline HTML page.

    python demo/build_demo.py                 # -> demo/out/bln600_demo.html
    python demo/build_demo.py --bertscore     # also score every repaired span with BERTScore (CPU, ~5-10 min
                                              # once; cached in demo/out/), which unlocks the "metric trap" plot

Inputs (nothing is sent to a model, nothing is re-tuned):
  - data/processed/corrupted_v2.jsonl, hash-checked like every notebook (built by notebooks 00 and 01)
  - runs/preds/*-test.jsonl, the stored test predictions of the six methods
  - notebooks/04_evaluation.ipynb, its one executed test run: every interval, test and rate shown in the
    demo is read from there, so the demo cannot drift from the paper. Per-sentence scores are recomputed
    with the same code (blnrepair.evaluation.row_scores) and checked against the notebook's means.
  - data/raw/BLN600 (optional), for the word list of the live dictionary lookup.

The page embeds BLN600 text (CC BY-NC-ND), so demo/out/ is git-ignored: build it locally, never commit it.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import yaml

from blnrepair.data import ROOT, load_config, load_gold
from blnrepair.evaluation import row_scores, wrong_fact_rate
from blnrepair.freeze import load_frozen
from blnrepair.lexicon import build_lexicon
from blnrepair.preds import load_preds
from blnrepair.slots import slot_core

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
NOTEBOOK = ROOT / "notebooks" / "04_evaluation.ipynb"
LEVELS = ["1w", "10", "25", "50", "75"]
SPLIT = "test"

# notebook-4 name -> (short key, display name, role). Order = the fixed colour order of the demo.
METHODS = [
    ("BERT + dictionary", "bertdict", "BERT + dict.", "main MLM method"),
    ("few-shot", "llm", "LLM few-shot", "main LLM method"),
    ("BART", "bart", "BART", "descriptive only"),
    ("dictionary lookup", "lookup", "Lookup", "non-GenAI baseline"),
    ("BERT, no dictionary", "bert", "BERT, no dict.", "ablation"),
    ("few-shot + article", "llmart", "LLM + article", "upper bound"),
]
NAME = {nb: key for nb, key, _, _ in METHODS} | {"no repair": "none"}
METRIC_TABLES = {"anchor recovery": "anchor", "Fact Recovery Rate": "frr", "BERTScore span": "bs",
                 "exact words": "exact", "CER after (pooled)": "cer"}


# ---- notebook 4: the reported numbers -----------------------------------------------------------------

class _Table(HTMLParser):
    """The first HTML table of a pandas display, as a list of rows of cell texts."""

    def __init__(self):
        super().__init__()
        self.rows, self.cell = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self.cell = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.rows[-1].append(self.cell.strip())
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data


def _html_rows(out):
    parser = _Table()
    parser.feed("".join(out["data"]["text/html"]))
    return parser.rows


CELL = re.compile(r"(-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\]")


def notebook_results(path=NOTEBOOK):
    """The interval tables (metric -> method -> level -> [value, low, high]), the paired tests and the SQ3
    rates, exactly as notebook 4 printed them in its test run."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    assert f'SPLIT = "{SPLIT}"' in "".join(nb["cells"][1]["source"]), "notebook 4 is not the test run"
    ci, tests, sq3, title = {}, [], {}, None
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                lines = "".join(out["text"]).strip().splitlines()
                title = lines[-1].strip() if lines else title
                continue
            if "text/html" not in out.get("data", {}):
                continue
            rows = _html_rows(out)
            if "paired_tests(" in source:
                head = rows[0][1:]
                for r in rows[1:]:
                    rec = dict(zip(head, r[1:]))
                    tests.append({"pair": [NAME[rec["first"]], NAME[rec["second"]]], "level": rec["level"],
                                  "anchor_only": [int(rec["anchor: first only"]), int(rec["anchor: second only"])],
                                  "bs_higher": [int(rec["BERTScore: first higher"]), int(rec["BERTScore: second higher"])],
                                  "mcnemar_holm": float(rec["McNemar p (Holm)"]),
                                  "wilcoxon_holm": float(rec["Wilcoxon p (Holm)"])})
            elif "wrong_fact_rate(" in source:
                head = rows[0][1:]
                for r in rows[1:]:
                    if r and r[0] in LEVELS:
                        for m, v in zip(head, r[1:]):
                            sq3.setdefault(NAME[m], {})[r[0]] = float(v) if v not in ("", "nan", "NaN") else None
            elif title in METRIC_TABLES:
                head = rows[0][1:]
                for r in rows[1:]:
                    if r and r[0] in LEVELS:
                        for m, v in zip(head, r[1:]):
                            hit = CELL.fullmatch(v)
                            if hit:
                                ci.setdefault(METRIC_TABLES[title], {}).setdefault(NAME[m], {})[r[0]] = \
                                    [float(x) for x in hit.groups()]
                title = None
            else:
                title = None
    return {"ci": ci, "tests": tests, "sq3": sq3}


# ---- per-sentence scores -------------------------------------------------------------------------------

def method_files(path=NOTEBOOK):
    """The prediction file of each method, read from the version table that notebook 4 printed in its test
    run, so the demo scores exactly the files the paper scored."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if "text/html" in out.get("data", {}):
                rows = _html_rows(out)
                if rows and "version" in rows[0]:
                    col = rows[0].index("version")
                    return {r[0]: ROOT / "runs" / "preds" / f"{r[col]}.jsonl" for r in rows[1:] if len(r) > col}
    raise RuntimeError("notebook 4 has no version table")


def slot_marks(row, pred_words):
    """Per slot (in idx order): the repaired word and whether it restores the gold word."""
    plans = sorted(row["ops_per_word"], key=lambda p: p["idx"])
    if pred_words is None:
        return [None] * len(plans), [False] * len(plans)
    return list(pred_words), [slot_core(w) == slot_core(p["gold"]) for w, p in zip(pred_words, plans)]


def sentence_record(rows_by_level, preds, bs_cache):
    base = rows_by_level["1w"]
    s = {"id": base["id"], "band": base["band"], "tokens": base["gold_tokens"], "anchor": base["anchor_idx"],
         "levels": {}, "repairs": {}}
    facts = set()
    for lv in LEVELS:
        row = rows_by_level[lv]
        plans = sorted(row["ops_per_word"], key=lambda p: p["idx"])
        facts |= set(row["fact_idx_in_span"])
        none = row_scores(row, None)
        s["levels"][lv] = {"span": [row["span_start"], row["span_end"]],
                           "slots": [[p["idx"], p["out"], int(p["dropped"])] for p in plans],
                           "facts": row["fact_idx_in_span"], "cer": round(none["CER before"], 3),
                           "bs_none": bs_cache.get(f"none|{row['id']}|{lv}")}
        s["repairs"][lv] = {}
        for nb, key, _, _ in METHODS:
            p = preds[nb][(row["id"], lv)]
            words = p["pred_words"] if p["format_ok"] else None
            sc = row_scores(row, words)
            filled, right = slot_marks(row, words)
            s["repairs"][lv][key] = {"w": filled, "ok": [int(x) for x in right], "fmt": int(sc["format_ok"]),
                                     "anchor": int(sc["anchor ok"]), "frr": round(sc["fact recovery"], 3),
                                     "exact": round(sc["exact"], 3), "cer": round(sc["CER after"], 3),
                                     "bs": bs_cache.get(f"{key}|{row['id']}|{lv}")}
    s["facts"] = sorted(facts)
    return s


def bertscore_cache(rows, preds, compute):
    """Span BERTScore per method, sentence and level, with the notebook's settings. Read from the cache;
    computed (and cached) only with --bertscore."""
    path = OUT / "bertscore_span.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not compute:
        return cache
    from bert_score import BERTScorer  # noqa: E402  (heavy; only on request)
    keys, cands, refs = [], [], []
    for row in rows:
        for nb, key, _, _ in METHODS + [("no repair", "none", "", "")]:
            k = f"{key}|{row['id']}|{row['severity']}"
            if k in cache:
                continue
            words = None
            if key != "none":
                p = preds[nb][(row["id"], row["severity"])]
                words = p["pred_words"] if p["format_ok"] else None
            sc = row_scores(row, words)
            keys.append(k)
            cands.append(sc["repaired span"] if key != "none" else sc["damaged span"])
            refs.append(sc["gold span"])
    if keys:
        print(f"BERTScore for {len(keys)} spans (roberta-large, rescaled, as in notebook 4) ...")
        scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)
        f1 = scorer.score(cands, refs)[2].numpy()
        cache.update({k: round(float(v), 4) for k, v in zip(keys, f1)})
        OUT.mkdir(exist_ok=True)
        path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def manual_ids(ids, seed):
    """The 6 sentences of notebook 4's seeded manual check (same ranking rule)."""
    rank = lambda i: hashlib.sha256(f"{seed}-manual-{i}".encode()).hexdigest()
    return sorted(ids, key=rank)[:6]


def check_against_notebook(sentences, ci):
    """Recomputed means must equal the notebook's reported values (2 decimals). Returns (checked, mismatches)."""
    checked, bad = 0, []
    for metric, field in (("anchor", "anchor"), ("frr", "frr"), ("exact", "exact")):
        for nb, key, _, _ in METHODS:
            for lv in LEVELS:
                reported = ci.get(metric, {}).get(key, {}).get(lv)
                if reported is None:
                    continue
                mean = float(np.mean([s["repairs"][lv][key][field] for s in sentences]))
                checked += 1
                if abs(round(mean, 2) - reported[0]) > 0.0051:
                    bad.append(f"{metric} {key} {lv}: recomputed {mean:.3f}, notebook {reported[0]:.2f}")
    return checked, bad


def valid_only(sentences):
    """LLM (and other free-text methods) over their valid answers only: means of anchor and FRR."""
    out = {}
    for key in ("llm", "llmart", "bart"):
        for lv in LEVELS:
            ok = [s["repairs"][lv][key] for s in sentences if s["repairs"][lv][key]["fmt"]]
            out.setdefault(key, {})[lv] = {"n": len(ok), "anchor": round(float(np.mean([r["anchor"] for r in ok])), 3),
                                           "frr": round(float(np.mean([r["frr"] for r in ok])), 3),
                                           "fail": len(sentences) - len(ok)}
    return out


def lexicon_counts(rows, lexicon_json):
    if lexicon_json:
        return json.loads(Path(lexicon_json).read_text(encoding="utf-8"))
    raw = ROOT / load_config()["paths"]["raw_dir"]
    if not (raw / "Ground Truth").exists():
        print(f"note: {raw} not found, the live dictionary lookup is left out")
        return None
    counts = build_lexicon(load_gold(raw), {r["doc_id"] for r in rows})
    return dict(counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bertscore", action="store_true", help="compute span BERTScore per repair (cached)")
    ap.add_argument("--data", help="a damaged-data file to use instead of the hash-checked frozen file "
                                   "(for tests only; the page is then marked as unverified)")
    ap.add_argument("--lexicon-json", help="a saved {word: count} file instead of building the word list")
    ap.add_argument("--out", default=str(OUT / "bln600_demo.html"))
    args = ap.parse_args(argv)

    if args.data:
        rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
        data_note = f"UNVERIFIED data file {Path(args.data).name}: not the study's frozen data"
    else:
        rows = load_frozen("v2")
        sha = (ROOT / "runs" / "corrupted_v2.sha256").read_text(encoding="utf-8").split()[0]
        data_note = f"frozen corrupted_v2, SHA-256 {sha[:12]}… verified"
    test = [r for r in rows if r["split"] == SPLIT and r["k"]]
    by_id = {}
    for r in test:
        by_id.setdefault(r["id"], {})[r["severity"]] = r

    files = method_files()
    preds = {nb: {(p["id"], p["severity"]): p for p in load_preds(files[nb])} for nb, *_ in METHODS}
    for nb, *_ in METHODS:
        missing = {(r["id"], r["severity"]) for r in test} - set(preds[nb])
        assert not missing, f"{nb}: {len(missing)} test rows have no stored prediction"

    bs_cache = bertscore_cache(test, preds, args.bertscore)
    sentences = [sentence_record(by_id[i], preds, bs_cache) for i in sorted(by_id)]
    nb4 = notebook_results()
    checked, bad = check_against_notebook(sentences, nb4["ci"])
    if bad and not args.data:
        sys.exit("recomputed scores differ from notebook 4:\n  " + "\n  ".join(bad))

    seed = load_config()["seed"]
    has_bs = all(s["repairs"][lv][k]["bs"] is not None for s in sentences for lv in LEVELS for _, k, _, _ in METHODS)
    sq3_live = {}
    if has_bs:  # the same rate, recomputed from the cache: must agree with the notebook
        for _, key, _, _ in METHODS:
            for lv in LEVELS[1:]:
                rs = [s["repairs"][lv][key] for s in sentences]
                sq3_live.setdefault(key, {})[lv] = round(wrong_fact_rate([r["bs"] for r in rs], [r["anchor"] for r in rs]), 3)
    llm_model = yaml.safe_load((ROOT / "configs" / "llm.yaml").read_text(encoding="utf-8"))["model"]
    probe = load_preds(ROOT / "runs" / "preds" / f"llm_probe_v1-{llm_model}-{SPLIT}.jsonl")
    probe_rows = [json.loads(p["raw_output"]) for p in probe]
    lex = lexicon_counts(rows, args.lexicon_json)

    payload = {
        "meta": {"data": data_note, "verified": not args.data, "checked": checked, "mismatches": bad,
                 "n_sentences": len(sentences), "bertscore": has_bs, "seed": seed},
        "levels": LEVELS,
        "methods": [{"key": k, "name": n, "role": r, "nb": nb} for nb, k, n, r in METHODS],
        "sentences": sentences,
        "featured": manual_ids(sorted(by_id), seed),
        "ci": nb4["ci"], "tests": nb4["tests"], "sq3": nb4["sq3"], "sq3_live": sq3_live,
        "valid": valid_only(sentences),
        "probe": {"n": len(probe_rows), "near": sum(int(p.get("near_verbatim", 0)) for p in probe_rows),
                  "median": round(float(np.median([p["similarity"] for p in probe_rows])), 2) if probe_rows else None,
                  "max": round(float(max(p["similarity"] for p in probe_rows)), 2) if probe_rows else None},
        "lexicon": sorted(lex.items(), key=lambda kv: (-kv[1], kv[0])) if lex else None,
    }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (HERE / "template.html").read_text(encoding="utf-8").replace("/*__PAYLOAD__*/null", blob)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"written: {out} ({out.stat().st_size / 1e6:.1f} MB) · {len(sentences)} sentences · "
          f"{checked} means checked against notebook 4, {len(bad)} differ · BERTScore per repair: "
          f"{'yes' if has_bs else 'no (run with --bertscore)'} · word list: {'yes' if lex else 'no'}")


if __name__ == "__main__":
    main()
