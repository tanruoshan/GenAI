"""Section 0b: the append-only prediction store and the runner both repair tracks use.

One file per method version: runs/preds/<method>_<version>.jsonl. A row is never changed or removed.
The key (id, severity, method, method_version) makes a rerun skip finished rows, so a crash or a
stopped run never repeats an API call.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from blnrepair.data import ROOT
from blnrepair.slots import build_slots, slot_core, slot_view, splice

PRED_FIELDS = ("id", "severity", "split", "method", "method_version", "model_name", "slots_in", "raw_output",
               "pred_words", "spliced_text", "format_ok", "error", "seconds", "timestamp")


def pred_path(method, version, preds_dir=None):
    return Path(preds_dir or ROOT / "runs" / "preds") / f"{method}_{version}.jsonl"


def row_key(rec):
    return (rec["id"], rec["severity"], rec["method"], rec["method_version"])


def load_preds(path):
    """All stored prediction rows, in file order. An empty list if the file does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_pred(path, rec):
    """Add one row. Refuses a duplicate key and any row that does not have exactly the store's fields."""
    assert set(rec) == set(PRED_FIELDS), f"prediction row fields differ: {set(rec) ^ set(PRED_FIELDS)}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if row_key(rec) in {row_key(r) for r in load_preds(path)}:
        raise ValueError(f"key already stored: {row_key(rec)}")
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def format_ok(pred_words, k):
    """One string per slot, nothing else. A wrong length or a non-string item is a format failure."""
    return isinstance(pred_words, list) and len(pred_words) == k and all(isinstance(w, str) for w in pred_words)


def run_method(rows, predict, method, method_version, model_name, allow_test=False, preds_dir=None):
    """Run predict(view) on every row that has slots and is not stored yet. Returns the rows added.

    predict gets the build_slots view and returns a dict with raw_output, pred_words (a list with one
    string per slot, or None) and optionally error. An exception from predict stops the run and stores
    nothing for that row, so a resume tries it again (used for network problems, never for bad answers).
    A wrong answer is data: it is stored with format_ok False, and no retry or manual fix is made.
    """
    path = pred_path(method, method_version, preds_dir)
    todo = [r for r in rows if r["k"]]
    if not allow_test:
        assert all(r["split"] != "test" for r in todo), "the test split is closed (allow_test=False)"
    done = {row_key(r) for r in load_preds(path)}
    added = []
    for row in todo:
        key = (row["id"], row["severity"], method, method_version)
        if key in done:
            continue
        assert allow_test or row["split"] != "test"
        view = build_slots(row)
        t0 = time.perf_counter()
        out = predict(view)
        seconds = time.perf_counter() - t0
        pred_words = out.get("pred_words")
        ok = format_ok(pred_words, row["k"])
        rec = {"id": row["id"], "severity": row["severity"], "split": row["split"], "method": method,
               "method_version": method_version, "model_name": model_name, "slots_in": slot_view(view),
               "raw_output": out.get("raw_output"), "pred_words": pred_words if ok else None,
               "spliced_text": splice(row, pred_words) if ok else None, "format_ok": ok,
               "error": out.get("error"), "seconds": round(seconds, 3),
               "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        append_pred(path, rec)
        added.append(rec)
    return added


def slot_exact(pred_words, row):
    """Sanity check, not a result: the share of slots whose core equals the gold core exactly
    (case-sensitive; a punctuation-only gold word compares as a whole token). A missing or wrong-length
    prediction list counts as 0. The real scoring comes later, in notebook 04."""
    plans = sorted(row["ops_per_word"], key=lambda p: p["idx"])
    if not format_ok(pred_words, len(plans)) or not plans:
        return 0.0
    return sum(slot_core(w) == slot_core(p["gold"]) for w, p in zip(pred_words, plans)) / len(plans)
