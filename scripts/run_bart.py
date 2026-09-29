"""Repair every row of a split with the fine-tuned BART model and store the answers (notebook 2c).

Runs where the model was trained (the pod), because training and generation do not fit on an 8 GB laptop:
    python scripts/run_bart.py --split dev
    python scripts/run_bart.py --split test --allow-test     # only after configs/bart_repair.yaml is frozen
Resumes like every run: rows already in runs/preds/bart_<version>.jsonl are skipped.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

from blnrepair.bart_data import BartRepairer, bart_version, load_bart_repair_config  # noqa: E402
from blnrepair.freeze import load_frozen  # noqa: E402
from blnrepair.preds import load_preds, pred_path, run_method  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    cfg = load_bart_repair_config()
    if args.split == "test":
        assert args.allow_test and cfg["frozen"], "the test split is closed: pass --allow-test and freeze configs/bart_repair.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(ROOT / cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["model"])
    repairer = BartRepairer(model, tokenizer, device, cfg["num_beams"], cfg["max_length"])
    version = bart_version(cfg, args.split)
    rows = [r for r in load_frozen("v2") if r["split"] == args.split]
    added = run_method(rows, repairer, "bart", version, cfg["model"], allow_test=args.allow_test)
    stored = load_preds(pred_path("bart", version))
    failures = sum(not p["format_ok"] for p in stored)
    print(f"bart {version}: {len(stored)} rows stored ({len(added)} new), {failures} format failures, device {device}")
