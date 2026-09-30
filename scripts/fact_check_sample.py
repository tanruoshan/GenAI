"""Draw the 50 fact tokens for the manual fact-rule check (reports/fact_check/).

Pool: every fact slot of the 150 test sentences at level 75 (the 75% span holds every fact slot, 624 in total).
The draw uses the project seed, so it is reproducible. Writes reports/fact_check/fact_sample.csv with an empty
is_fact column (the marks were added by hand). Run: .venv/bin/python scripts/fact_check_sample.py
"""
import csv
import random
from pathlib import Path

from blnrepair.corrupt import seed_int
from blnrepair.freeze import load_frozen

OUT = Path(__file__).resolve().parents[1] / "reports" / "fact_check" / "fact_sample.csv"


def main():
    rows = [r for r in load_frozen() if r["split"] == "test" and str(r["severity"]) == "75"]
    pool = [(r, i) for r in rows for i in r["fact_idx_in_span"]]
    sample = random.Random(seed_int(42, "fact_check", "test75")).sample(pool, 50)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "id", "idx", "token", "left", "right", "is_fact"])
        w.writeheader()
        for n, (r, i) in enumerate(sample, 1):
            t = r["gold_tokens"]
            w.writerow({"n": n, "id": r["id"], "idx": i, "token": t[i], "left": " ".join(t[max(0, i - 12):i]),
                        "right": " ".join(t[i + 1:i + 13]), "is_fact": ""})
    print(f"{len(pool)} fact slots, 50 drawn -> {OUT}")


if __name__ == "__main__":
    main()
