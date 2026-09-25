"""Section G: freeze the damaged dataset with a version number and a file hash."""
import hashlib
import json
import sys
from collections import Counter
from importlib.metadata import version as package_version
from pathlib import Path

import yaml

from blnrepair.corrupt import corrupt_sentence

SCHEMA_KEYS = {"id", "doc_id", "gold_text", "gold_tokens", "band", "severity", "k", "frac", "anchor_idx",
               "span_start", "span_end", "damaged_idx", "ops_per_word", "corrupted_text", "mask_positions",
               "fact_idx_in_span", "seed", "split"}


def build_records(rows, table, cfg):
    """All sentences at all levels, in sentences.jsonl order, then level order 0, 1w, 10, 25, 50, 75."""
    return [corrupt_sentence(row, level, table, cfg) for row in rows for level in cfg["levels"]]


def serialize(records):
    """One JSON line per record, sorted keys, UTF-8, LF line endings (the same bytes on every system)."""
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text_file(path):
    """Hash of an input file with CRLF read as LF, so Windows and Linux copies give the same value."""
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def check_records(records, n_test, n_dev, n_levels):
    """The freeze checks: row count, unique (id, severity), exact schema, split sizes, clean level 0."""
    assert len(records) == (n_test + n_dev) * n_levels, f"row count {len(records)}"
    keys = [(r["id"], r["severity"]) for r in records]
    assert len(set(keys)) == len(keys), "(id, severity) is not unique"
    assert all(set(r) == SCHEMA_KEYS for r in records), "a record does not match the schema"
    assert Counter(r["split"] for r in records) == {"test": n_test * n_levels, "dev": n_dev * n_levels}
    assert all(r["corrupted_text"] == r["gold_text"] for r in records if r["severity"] == "0"), "level 0 is damaged"


def make_snapshot(version, cfg, data_name, digest, calibration_path, sentences_path):
    """Everything needed to see how the frozen file was made."""
    keys = ("drop_prob", "max_deletions", "intensity_low", "intensity_high", "p_deletion", "levels")
    return {"version": version, "seed": cfg["seed"],
            "corruption": {k: cfg[k] for k in keys},
            "level_sizes": {"1w": "k = 1",
                            "10, 25, 50, 75": "k = (pct * n + 50) // 100, n = words in the sentence (integer half-up rounding)"},
            "span": "start = clamp(anchor - k // 2, 0, n - k), length k; one seeded anchor per sentence among its fact tokens, never dropped",
            "corrupted_file": {"name": data_name, "sha256": digest},
            "inputs_sha256 (CRLF read as LF)": {"reports/calibration.json": sha256_text_file(calibration_path),
                                                "data/processed/sentences.jsonl": sha256_text_file(sentences_path)},
            "python": sys.version.split()[0], "pysbd": package_version("pysbd"), "rapidfuzz": package_version("rapidfuzz")}


def freeze(records, version, cfg, processed_dir, reports_dir, runs_dir, n_test=150, n_dev=20):
    """Write corrupted_<version>.jsonl, runs/corrupted_<version>.sha256 and runs/config_snapshot.yaml.
    Idempotent: if the file exists with the same hash, nothing is done. If the content would differ,
    an error asks for a new version number. Returns ("written" or "unchanged", sha256)."""
    check_records(records, n_test, n_dev, len(cfg["levels"]))
    data = serialize(records)
    digest = sha256_bytes(data)
    processed_dir, reports_dir, runs_dir = Path(processed_dir), Path(reports_dir), Path(runs_dir)
    path = processed_dir / f"corrupted_{version}.jsonl"
    if path.exists():
        if sha256_bytes(path.read_bytes()) == digest:
            return "unchanged", digest
        raise FileExistsError(f"{path.name} exists and the new content differs. Use a new version number.")
    snapshot = make_snapshot(version, cfg, path.name, digest, reports_dir / "calibration.json", processed_dir / "sentences.jsonl")
    runs_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    (runs_dir / f"corrupted_{version}.sha256").write_bytes(f"{digest}  {path.name}\n".encode("utf-8"))
    (runs_dir / "config_snapshot.yaml").write_bytes(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True).encode("utf-8"))
    return "written", digest


def verify_freeze(rows, table, cfg, version, processed_dir, runs_dir):
    """Rebuild everything in memory, serialize the same way, and compare with the stored hash and the file."""
    stored = (Path(runs_dir) / f"corrupted_{version}.sha256").read_text(encoding="utf-8").split()[0]
    on_disk = sha256_bytes((Path(processed_dir) / f"corrupted_{version}.jsonl").read_bytes())
    rebuilt = sha256_bytes(serialize(build_records(rows, table, cfg)))
    assert rebuilt == stored == on_disk, f"hash mismatch: rebuilt {rebuilt}, stored {stored}, file {on_disk}"
    return rebuilt
