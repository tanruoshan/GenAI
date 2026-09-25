import json

import pytest
import yaml

from blnrepair.corrupt import load_table, settings
from blnrepair.data import ROOT, load_config
from blnrepair.freeze import build_records, check_records, freeze, serialize, verify_freeze

SENTENCES = ROOT / "data" / "processed" / "sentences.jsonl"
CALIBRATION = ROOT / "reports" / "calibration.json"
pytestmark = pytest.mark.skipif(not (SENTENCES.exists() and CALIBRATION.exists()),
                                reason="needs sections D and E outputs")


@pytest.fixture(scope="module")
def small():
    """Two test sentences and one dev sentence, with the real settings and table."""
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in SENTENCES.read_text(encoding="utf-8").splitlines()]
    rows = [rows[0], rows[1], rows[-1]]
    assert [r["split"] for r in rows] == ["test", "test", "dev"]
    cfg = settings(load_config(), calibration)
    return rows, load_table(calibration), cfg


def files(tmp_path):
    processed, reports, runs = tmp_path / "processed", tmp_path / "reports", tmp_path / "runs"
    processed.mkdir(), reports.mkdir()
    (processed / "sentences.jsonl").write_text("x\n", encoding="utf-8")
    (reports / "calibration.json").write_text("{}\n", encoding="utf-8")
    return processed, reports, runs


def test_serialize_is_lf_only_with_sorted_keys_and_real_characters(small):
    rows, table, cfg = small
    data = serialize(build_records(rows, table, cfg))
    assert b"\r" not in data and data.endswith(b"\n")
    first = json.loads(data.decode("utf-8").splitlines()[0])
    assert list(first) == sorted(first)
    assert data == serialize(build_records(rows, table, cfg))


def test_freeze_writes_once_and_is_idempotent(small, tmp_path):
    rows, table, cfg = small
    records = build_records(rows, table, cfg)
    processed, reports, runs = files(tmp_path)
    status, digest = freeze(records, "v1", cfg, processed, reports, runs, n_test=2, n_dev=1)
    assert status == "written" and len(records) == 18
    assert (runs / "corrupted_v1.sha256").read_text(encoding="utf-8") == f"{digest}  corrupted_v1.jsonl\n"
    snapshot = yaml.safe_load((runs / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert snapshot["version"] == "v1" and snapshot["corruption"]["p_deletion"] == 0.15
    assert freeze(records, "v1", cfg, processed, reports, runs, n_test=2, n_dev=1) == ("unchanged", digest)
    assert verify_freeze(rows, table, cfg, "v1", processed, runs) == digest


def test_freeze_refuses_different_content_under_the_same_version(small, tmp_path):
    rows, table, cfg = small
    processed, reports, runs = files(tmp_path)
    records = build_records(rows, table, cfg)
    freeze(records, "v1", cfg, processed, reports, runs, n_test=2, n_dev=1)
    changed = build_records(rows, table, dict(cfg, seed=7))
    with pytest.raises(FileExistsError, match="new version number"):
        freeze(changed, "v1", cfg, processed, reports, runs, n_test=2, n_dev=1)


def test_checks_catch_duplicates_missing_keys_and_damaged_level_0(small):
    rows, table, cfg = small
    records = build_records(rows, table, cfg)
    check_records(records, 2, 1, 6)
    with pytest.raises(AssertionError):
        check_records(records + [records[0]], 2, 1, 6)
    with pytest.raises(AssertionError):
        check_records([{k: v for k, v in records[0].items() if k != "k"}] + records[1:], 2, 1, 6)
    bad = [dict(records[0], corrupted_text="changed")] + records[1:]
    with pytest.raises(AssertionError):
        check_records(bad, 2, 1, 6)
