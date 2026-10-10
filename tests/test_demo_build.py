"""The demo reads its aggregate numbers from the executed notebook 4. These tests check that parsing against
values reported in the paper, and that the page template and the prediction files it needs are in place.
No BLN600 text is needed."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
build_demo = pytest.importorskip("build_demo")


@pytest.fixture(scope="module")
def results():
    return build_demo.notebook_results(build_demo.NOTEBOOK)


def test_all_metrics_methods_and_levels(results):
    assert set(results["ci"]) == {"anchor", "frr", "bs", "exact", "cer"}
    for metric in results["ci"].values():
        assert {"bertdict", "llm", "bart", "lookup", "bert", "llmart", "none"} <= set(metric)
        for levels in metric.values():
            assert set(levels) == set(build_demo.LEVELS)


def test_values_as_in_the_paper(results):
    assert results["ci"]["anchor"]["bertdict"]["75"] == [0.53, 0.45, 0.61]
    assert {k: v["75"] for k, v in results["sq3"].items() if k in ("bertdict", "llm", "lookup", "bart")} == \
        {"bertdict": 0.45, "llm": 0.18, "lookup": 0.43, "bart": 0.28}


def test_paired_tests(results):
    pairs = {(tuple(t["pair"]), t["level"]) for t in results["tests"]}
    assert len(pairs) == 10
    assert {p for p, _ in pairs} == {("bertdict", "llm"), ("llm", "lookup")}


def test_prediction_files_exist():
    files = build_demo.method_files()
    assert len(files) == len(build_demo.METHODS)
    assert all(f.exists() for f in files.values())


def test_template_has_payload_slot():
    assert (ROOT / "demo" / "template.html").read_text(encoding="utf-8").count("/*__PAYLOAD__*/null") == 1
