import pandas as pd
import pytest

from blnrepair.data import band_quotas, draw_sample, sample_tables

CFG = {"n_test": 30, "n_dev": 6}


@pytest.fixture
def pool():
    bands = ["20-29"] * 60 + ["30-44"] * 40 + ["45-60"] * 20
    rows = [{"id": f"d{i // 4:02d}-{i:03d}", "doc_id": f"d{i // 4:02d}", "text": f"sentence {i}", "n_words": 25,
             "band": b, "fact_idx": [1, 2], "n_facts": 2} for i, b in enumerate(bands)]
    return pd.DataFrame(rows)


def test_quotas_are_proportional_and_sum_to_n(pool):
    quota = band_quotas(pool, 30)
    assert quota.to_dict() == {"20-29": 15, "30-44": 10, "45-60": 5}
    assert band_quotas(pool, 7).sum() == 7


def test_split_sizes_and_band_mix(pool):
    sample = draw_sample(pool, CFG, 3, seed=42)
    assert sample["split"].value_counts().to_dict() == {"test": 30, "dev": 6}
    assert sample[sample["split"] == "test"]["band"].value_counts().to_dict() == {"20-29": 15, "30-44": 10, "45-60": 5}


def test_no_overlap_and_excerpt_cap(pool):
    sample = draw_sample(pool, CFG, 3, seed=42)
    assert sample["id"].is_unique
    assert sample.groupby("doc_id").size().max() <= 3


def test_dev_uses_only_excerpts_that_test_does_not_use(pool):
    sample = draw_sample(pool, CFG, 3, seed=42)
    test_docs = set(sample[sample["split"] == "test"]["doc_id"])
    dev_docs = set(sample[sample["split"] == "dev"]["doc_id"])
    assert dev_docs and not (test_docs & dev_docs)
    assert sample_tables(sample)[1]["excerpts with both test and dev sentences"] == 0
    assert sample[sample["split"] == "dev"]["band"].value_counts().to_dict() == {"20-29": 3, "30-44": 2, "45-60": 1}


def test_same_seed_same_sample_and_pool_order_does_not_matter(pool):
    a = draw_sample(pool, CFG, 3, seed=42)
    b = draw_sample(pool.sample(frac=1, random_state=0), CFG, 3, seed=42)
    assert a[["id", "split"]].sort_values("id").equals(b[["id", "split"]].sort_values("id"))
    c = draw_sample(pool, CFG, 3, seed=7)
    assert set(a["id"]) != set(c["id"])


def test_summary_tables(pool):
    by_band, spread = sample_tables(draw_sample(pool, CFG, 3, seed=42))
    assert by_band.loc["total", "total"] == 36
    assert spread["sentences"] == 36
