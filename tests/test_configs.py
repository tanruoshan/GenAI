"""The test-split gate: every track's config has a boolean `frozen` flag (checked by notebooks 2 and 3)."""
import pytest

from blnrepair.bert_repair import load_repair_config
from blnrepair.llm import load_llm_config


@pytest.mark.parametrize("load", [load_repair_config, load_llm_config])
def test_config_has_a_boolean_frozen_flag(load):
    assert isinstance(load()["frozen"], bool)
