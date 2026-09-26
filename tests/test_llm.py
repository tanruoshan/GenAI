"""The SAIA client, tested with a fake API object: no request is sent and no key is needed."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from blnrepair.llm import SaiaClient, load_llm_config, variant_versions

CFG = {**load_llm_config(), "max_retries": 3, "backoff_s": 2}
GOOD = CFG["allowed_models"][0]


def _status_error(code, headers=None):
    req = httpx.Request("POST", "http://saia.test/v1/chat/completions")
    cls = {429: openai.RateLimitError, 400: openai.BadRequestError, 503: openai.InternalServerError}[code]
    return cls("boom", response=httpx.Response(code, request=req, headers=headers), body=None)


class FakeRaw:
    def __init__(self, text, finish="stop"):
        self.headers = {"x-ratelimit-remaining-minute": "41", "content-type": "application/json"}
        self._resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
                                     model="returned-id", usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2))

    def parse(self):
        return self._resp


def fake_api(*results):
    """Each call returns or raises the next item; the last one repeats."""
    queue = list(results)

    def create(**kwargs):
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=create))))


def make(*results):
    sleeps = []
    return SaiaClient(CFG, client=fake_api(*results), sleep=sleeps.append), sleeps


def test_answer_fields_and_rate_limits():
    client, _ = make(FakeRaw('["a"]'))
    out = client.chat(GOOD, [{"role": "user", "content": "hi"}])
    assert out["content"] == '["a"]' and out["finish_reason"] == "stop" and out["model"] == "returned-id"
    assert out["prompt_tokens"] == 7 and out["completion_tokens"] == 2
    assert out["rate_limits"] == {"x-ratelimit-remaining-minute": "41"}


def test_model_id_is_checked_before_any_request():
    client, _ = make(FakeRaw("x"))
    with pytest.raises(AssertionError):
        client.chat("gpt-4o", [{"role": "user", "content": "hi"}])
    assert client.calls == 0


def test_rate_limit_is_retried_with_growing_waits():
    client, sleeps = make(_status_error(429), _status_error(503), FakeRaw("ok"))
    assert client.chat(GOOD, [])["content"] == "ok"
    assert client.calls == 3 and sleeps == [2, 4]


def test_rate_limit_waits_for_the_reported_reset():
    client, sleeps = make(_status_error(429, {"ratelimit-reset": "1500"}), FakeRaw("ok"))
    assert client.chat(GOOD, [])["content"] == "ok" and sleeps == [1501]


def test_timeout_is_retried():
    timeout = openai.APITimeoutError(httpx.Request("POST", "http://saia.test"))
    client, _ = make(timeout, FakeRaw("ok"))
    assert client.chat(GOOD, [])["content"] == "ok" and client.calls == 2


def test_other_errors_are_not_retried():
    client, sleeps = make(_status_error(400), FakeRaw("ok"))
    with pytest.raises(openai.BadRequestError):
        client.chat(GOOD, [])
    assert client.calls == 1 and sleeps == []


def test_retries_stop_at_the_limit():
    client, _ = make(_status_error(429))
    with pytest.raises(openai.RateLimitError):
        client.chat(GOOD, [])
    assert client.calls == CFG["max_retries"] + 1


def test_a_length_stop_is_returned_not_retried():
    client, _ = make(FakeRaw("[", finish="length"))
    assert client.chat(GOOD, [])["finish_reason"] == "length" and client.calls == 1


def test_missing_env_names_the_variable_without_a_value(monkeypatch, tmp_path):
    monkeypatch.setattr("blnrepair.llm.ROOT", tmp_path)  # no .env there
    monkeypatch.delenv("SAIA_BASE_URL", raising=False)
    monkeypatch.setenv("SAIA_API_KEY", "")
    with pytest.raises(RuntimeError) as err:
        SaiaClient(CFG)
    assert "SAIA_BASE_URL" in str(err.value) and "SAIA_API_KEY" in str(err.value)


def test_variant_versions_hold_prompt_model_examples_and_split():
    cfg = {**CFG, "fewshot_picks": [["a-001", "1w"]]}
    dev, test = variant_versions(cfg, "dev"), variant_versions(cfg, "test")
    method, version = dev["few-shot"]
    assert method == "llm_fewshot" and version.startswith(f"v2-{cfg['model']}-ex") and version.endswith("-dev")
    assert test["few-shot"][1].endswith("-test") and test["few-shot"][1][:-4] == version[:-3]
    other = variant_versions({**cfg, "fewshot_picks": [["b-002", "25"]]}, "dev")
    assert other["few-shot"][1] != version and other["probe"] == dev["probe"]
