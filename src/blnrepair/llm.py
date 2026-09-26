"""Connection to the GWDG SAIA API (OpenAI-compatible). Plumbing only: nothing here is part of the method.

The base URL and the key come from .env and are never printed. Only model ids listed in
configs/llm.yaml can be called, checked before every request. A request is retried only for a
rate limit (429), a server error (5xx) or a timeout, never because of what an answer says.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv

from blnrepair.data import ROOT


def load_llm_config(path=None):
    return yaml.safe_load(Path(path or ROOT / "configs" / "llm.yaml").read_text(encoding="utf-8"))


def variant_versions(cfg, split):
    """Method and version names of the stored LLM predictions for one split. The version holds the prompt
    version, the model id and a short code of the few-shot picks, so a new prompt, model or example set
    never reads old predictions."""
    tag = hashlib.sha256(json.dumps(cfg["fewshot_picks"]).encode()).hexdigest()[:6]
    model = cfg["model"]
    return {"few-shot": ("llm_fewshot", f"v2-{model}-ex{tag}-{split}"),
            "few-shot + article": ("llm_fewshot_article", f"v1-{model}-ex{tag}-{split}"),
            "probe": ("llm_probe", f"v1-{model}-{split}")}


def _reset_wait(err):
    """Seconds until the rate limit resets, as the server reports it on a 429 (0 if it does not say)."""
    headers = getattr(getattr(err, "response", None), "headers", {}) or {}
    for name in ("retry-after", "ratelimit-reset"):
        try:
            return int(float(headers.get(name, "")))
        except ValueError:
            continue
    return 0


def _retryable(err):
    return isinstance(err, (openai.RateLimitError, openai.APITimeoutError)) or (
        isinstance(err, openai.APIStatusError) and err.status_code >= 500)


class SaiaClient:
    """client and sleep can be replaced in tests; `calls` counts every request sent, retries included."""

    def __init__(self, cfg=None, client=None, sleep=time.sleep):
        self.cfg = cfg or load_llm_config()
        if client is None:
            load_dotenv(ROOT / ".env")
            missing = [k for k in ("SAIA_BASE_URL", "SAIA_API_KEY") if not os.environ.get(k)]
            if missing:
                raise RuntimeError(f"{', '.join(missing)} not set: copy .env.example to .env and fill it in")
            client = openai.OpenAI(base_url=os.environ["SAIA_BASE_URL"], api_key=os.environ["SAIA_API_KEY"],
                                   timeout=self.cfg["timeout_s"], max_retries=0)  # retries are done below
        self.client, self.sleep, self.calls = client, sleep, 0

    def check_model(self, model):
        assert model in self.cfg["allowed_models"], f"model id not allowed: {model!r}"

    def _send(self, send):
        for attempt in range(self.cfg["max_retries"] + 1):
            self.calls += 1
            try:
                return send()
            except (openai.APIStatusError, openai.APITimeoutError) as err:
                if not _retryable(err) or attempt == self.cfg["max_retries"]:
                    raise
                wait = min(self.cfg["backoff_s"] * 2 ** attempt, 60)
                if isinstance(err, openai.RateLimitError):  # wait for the reported reset (the hourly limit can be long)
                    wait = max(wait, min(_reset_wait(err) + 1, self.cfg["max_wait_s"]))
                self.sleep(wait)

    def list_models(self):
        """The model ids the API offers (a request without any BLN600 text)."""
        return sorted(m.id for m in self._send(lambda: self.client.models.list()).data)

    def chat(self, model, messages):
        """One request at temperature 0. Returns the answer text, finish_reason, the model id the API
        reports, token counts and any rate-limit headers the API sends."""
        self.check_model(model)
        raw = self._send(lambda: self.client.chat.completions.with_raw_response.create(
            model=model, messages=messages, temperature=self.cfg["temperature"], max_tokens=self.cfg["max_tokens"]))
        resp = raw.parse()
        return {"content": resp.choices[0].message.content or "",
                "finish_reason": resp.choices[0].finish_reason,
                "model": resp.model,
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "rate_limits": {k: v for k, v in raw.headers.items() if "ratelimit" in k.lower()}}
