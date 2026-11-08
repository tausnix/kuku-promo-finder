"""LLM wrapper. One interface, many providers. Swap with --provider / LLM_PROVIDER.

    client = get_client("gemini")            # free tier, default
    client = get_client("groq")              # free tier, OpenAI-compatible
    client = get_client("anthropic", "claude-sonnet-5-5")
    client = get_client("mock")              # offline rules baseline, no key

Adding a provider = subclass LLMClient, implement _call(), add one line to PROVIDERS.
Uses only the standard library (urllib), so there is no SDK version drift.
Every response is cached on disk by a hash of (provider, model, temperature, prompts),
which makes reruns reproducible and free.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config import ROOT


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float = 0.0
    cached: bool = False
    usage: dict = field(default_factory=dict)
    prompt_sha: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _http_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:800]
        err = LLMError(f"HTTP {e.code}: {body}")
        err.status = e.code  # type: ignore[attr-defined]
        err.retry_after = e.headers.get("Retry-After")  # type: ignore[attr-defined]
        raise err from None
    except (urllib.error.URLError, TimeoutError) as e:
        err = LLMError(f"network error: {e}")
        err.status = 0  # type: ignore[attr-defined]
        raise err from None


class LLMClient(ABC):
    provider = "base"
    default_model = ""
    api_key_env: str | None = None
    min_interval_s = 0.0  # polite gap between live calls (free tiers rate-limit hard)

    def __init__(self, model: str | None = None, temperature: float | None = 0.0, max_tokens: int = 8192,
                 timeout: float = 90.0, max_retries: int = 2, cache_dir: Path | None = ROOT / ".llm_cache",
                 use_cache: bool = True, min_interval_s: float | None = None):
        self.model = model or self.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache and self.cache_dir is not None
        if min_interval_s is not None:
            self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self.api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        if self.api_key_env and not self.api_key:
            raise LLMError(f"{self.provider}: set {self.api_key_env} in your environment or .env file")

    # ---- public -------------------------------------------------------------------------
    def generate(self, system: str, user: str) -> LLMResponse:
        sha = hashlib.sha256(json.dumps([self.provider, self.model, self.temperature, system, user]).encode()).hexdigest()
        cache_file = self.cache_dir / f"{sha}.json" if self.cache_dir else None
        if self.use_cache and cache_file and cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return LLMResponse(**{**data, "cached": True, "latency_s": 0.0})

        wait = self.min_interval_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                text, usage = self._call(system, user)
                self._last_call = time.time()
                resp = LLMResponse(text=text, provider=self.provider, model=self.model,
                                   latency_s=round(time.time() - t0, 2), usage=usage, prompt_sha=sha)
                if self.cache_dir:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(resp.to_dict(), ensure_ascii=False), encoding="utf-8")
                return resp
            except LLMError as e:
                self._last_call = time.time()
                last_err = e
                status = getattr(e, "status", None)
                daily_quota = status == 429 and ("perday" in str(e).lower() or "free_tier_requests" in str(e).lower())
                if daily_quota or status not in (0, 408, 429, 500, 502, 503, 504) or attempt == self.max_retries:
                    break
                ra = getattr(e, "retry_after", None)
                delay = float(ra) if ra and str(ra).replace(".", "").isdigit() else min(60, 2 ** (attempt + 1))
                time.sleep(delay + random.uniform(0, 1))
        final = LLMError(f"{self.provider}/{self.model} failed: {last_err}")
        status = getattr(last_err, "status", None)
        # Bad key / unknown model will fail for every episode: let the caller stop the run.
        # 402 = out of credits / billing required.
        # 429 on a daily free-tier quota won't recover until tomorrow, so stop too.
        daily = status == 429 and ("perday" in str(last_err).lower() or "free_tier_requests" in str(last_err).lower())
        final.fatal = status in (401, 402, 403, 404) or daily or "API key" in str(last_err)  # type: ignore[attr-defined]
        raise final

    # ---- provider hook --------------------------------------------------------------------
    @abstractmethod
    def _call(self, system: str, user: str) -> tuple[str, dict]:
        """Return (text, usage dict)."""


class GeminiClient(LLMClient):
    provider = "gemini"
    # 2.5-flash is closed to new keys and 3.8-flash timed out on the free tier (checked 2026-11-08).
    default_model = "gemini-3.5-flash"
    api_key_env = "GEMINI_API_KEY"
    min_interval_s = 6.0  # free tier is ~10 requests/minute for Flash

    def __init__(self, *a, **kw):
        if not os.getenv(self.api_key_env):  # accept other common names for the same key
            for alt in ("GOOGLE_GEMINI_API_KEY", "GOOGLE_API_KEY"):
                if os.getenv(alt):
                    os.environ[self.api_key_env] = os.environ[alt]
                    break
        super().__init__(*a, **kw)

    def _call(self, system, user):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        gen = {"maxOutputTokens": self.max_tokens, "responseMimeType": "application/json"}
        if self.temperature is not None:
            gen["temperature"] = self.temperature
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generationConfig": gen}
        data = _http_post(url, payload, {"x-goog-api-key": self.api_key}, self.timeout)
        cands = data.get("candidates") or []
        if not cands:
            raise LLMError(f"gemini returned no candidates: {json.dumps(data)[:400]}")
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text:
            raise LLMError(f"gemini empty text, finishReason={cands[0].get('finishReason')}")
        return text, data.get("usageMetadata", {})


class OpenAICompatibleClient(LLMClient):
    """OpenAI, Groq, Ollama, OpenRouter, Together... all speak /chat/completions."""
    provider = "openai"
    default_model = "gpt-4.1-mini"
    api_key_env = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1"
    json_mode = True

    def extra_headers(self) -> dict:
        return {}

    def _call(self, system, user):
        payload = {"model": self.model, "max_completion_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        headers.update(self.extra_headers())
        url = f"{self.base_url}/chat/completions"
        for _ in range(3):
            try:
                data = _http_post(url, payload, headers, self.timeout)
                break
            except LLMError as e:
                # Some models/hosts reject temperature or JSON mode: drop the option and retry.
                # The validator still parses and checks the JSON, so nothing is lost.
                msg = str(e).lower()
                if getattr(e, "status", None) in (400, 422) and "temperature" in msg and "temperature" in payload:
                    payload.pop("temperature")
                elif getattr(e, "status", None) in (400, 422) and "response_format" in msg and "response_format" in payload:
                    payload.pop("response_format")
                else:
                    raise
        else:
            raise LLMError("request kept being rejected after dropping optional fields")
        return data["choices"][0]["message"]["content"] or "", data.get("usage", {})


class GroqClient(OpenAICompatibleClient):
    provider = "groq"
    default_model = "llama-3.3-70b-versatile"
    api_key_env = "GROQ_API_KEY"
    base_url = "https://api.groq.com/openai/v1"
    min_interval_s = 2.5


class HuggingFaceClient(OpenAICompatibleClient):
    """Hugging Face Inference Providers router (OpenAI-compatible). Token: huggingface.co/settings/tokens
    with the 'Make calls to Inference Providers' permission. Free accounts get a small monthly credit."""
    provider = "huggingface"
    default_model = "meta-llama/Llama-3.3-70B-Instruct"
    api_key_env = "HUGGING_FACE_API_KEY"
    base_url = "https://router.huggingface.co/v1"
    min_interval_s = 2.0

    def __init__(self, *a, **kw):
        # Accept the standard HF variable names too.
        if not os.getenv(self.api_key_env):
            for alt in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"):
                if os.getenv(alt):
                    os.environ[self.api_key_env] = os.environ[alt]
                    break
        super().__init__(*a, **kw)

    def extra_headers(self) -> dict:
        # Credits bought for an organization: set HF_BILL_TO=<org name> so that org pays.
        org = os.getenv("HF_BILL_TO", "").strip()
        return {"X-HF-Bill-To": org} if org else {}


class OllamaClient(OpenAICompatibleClient):
    provider = "ollama"
    default_model = "llama3.1:8b"
    api_key_env = None
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


class AnthropicClient(LLMClient):
    provider = "anthropic"
    default_model = "claude-sonnet-5-5"
    api_key_env = "ANTHROPIC_API_KEY"

    def _call(self, system, user):
        payload = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                   "messages": [{"role": "user", "content": user}]}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        url = "https://api.anthropic.com/v1/messages"
        try:
            data = _http_post(url, payload, headers, self.timeout)
        except LLMError as e:
            if getattr(e, "status", None) == 400 and "temperature" in str(e) and "temperature" in payload:
                payload.pop("temperature")
                data = _http_post(url, payload, headers, self.timeout)
            else:
                raise
        text = "".join(c.get("text", "") for c in data.get("content", []) if c.get("type") == "text")
        return text, data.get("usage", {})


class MockClient(LLMClient):
    """Offline, deterministic, free. Runs the rules-only baseline on the lines inside the prompt,
    so the whole pipeline (parsing JSON, validation, ranking, exports) can be tested without a key."""
    provider = "mock"
    default_model = "rules-baseline"

    def __init__(self, *a, **kw):
        kw["use_cache"], kw["cache_dir"] = False, None
        super().__init__(*a, **kw)

    def _call(self, system, user):
        from types import SimpleNamespace
        from prompts import parse_rendered_lines
        from signals import heuristic_promos
        from parser import parse_ts
        cur = re.search(r"^CURRENT_EPISODE: (E\d+)", user, re.M)
        dur = re.search(r"^EPISODE_DURATION: (\d+):(\d+\.\d+)", user, re.M)
        max_p = re.search(r"Return up to (\d+) promos", user)
        prefix = (cur.group(1) + "-") if cur else ""
        blocks = []
        for i, d in enumerate(parse_rendered_lines(user)):
            if d["id"].startswith(prefix):
                blocks.append(SimpleNamespace(idx=i, **d))
        if "CANDIDATE_PROMOS" in user:  # side-by-side ranking: rules have no opinion, keep the order shown
            ids = re.findall(r"^\[(E\d+-P\d+)\]", user, re.M)
            return json.dumps({"order": ids, "notes": {}}), {}
        if "validator REJECTED" in user:  # rules baseline can't do better on a retry
            return json.dumps({"promos": []}), {}
        promos = heuristic_promos(blocks, parse_ts(*dur.groups()) if dur else 0.0,
                                  max_promos=int(max_p.group(1)) if max_p else 3)
        return json.dumps({"promos": promos}), {}


PROVIDERS: dict[str, type[LLMClient]] = {
    "gemini": GeminiClient,
    "groq": GroqClient,
    "openai": OpenAICompatibleClient,
    "anthropic": AnthropicClient,
    "huggingface": HuggingFaceClient,
    "ollama": OllamaClient,
    "mock": MockClient,
}
ALIASES = {"hugging_face": "huggingface", "hugging-face": "huggingface", "hf": "huggingface", "claude": "anthropic"}


def normalize_provider(name: str | None) -> str:
    n = (name or "").strip().lower()
    return ALIASES.get(n, n)


def get_client(provider: str | None = None, model: str | None = None, **kw) -> LLMClient:
    env_provider = normalize_provider(os.getenv("LLM_PROVIDER"))
    provider = normalize_provider(provider) or env_provider or "mock"
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider '{provider}'. Options: {', '.join(PROVIDERS)}")
    # LLM_MODEL only applies to the provider named in LLM_PROVIDER, so switching provider never
    # sends e.g. a Gemini model name to Groq.
    if not model and provider == env_provider:
        model = os.getenv("LLM_MODEL") or None
    return PROVIDERS[provider](model=model, **kw)
