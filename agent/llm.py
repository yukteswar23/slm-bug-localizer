"""Local model client: starts llama-server for one GGUF and sends chat requests.

Thinking is turned off (--reasoning off): at 30-60 tok/s on the RTX 2050,
thinking tokens would multiply episode time. One server slot (-np 1) keeps
VRAM low and lets llama.cpp reuse the cached prompt prefix between steps.
"""

import json
import subprocess
import time
from pathlib import Path

import requests

LLAMA_DIR = Path("C:/gemma_work/llama.cpp")
MODEL_DIR = Path("C:/gemma_work/models")
LOG_DIR = Path("C:/gemma_work/logs")

MODELS = {  # key -> (GGUF file, context size that fits in 4 GB)
    "gemma-4-e2b": ("gemma-4-E2B_q4_0-it.gguf", 12288),
    "qwen3.5-4b": ("Qwen3.5-4B-Q4_K_M.gguf", 12288),
}


class LlamaServer:
    def __init__(self, model: str, port: int = 8081):
        self.model, self.port = model, port
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None

    def __enter__(self):
        gguf, ctx = MODELS[self.model]
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log = open(LOG_DIR / f"llama-server_{self.model}.log", "w")
        self.proc = subprocess.Popen(
            [str(LLAMA_DIR / "llama-server.exe"), "-m", str(MODEL_DIR / gguf),
             "-ngl", "99", "-fa", "on", "-c", str(ctx), "-np", "1",
             "--reasoning", "off", "--reasoning-format", "none", "--port", str(self.port), "--no-webui"],
            stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited; see {self.log.name}")
            try:
                if requests.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return self
            except requests.ConnectionError:
                pass
            time.sleep(1)
        raise TimeoutError("llama-server did not become healthy")

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=30)
        self.log.close()

    def chat(self, messages: list, schema: dict = None, max_tokens: int = 512) -> dict:
        """Returns {"text", "reasoning", "prompt_tokens", "completion_tokens", "prompt_ms", "gen_ms",
        "cached_tokens"}. With `schema`, output is grammar-constrained to it."""
        # The chat endpoint applies the grammar lazily, after any thinking block, and
        # Gemma 4 opens its thought channel on its own when it sees a "thought" key.
        # Rendering the template ourselves and calling /completion constrains the
        # output from the first token, for every model.
        r = requests.post(f"{self.url}/apply-template", json={"messages": messages}, timeout=60)
        r.raise_for_status()
        body = {"prompt": r.json()["prompt"], "temperature": 0.0, "seed": 0,
                "n_predict": max_tokens, "cache_prompt": True}
        if schema:
            body["json_schema"] = schema
        r = requests.post(f"{self.url}/completion", json=body, timeout=600)
        r.raise_for_status()
        d = r.json()
        t = d.get("timings", {})
        return {"text": d.get("content") or "", "reasoning": "",
                "finish_reason": "length" if d.get("stop_type") == "limit" else "stop",
                "prompt_tokens": d.get("tokens_evaluated", 0),
                "completion_tokens": d.get("tokens_predicted", 0),
                "cached_tokens": t.get("cache_n", 0),
                "prompt_ms": t.get("prompt_ms", 0.0), "gen_ms": t.get("predicted_ms", 0.0)}


class ApiLLM:
    """Gemma 4 31B on the free Gemini API, with the same chat() interface as
    LlamaServer. The API cannot apply our grammar, so episodes must run with
    constrained=False (replies are parsed from text). Gemma 4 on the API always
    thinks; thought tokens count toward the per-minute token limit, so calls are
    slow (~1-2 per minute). Used only for the large-model reference, never as
    part of the offline system.
    """

    def __init__(self, max_output_tokens: int = 4000):
        from . import gemma_client as gc
        gc.MAX_PROMPT_TOKENS = int(gc.TPM * gc.SAFETY)  # check prompt + output against the real TPM limit
        self.gc, self.client = gc, gc.GemmaClient()
        self.max_output = max_output_tokens

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def _log(self, rec: dict):
        with self.gc.USAGE_LOG.open("a") as f:
            f.write(json.dumps({"tag": "agent", "model": self.gc.MODEL, **rec}) + "\n")

    def chat(self, messages: list, schema: dict = None, max_tokens: int = 512) -> dict:
        gc = self.gc
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            text = m["content"]
            if not contents and system:  # system text goes in front of the first user turn
                text = f"{system}\n\n{text}"
            contents.append({"role": "model" if m["role"] == "assistant" else "user",
                             "parts": [{"text": text}]})
        out_cap = self.max_output
        est = gc.estimate_tokens("".join(m["content"] for m in messages)) + out_cap
        for attempt in range(8):
            self.client.limiter.acquire(est)
            t0 = time.time()
            try:
                resp = self.client.client.models.generate_content(
                    model=gc.MODEL, contents=contents,
                    config={"max_output_tokens": out_cap, "temperature": 0.0})
            except gc.errors.APIError as e:
                self._log({"ts": t0, "error": e.code, "attempt": attempt, "est": est,
                           "message": str(e)[:200]})
                if e.code in (429, 500, 503):
                    time.sleep(min(60, 2 ** attempt * 5))
                    continue
                raise
            u = resp.usage_metadata
            self.client.limiter.correct(est, u.total_token_count or est)
            self._log({"ts": t0, "latency_s": round(time.time() - t0, 1), "attempt": attempt, "est": est,
                       "prompt_tokens": u.prompt_token_count, "thought_tokens": u.thoughts_token_count,
                       "output_tokens": u.candidates_token_count, "empty": not resp.text})
            if not resp.text:  # thinking used the whole budget
                continue
            return {"text": resp.text, "reasoning": "", "finish_reason": "stop",
                    "prompt_tokens": u.prompt_token_count or 0,
                    "completion_tokens": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
                    "cached_tokens": 0, "prompt_ms": (time.time() - t0) * 1000, "gen_ms": 0.0}
        raise RuntimeError("API: gave up after repeated errors or empty replies")
