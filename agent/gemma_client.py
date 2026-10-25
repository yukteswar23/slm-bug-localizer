"""Rate-limited client for Gemma 4 31B on the free Gemini API tier.

Free-tier limits (AI Studio, checked 2026-10-21): 30 RPM, 16K TPM, 14.4K RPD.
A single request larger than the TPM budget can never be sent, so callers
must keep each prompt under ~9K tokens (the Evidence Buffer's job).

Observed 2026-10-21: Gemma 4 always "thinks" before answering (~75-100 tokens
on trivial prompts); thinking_budget / thinking_level are rejected with 400.
Thoughts count against max_output_tokens, so a small cap returns empty text.
Both 31B and 26B-A4B return intermittent 500s; 31B latency ~20-30 s/call.

Usage:
    pip install google-genai
    put GEMINI_API_KEY=... in .env at the repo root (gitignored)
    python agent/gemma_client.py     # one-call smoke test
"""

import json
import os
import random
import threading
import time
from collections import deque
from datetime import date
from pathlib import Path

from google import genai
from google.genai import errors

def _load_dotenv(path: Path = Path(__file__).resolve().parent.parent / ".env"):
    # Minimal .env reader so the key never has to be typed into a shell.
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-31b-it")
RPM, TPM, RPD = 30, 16_000, 14_400
SAFETY = 0.85  # stay below the hard limits; token estimates are approximate
MAX_PROMPT_TOKENS = int(TPM * SAFETY) - 4_000  # leave room for thinking + reply
USAGE_LOG = Path(os.environ.get("GEMMA_USAGE_LOG", "logs/gemma_usage.jsonl"))


def estimate_tokens(text: str) -> int:
    # ~4 chars/token for English and code; refined by real usage_metadata below.
    return len(text) // 4 + 1


class RateLimiter:
    """Sliding 60 s window over requests and tokens, plus a daily request count."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events = deque()  # (timestamp, tokens)
        self.day, self.day_requests = date.today(), 0

    def acquire(self, tokens: int):
        if tokens > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt ~{tokens} tokens exceeds per-call cap {MAX_PROMPT_TOKENS}")
        while True:
            with self.lock:
                now = time.time()
                if date.today() != self.day:
                    self.day, self.day_requests = date.today(), 0
                if self.day_requests >= RPD * SAFETY:
                    raise RuntimeError("daily request budget used; resume tomorrow or switch to Nebius")
                while self.events and now - self.events[0][0] > 60:
                    self.events.popleft()
                used = sum(t for _, t in self.events)
                if len(self.events) < RPM * SAFETY and used + tokens <= TPM * SAFETY:
                    self.events.append((now, tokens))
                    self.day_requests += 1
                    return
                wait = 60 - (now - self.events[0][0]) + 0.1
            time.sleep(max(wait, 0.5))

    def correct(self, estimated: int, actual: int):
        # Replace the estimate with the real count so the window stays accurate.
        with self.lock:
            for i in range(len(self.events) - 1, -1, -1):
                if self.events[i][1] == estimated:
                    self.events[i] = (self.events[i][0], actual)
                    return


class GemmaClient:
    def __init__(self, api_key: str | None = None):
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.limiter = RateLimiter()
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str, system: str = "", max_output_tokens: int = 4_000,
                 temperature: float = 0.7, tag: str = "") -> str:
        # System text is prepended to the prompt; Gemma models on the Gemini API
        # have not always accepted system_instruction.
        full = f"{system}\n\n{prompt}" if system else prompt
        est = estimate_tokens(full) + max_output_tokens
        for attempt in range(6):
            self.limiter.acquire(est)
            try:
                t0 = time.time()
                resp = self.client.models.generate_content(
                    model=MODEL,
                    contents=full,
                    config={"max_output_tokens": max_output_tokens, "temperature": temperature},
                )
            except errors.APIError as e:
                if e.code in (429, 500, 503):
                    time.sleep(min(60, 2 ** attempt * 5) + random.random())
                    continue
                raise
            u = resp.usage_metadata
            actual = u.total_token_count or est  # includes thought tokens
            self.limiter.correct(est, actual)
            with USAGE_LOG.open("a") as f:
                f.write(json.dumps({
                    "ts": t0, "tag": tag, "model": MODEL, "latency_s": round(time.time() - t0, 2),
                    "prompt_tokens": u.prompt_token_count, "thought_tokens": u.thoughts_token_count,
                    "output_tokens": u.candidates_token_count, "attempt": attempt,
                    "finish": str(resp.candidates[0].finish_reason) if resp.candidates else None,
                }) + "\n")
            if not resp.text:
                # Usually thinking used the whole output budget; retry with more room.
                max_output_tokens = min(max_output_tokens * 2, TPM // 2)
                est = estimate_tokens(full) + max_output_tokens
                continue
            return resp.text
        raise RuntimeError("gave up after repeated rate-limit/server errors")


if __name__ == "__main__":
    c = GemmaClient()
    print(f"model={MODEL} per-call prompt cap={MAX_PROMPT_TOKENS} tokens")
    out = c.generate("Write a Python function that returns the n-th Fibonacci number. Code only.",
                     tag="smoke")
    print(out)
    print("usage:", USAGE_LOG.read_text().splitlines()[-1])
