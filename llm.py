"""
LLM client for the frozen policy.

Unlike the TTT codebase there is no training, no LoRA, no adapter sync: the model
is a pure generation service. The natural deployment is to serve gpt-oss-120b or
Qwen with vLLM and point an OpenAI-compatible client at it (the same pattern the
TTT reranker used for its judge). A DummyClient is included so the whole pipeline
can be smoke-tested offline without an endpoint.

Interface:  client.complete(messages) -> str
            client.complete_batch(list_of_messages) -> list[str]
"""

from concurrent.futures import ThreadPoolExecutor
import os


class BaseLLM:
    concurrency = 1

    def complete(self, messages) -> str:
        raise NotImplementedError

    def complete_batch(self, list_of_messages):
        n = max(1, int(getattr(self, "concurrency", 1)))
        if n == 1 or len(list_of_messages) <= 1:
            return [self.complete(m) for m in list_of_messages]
        out = [""] * len(list_of_messages)
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = {pool.submit(self.complete, m): i
                    for i, m in enumerate(list_of_messages)}
            for fut in futs:
                i = futs[fut]
                try:
                    out[i] = fut.result()
                except Exception:
                    out[i] = ""
        return out


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, model, base_url="", api_key_env="LLM_API_KEY",
                 temperature=1.0, top_p=1.0, max_tokens=26000,
                 request_timeout_s=600.0, concurrency=8):
        from openai import OpenAI
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout = float(request_timeout_s)
        self.concurrency = int(concurrency)
        key = os.environ.get(api_key_env, "EMPTY")
        self.client = OpenAI(api_key=key, base_url=(base_url or None))

    def complete(self, messages) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, top_p=self.top_p,
            max_tokens=self.max_tokens, timeout=self.timeout,
        )
        return resp.choices[0].message.content or ""


class DummyLLM(BaseLLM):
    """Offline stand-in. Returns a trivial program so the pipeline runs end to
    end without a real model. Useful only for wiring / smoke tests."""

    def __init__(self, entrypoint="run_packing"):
        self.entrypoint = entrypoint

    def complete(self, messages) -> str:
        return (
            "Here is a program.\n```python\n"
            f"def {self.entrypoint}():\n"
            "    raise NotImplementedError('dummy LLM: plug in a real backend')\n"
            "```\n"
        )


def make_llm(cfg):
    backend = (getattr(cfg, "llm_backend", "openai") or "openai").lower()
    if backend in ("dummy", "offline"):
        return DummyLLM(entrypoint=getattr(cfg, "_entrypoint", "run_packing"))
    return OpenAICompatibleLLM(
        model=cfg.llm_model,
        base_url=getattr(cfg, "llm_base_url", ""),
        api_key_env=getattr(cfg, "llm_api_key_env", "LLM_API_KEY"),
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        request_timeout_s=getattr(cfg, "llm_timeout_s", 600.0),
        concurrency=getattr(cfg, "llm_concurrency", 8),
    )
