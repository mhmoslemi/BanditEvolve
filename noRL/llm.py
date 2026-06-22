"""
LLM client for the frozen policy.

There is no training, no LoRA, no adapter sync here: the model is loaded once and
used purely for generation. This uses Unsloth's FastLanguageModel for fast local
inference, i.e. the same stack as the TTT codebase minus everything that touches
the backward pass. A DummyLLM is kept so the whole pipeline can be wired and
tested offline without loading a model.

Interface:
    client.complete(messages)            -> str
    client.complete_batch([messages...]) -> list[str]   (real batched generate)

Import-order note: Unsloth must be imported before transformers / peft. Nothing
on the path before make_llm() pulls in transformers, so importing unsloth inside
UnslothLLM.__init__ keeps it first.

PERFORMANCE: generation is genuinely batched here (one generate() over up to
gen_batch_size prompts via left padding), which is what keeps the GPU busy. If a
batch OOMs, the batch size is halved and retried, same as the TTT runner, so the
engine can hand us a big iteration batch without us crashing on it.
"""

from typing import List


class BaseLLM:
    def complete(self, messages) -> str:
        return self.complete_batch([messages])[0]

    def complete_batch(self, list_of_messages) -> List[str]:
        raise NotImplementedError


class UnslothLLM(BaseLLM):
    """Frozen model loaded once via Unsloth, batched decoder-only generation."""

    def __init__(self, model_name, max_seq_length=32000, load_in_4bit=False,
                 temperature=1.0, top_p=1.0, max_new_tokens=6000,
                 enable_thinking=False, gen_batch_size=8):
        from unsloth import FastLanguageModel        # MUST precede transformers
        import torch
        self.torch = torch
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.gen_batch_size = int(gen_batch_size)

        print(f"[llm] loading {model_name} via Unsloth (4bit={load_in_4bit}) ...",
              flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=torch.bfloat16,
        )
        FastLanguageModel.for_inference(model)       # Unsloth fast-inference path
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"              # left-pad for batched gen
        # The model ships a generation_config with max_length set (e.g. 40960).
        # Passing max_new_tokens alongside it triggers a transformers warning and
        # an ambiguous cap. Null it so only our explicit max_new_tokens applies.
        gc = getattr(model, "generation_config", None)
        if gc is not None:
            gc.max_length = None
            gc.max_new_tokens = None
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id or self.eos_id
        print(f"[llm] ready on {self.device}", flush=True)

    def _render(self, messages):
        # enable_thinking=False keeps Qwen3/gpt-oss from spending the whole token
        # budget inside a <think> block before emitting the code fence.
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def _generate_one_batch(self, prompts):
        """Generate completions for a list of prompts in ONE generate() call."""
        torch = self.torch
        enc = self.tokenizer(prompts, return_tensors="pt",
                             padding=True).to(self.device)
        input_len = enc.input_ids.shape[1]           # left-padded: same for all rows
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.pad_id,
            )
        texts = []
        for i in range(out.shape[0]):
            gen = out[i, input_len:].tolist()
            if self.eos_id is not None and self.eos_id in gen:
                gen = gen[: gen.index(self.eos_id) + 1]
            texts.append(self.tokenizer.decode(gen, skip_special_tokens=True))
        return texts

    def _generate(self, prompts):
        """Generate a chunk of prompts, halving the sub-batch on CUDA OOM and
        retrying, so a large iteration batch never hard-crashes the run."""
        torch = self.torch
        results = []
        i = 0
        sub = max(1, len(prompts))
        while i < len(prompts):
            chunk = prompts[i:i + sub]
            try:
                results.extend(self._generate_one_batch(chunk))
                i += len(chunk)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if sub == 1:
                    raise
                sub = max(1, sub // 2)
                print(f"  [oom] halving generation sub-batch to {sub}", flush=True)
        return results

    def complete_batch(self, list_of_messages):
        if not list_of_messages:
            return []
        prompts = [self._render(m) for m in list_of_messages]
        out = []
        bs = max(1, self.gen_batch_size)
        for s in range(0, len(prompts), bs):
            out.extend(self._generate(prompts[s:s + bs]))
        return out


_DUMMY_PACKING = """```python
import numpy as np
def run_packing():
    r = 0.04
    xs = [0.08, 0.24, 0.40, 0.56, 0.72, 0.88]
    ys = [0.08, 0.24, 0.40, 0.56, 0.72]
    pts = [(x, y) for y in ys for x in xs][:26]
    centers = np.array(pts, dtype=float)
    radii = np.full(26, r, dtype=float)
    return centers, radii, float(radii.sum())
```"""


class DummyLLM(BaseLLM):
    """Offline stand-in. For circle_packing it returns a valid (mediocre) packing
    so bootstrap and the full loop run without a model; otherwise a parseable
    stub. Mutations come back structurally identical, so they log as sterile,
    which is itself a useful check of the gate. Wiring / smoke tests only."""

    def __init__(self, entrypoint="run_packing"):
        self.entrypoint = entrypoint

    def complete_batch(self, list_of_messages):
        if self.entrypoint == "run_packing":
            return [_DUMMY_PACKING for _ in list_of_messages]
        stub = ("```python\n"
                f"def {self.entrypoint}():\n"
                "    raise NotImplementedError('dummy LLM: plug in Unsloth')\n```")
        return [stub for _ in list_of_messages]


def make_llm(cfg):
    backend = (getattr(cfg, "llm_backend", "unsloth") or "unsloth").lower()
    if backend in ("dummy", "offline"):
        return DummyLLM(entrypoint=getattr(cfg, "_entrypoint", "run_packing"))
    if backend in ("unsloth", "local", "hf"):
        return UnslothLLM(
            model_name=cfg.llm_model,
            max_seq_length=getattr(cfg, "max_seq_length", 32000),
            load_in_4bit=getattr(cfg, "load_in_4bit", False),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            enable_thinking=getattr(cfg, "enable_thinking", False),
            gen_batch_size=getattr(cfg, "gen_batch_size", 8),
        )
    raise ValueError(f"unknown llm_backend '{backend}'")