"""
LLM client for the mutator policy.

Two modes:
  - FROZEN (UnslothLLM): model loaded once, used purely for generation. The
    original frozen-policy path; nothing touches the backward pass. Used when
    cfg.rl_enabled is False.
  - TRAINABLE (TrainableLLM): one base model with a SEPARATE LoRA adapter per
    band, plus a per-adapter optimizer, so GRPO can fine-tune a specialized
    mutator for each band (weak / good / elite / near_sota). It still generates
    (the search loop is the environment) and exposes the primitives GRPO needs:
    band-routed generation with token-id capture, per-token logprobs under a
    chosen adapter, the shared reference via adapter-disable, and a per-adapter
    step. Used when cfg.rl_enabled is True.

A DummyLLM is kept so the whole pipeline (including the RL loop STRUCTURE) can be
wired and tested offline without loading a model.

Frozen interface:
    client.complete(messages)            -> str
    client.complete_batch([messages...]) -> list[str]      (real batched generate)

Trainable interface adds:
    client.generate_with_meta([messages...], band=None) -> list[GenMeta]
    client.token_logprobs(prompt_ids, completion_ids, band, with_grad, use_reference)
    client.zero_grad(band) / client.step(band, grad_clip) / client.set_train(mode)
    client._adapter_for(band)            # band -> adapter name (identity, or 'shared')

Import-order note: Unsloth must be imported before transformers / peft. Nothing
on the path before make_llm() pulls in transformers, so importing unsloth inside
the LLM __init__ keeps it first.
"""

from dataclasses import dataclass, field
from typing import List, Optional


class BaseLLM:
    def complete(self, messages) -> str:
        return self.complete_batch([messages])[0]

    def complete_batch(self, list_of_messages) -> List[str]:
        raise NotImplementedError


@dataclass
class GenMeta:
    """One generation, with the token ids needed for the RL logprob math.

    prompt_ids:     the real (unpadded) prompt token ids for this row.
    completion_ids: the generated token ids, truncated at (and including) EOS.
    Under the frozen / dummy paths the id lists may be empty; only the RL path
    needs them."""
    text: str
    prompt_ids: List[int] = field(default_factory=list)
    completion_ids: List[int] = field(default_factory=list)


# ============================================================ FROZEN (original)
class UnslothLLM(BaseLLM):
    """Frozen model loaded once via Unsloth, batched decoder-only generation."""

    def __init__(self, model_name, max_seq_length=32000, load_in_4bit=False,
                 temperature=1.0, top_p=1.0, max_new_tokens=6000,
                 enable_thinking=False, gen_batch_size=8):
        from unsloth import FastLanguageModel        # type: ignore # MUST precede transformers
        import torch # type: ignore
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
        torch = self.torch
        enc = self.tokenizer(prompts, return_tensors="pt",
                             padding=True).to(self.device)
        input_len = enc.input_ids.shape[1]
        with torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=self.temperature, top_p=self.top_p,
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
        torch = self.torch
        results, i, sub = [], 0, max(1, len(prompts))
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


# ============================================================ TRAINABLE (RL)
class TrainableLLM(BaseLLM):
    """One base model, one LoRA adapter per band, one optimizer per adapter.

    Adapters are created on the same wrapped LoRA layers (get_peft_model sets the
    layers up; add_adapter adds each band's params into them), so all adapters
    share the Unsloth gradient-checkpointing patch. set_adapter(band) flips which
    adapter the forward uses; disable_adapter() drops to the shared base, which is
    the fixed KL reference for every band.

    Correctness notes:
    * No for_inference/for_training toggling. The model is built training-capable
      and we generate under model.eval()+inference_mode, train under model.train().
      LoRA dropout is forced to 0 so eval and train give identical mappings; the
      only behavior/update logprob difference is the weight change GRPO corrects
      for via the importance ratio.
    * token_logprobs divides logits by the generation temperature, matching the
      sampling distribution exactly when top_p == 1.0 (the RL config). top_p < 1
      is not modeled in the logprob (a standard, documented approximation).
    * Memory: never materialize an [L, vocab] tensor. _completion_logits runs the
      trunk and applies the LM head ONLY to the ~comp_len positions predicting
      completion tokens; logsumexp gives the normalizer without a second alloc.
    """

    def __init__(self, model_name, max_seq_length=8192, load_in_4bit=False,
                 temperature=1.0, top_p=1.0, max_new_tokens=2048,
                 enable_thinking=False, gen_batch_size=8,
                 lora_r=16, lora_alpha=32, lora_dropout=0.0, lr=1e-6, seed=42,
                 bands=None, per_band=True):
        from unsloth import FastLanguageModel # type: ignore        # MUST precede transformers
        import torch # type: ignore
        from peft import LoraConfig # type: ignore
        self.torch = torch
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.gen_batch_size = int(gen_batch_size)
        self.lr = float(lr)

        if bands is None:
            from bands import WEAK, GOOD, ELITE, NEAR_SOTA
            bands = [WEAK, GOOD, ELITE, NEAR_SOTA]
        self.bands = list(bands)
        self.per_band = bool(per_band)
        self.adapter_names = list(self.bands) if self.per_band else ["shared"]

        if self.enable_thinking:
            print("[llm][rl] WARNING: enable_thinking=True under RL. The action is "
                  "the ENTIRE generation (think + code); outcome-only GRPO over "
                  "thousands of CoT tokens is memory-prohibitive at 8B and gives "
                  "poor credit assignment, and the code lives at the END so a "
                  "front cap would drop it. Strongly recommend enable_thinking: "
                  "false for RL. The loss is capped at rl_max_completion_tokens "
                  "tokens regardless.", flush=True)

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        print(f"[llm][rl] loading {model_name} via Unsloth (4bit={load_in_4bit}) "
              f"+ LoRA(r={lora_r}, alpha={lora_alpha}) x {len(self.adapter_names)} "
              f"adapter(s): {self.adapter_names} ...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit, dtype=torch.bfloat16,
        )
        # set the LoRA layers up (Unsloth path: gradient checkpointing etc.)
        model = FastLanguageModel.get_peft_model(
            model, r=int(lora_r), lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=int(seed), bias="none",
        )
        # add one adapter per band into the same layers, then drop the throwaway
        # "default" created by get_peft_model.
        lcfg = LoraConfig(r=int(lora_r), lora_alpha=int(lora_alpha),
                          lora_dropout=float(lora_dropout), bias="none",
                          target_modules=target_modules, task_type="CAUSAL_LM")
        for name in self.adapter_names:
            if name == "default":
                continue
            try:
                model.add_adapter(name, lcfg)
            except Exception as e:
                raise RuntimeError(
                    f"add_adapter('{name}') failed: {e}. Multi-adapter needs a "
                    f"recent peft/unsloth. Fall back with rl_adapter_per_band: "
                    f"false (one shared adapter).") from e
        self.model = model
        self._active = None
        self._set_adapter(self.adapter_names[0])
        if "default" not in self.adapter_names:
            try:
                model.delete_adapter("default")
            except Exception:
                pass  # harmless if it cannot be removed; it just stays unused

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        gc = getattr(model, "generation_config", None)
        if gc is not None:
            gc.max_length = None
            gc.max_new_tokens = None

        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id or self.eos_id

        # one optimizer per adapter, over that adapter's params (matched by name)
        self._params = {}
        self.optimizers = {}
        for name in self.adapter_names:
            ps = self._adapter_params(name)
            if not ps:
                raise RuntimeError(f"no trainable params for adapter '{name}'")
            self._params[name] = ps
            self.optimizers[name] = torch.optim.AdamW(ps, lr=self.lr)
        model.train()
        tot = sum(sum(p.numel() for p in ps) for ps in self._params.values()) / 1e6
        print(f"[llm][rl] trainable params: {tot:.1f}M across "
              f"{len(self.adapter_names)} adapter(s) on {self.device}  lr={self.lr}",
              flush=True)

    # --------------------------------------------------------- adapters
    def _adapter_for(self, band: Optional[str]) -> str:
        if not self.per_band:
            return "shared"
        if band in self.adapter_names:
            return band
        return self.adapter_names[0]            # defensive fallback

    def _adapter_params(self, name):
        a, b = f"lora_A.{name}.", f"lora_B.{name}."
        return [p for n, p in self.model.named_parameters() if (a in n or b in n)]

    def _set_adapter(self, adapter_name):
        if self._active != adapter_name:
            self.model.set_adapter(adapter_name)
            self._active = adapter_name

    # --------------------------------------------------------- rendering
    def _render(self, messages):
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    # --------------------------------------------------------- generation
    def _gen_batch(self, prompts):
        torch = self.torch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        attn = enc.attention_mask
        input_len = enc.input_ids.shape[1]
        self.model.eval()
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=enc.input_ids, attention_mask=attn,
                max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=self.temperature, top_p=self.top_p,
                pad_token_id=self.pad_id,
            )
        results = []
        for i in range(out.shape[0]):
            row_prompt = enc.input_ids[i][attn[i].bool()].tolist()   # unpadded prompt
            gen = out[i, input_len:].tolist()
            if self.eos_id is not None and self.eos_id in gen:
                gen = gen[: gen.index(self.eos_id) + 1]
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            results.append(GenMeta(text=text, prompt_ids=row_prompt,
                                   completion_ids=gen))
        return results

    def _generate_meta_chunks(self, prompts):
        torch = self.torch
        results, i, sub = [], 0, max(1, len(prompts))
        while i < len(prompts):
            chunk = prompts[i:i + sub]
            try:
                results.extend(self._gen_batch(chunk))
                i += len(chunk)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if sub == 1:
                    raise
                sub = max(1, sub // 2)
                print(f"  [oom] halving generation sub-batch to {sub}", flush=True)
        return results

    def generate_with_meta(self, list_of_messages, band=None):
        """Generate under the band's adapter. band=None => base model (used for
        explore rollouts and reflection), so untrained-base diversity is injected
        rather than a band policy."""
        if not list_of_messages:
            return []
        prompts = [self._render(m) for m in list_of_messages]
        bs = max(1, self.gen_batch_size)
        chunks = [prompts[s:s + bs] for s in range(0, len(prompts), bs)]
        out = []
        if band is None:
            with self.model.disable_adapter():
                for ch in chunks:
                    out.extend(self._generate_meta_chunks(ch))
        else:
            self._set_adapter(self._adapter_for(band))
            for ch in chunks:
                out.extend(self._generate_meta_chunks(ch))
        return out

    def complete_batch(self, list_of_messages):
        # reflection / bootstrap: use the base model (band=None)
        return [m.text for m in self.generate_with_meta(list_of_messages, band=None)]

    # --------------------------------------------------------- logprobs
    def _completion_logits(self, full_ids, prompt_len, with_grad):
        """Logits for exactly the positions that PREDICT completion tokens.
        full_ids: 1D LongTensor [L] = prompt_ids ++ completion_ids. Returns
        [comp_len, vocab]; row k is the distribution over token (prompt_len+k).
        Applies the LM head to only comp_len hidden states, never all L."""
        torch = self.torch
        from contextlib import nullcontext
        ids = full_ids.unsqueeze(0)
        attn = torch.ones_like(ids)
        L = full_ids.shape[0]
        sl = slice(prompt_len - 1, L - 1)
        ctx = nullcontext() if with_grad else torch.no_grad()
        with ctx:
            try:
                inner = self.model.model              # PEFT CausalLM
                trunk = getattr(inner, "model", inner)  # CausalLM.model = trunk
                head = self.model.get_output_embeddings()
                hidden = trunk(input_ids=ids, attention_mask=attn).last_hidden_state[0]
                return head(hidden[sl])               # [comp_len, vocab]
            except Exception:
                logits = self.model(input_ids=ids, attention_mask=attn).logits[0]
                return logits[sl]

    def token_logprobs(self, prompt_ids, completion_ids, band, with_grad=False,
                       use_reference=False):
        """Per-token logprob of each completion token. use_reference => the shared
        base (all adapters disabled), always no-grad. Otherwise the band's
        adapter."""
        torch = self.torch
        prompt_len = len(prompt_ids)
        comp_len = len(completion_ids)
        if prompt_len < 1 or comp_len < 1:
            return torch.zeros(0, device=self.device,
                               requires_grad=bool(with_grad))
        full = torch.tensor(list(prompt_ids) + list(completion_ids),
                            dtype=torch.long, device=self.device)
        targets = full[prompt_len:prompt_len + comp_len]

        def _logp():
            logits = self._completion_logits(full, prompt_len, with_grad).float()
            logits = logits / self.temperature
            tgt = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            logz = torch.logsumexp(logits, dim=-1)
            return tgt - logz

        if use_reference:
            try:
                with self.model.disable_adapter():
                    with torch.no_grad():
                        return _logp()
            except (AttributeError, RuntimeError):
                with torch.no_grad():
                    return _logp()
        self._set_adapter(self._adapter_for(band))
        return _logp() if with_grad else self._no_grad(_logp)

    def _no_grad(self, fn):
        with self.torch.no_grad():
            return fn()

    # --------------------------------------------------------- optim (per band)
    def zero_grad(self, band):
        self.optimizers[self._adapter_for(band)].zero_grad(set_to_none=True)

    def step(self, band, grad_clip=1.0):
        torch = self.torch
        name = self._adapter_for(band)
        self._set_adapter(name)                          # ensure this adapter is active
        gn = torch.nn.utils.clip_grad_norm_(self._params[name], float(grad_clip))
        self.optimizers[name].step()
        self.optimizers[name].zero_grad(set_to_none=True)
        return float(gn)

    def set_train(self, mode=True):
        self.model.train(bool(mode))


# ============================================================ OFFLINE
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
    """Offline stand-in. Returns a valid (mediocre) packing for circle_packing so
    bootstrap and the full loop run without a model; otherwise a parseable stub.
    Mutations come back structurally identical, so they log as sterile (a useful
    check of the gate). generate_with_meta returns empty token-id lists so the RL
    loop STRUCTURE can be exercised offline; the engine skips the GRPO update when
    the model is not trainable. Wiring / smoke tests only."""

    def __init__(self, entrypoint="run_packing"):
        self.entrypoint = entrypoint

    def complete_batch(self, list_of_messages):
        if self.entrypoint == "run_packing":
            return [_DUMMY_PACKING for _ in list_of_messages]
        stub = ("```python\n"
                f"def {self.entrypoint}():\n"
                "    raise NotImplementedError('dummy LLM: plug in Unsloth')\n```")
        return [stub for _ in list_of_messages]

    def generate_with_meta(self, list_of_messages, band=None):
        return [GenMeta(text=t, prompt_ids=[], completion_ids=[])
                for t in self.complete_batch(list_of_messages)]


def make_llm(cfg):
    backend = (getattr(cfg, "llm_backend", "unsloth") or "unsloth").lower()
    if backend in ("dummy", "offline"):
        return DummyLLM(entrypoint=getattr(cfg, "_entrypoint", "run_packing"))

    # RL flips the policy from frozen UnslothLLM to multi-adapter TrainableLLM.
    if getattr(cfg, "rl_enabled", False):
        return TrainableLLM(
            model_name=cfg.llm_model,
            max_seq_length=getattr(cfg, "max_seq_length", 8192),
            load_in_4bit=getattr(cfg, "load_in_4bit", False),
            temperature=cfg.temperature, top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            enable_thinking=getattr(cfg, "enable_thinking", False),
            gen_batch_size=getattr(cfg, "gen_batch_size", 8),
            lora_r=getattr(cfg, "rl_lora_r", 16),
            lora_alpha=getattr(cfg, "rl_lora_alpha", 32),
            lora_dropout=getattr(cfg, "rl_lora_dropout", 0.0),
            lr=getattr(cfg, "rl_lr", 1e-6),
            seed=getattr(cfg, "seed", 42),
            per_band=getattr(cfg, "rl_adapter_per_band", True),
        )

    if backend in ("unsloth", "local", "hf"):
        return UnslothLLM(
            model_name=cfg.llm_model,
            max_seq_length=getattr(cfg, "max_seq_length", 32000),
            load_in_4bit=getattr(cfg, "load_in_4bit", False),
            temperature=cfg.temperature, top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            enable_thinking=getattr(cfg, "enable_thinking", False),
            gen_batch_size=getattr(cfg, "gen_batch_size", 8),
        )
    raise ValueError(f"unknown llm_backend '{backend}'")
