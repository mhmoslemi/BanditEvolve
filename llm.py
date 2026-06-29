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

MULTI-GPU: the frozen policy can be replicated across several GPUs (data
parallel, NOT tensor parallel). MultiGPUUnslothLLM spawns one process per GPU,
each loading its own full copy of the model, and complete_batch() shards the
iteration's prompts round-robin across the replicas so they decode concurrently.
Enable it by passing gpu_ids (config gpu_ids / --gpu-ids / env BANDIT_GPUS).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


class BaseLLM:
    def complete(self, messages) -> str:
        return self.complete_batch([messages])[0]

    def complete_batch(self, list_of_messages) -> List[str]:
        raise NotImplementedError


@dataclass
class GenMeta:
    """One generation, with the token ids the RL logprob math needs.

    prompt_ids:     the real (unpadded) prompt token ids for this row.
    completion_ids: the generated token ids, truncated at (and including) EOS.
    Under the frozen / dummy paths the id lists may be empty; only the RL path
    needs them."""
    text: str
    prompt_ids: List[int] = field(default_factory=list)
    completion_ids: List[int] = field(default_factory=list)


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


def _resolve_gpu_ids(cfg):
    """Explicit GPU list from env BANDIT_GPUS or cfg.gpu_ids. Accepts "0,3",
    "0 3", or a YAML list [0, 3]. Empty -> [] (single-GPU path)."""
    raw = os.environ.get("BANDIT_GPUS")
    if raw is None or raw == "":
        raw = getattr(cfg, "gpu_ids", "")
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return [int(x) for x in str(raw).replace(",", " ").split()]


def _gpu_worker(gpu_id, model_kwargs, in_q, out_q, ready_q):
    """One process pinned to a single physical GPU.

    CUDA_VISIBLE_DEVICES is set BEFORE unsloth/torch are imported (UnslothLLM's
    ctor imports them), so this replica sees exactly one device as cuda:0. It
    then loops: pull a (tag, messages) shard, generate, push (tag, texts, err).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        llm = UnslothLLM(**model_kwargs)
    except BaseException:
        import traceback
        ready_q.put(("error", gpu_id, traceback.format_exc()))
        return
    ready_q.put(("ok", gpu_id, None))
    while True:
        job = in_q.get()
        if job is None:                     # shutdown sentinel
            break
        tag, messages = job
        try:
            out_q.put((tag, llm.complete_batch(messages), None))
        except BaseException:
            import traceback
            out_q.put((tag, None, traceback.format_exc()))


class MultiGPUUnslothLLM(BaseLLM):
    """Data-parallel frozen policy: one full model copy per GPU, each in its own
    process. complete_batch() shards prompts round-robin across the replicas and
    reassembles the results in the original order. The weights are frozen, so the
    replicas never need to be synchronised.

    Worker stdout (the per-replica "[llm] loading ..." lines) goes to the
    terminal; it is not threaded back into the main process's run log.
    """

    def __init__(self, gpu_ids, **model_kwargs):
        import atexit
        import multiprocessing as mp

        self.gpu_ids = [int(g) for g in gpu_ids]
        self.n = len(self.gpu_ids)
        ctx = mp.get_context("spawn")        # CUDA requires spawn, not fork
        self.out_q = ctx.Queue()
        ready_q = ctx.Queue()
        self.in_qs, self.procs = [], []

        print(f"[llm] spawning {self.n} replicas on GPUs {self.gpu_ids} ...",
              flush=True)
        for gid in self.gpu_ids:
            in_q = ctx.Queue()
            p = ctx.Process(target=_gpu_worker,
                            args=(gid, model_kwargs, in_q, self.out_q, ready_q),
                            daemon=True)
            p.start()
            self.in_qs.append(in_q)
            self.procs.append(p)

        # block until every replica has loaded its model (or fail loudly)
        for _ in range(self.n):
            status, gid, err = self._get_live(ready_q, timeout=60.0)
            if status == "error":
                self.close()
                raise RuntimeError(f"GPU {gid} replica failed to load:\n{err}")
            print(f"[llm] replica ready on GPU {gid}", flush=True)
        atexit.register(self.close)

    def _get_live(self, q, timeout=120.0):
        """Block on q, but periodically verify the workers are still alive so a
        crashed replica raises instead of hanging the whole run forever."""
        import queue as _queue
        while True:
            try:
                return q.get(timeout=timeout)
            except _queue.Empty:
                dead = [g for g, p in zip(self.gpu_ids, self.procs)
                        if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"GPU replica(s) {dead} died unexpectedly")

    def complete_batch(self, list_of_messages):
        if not list_of_messages:
            return []
        # round-robin so each replica gets a balanced mix of prompt lengths
        shards = [[] for _ in range(self.n)]
        idx_map = [[] for _ in range(self.n)]
        for i, m in enumerate(list_of_messages):
            w = i % self.n
            shards[w].append(m)
            idx_map[w].append(i)

        pending = 0
        for w in range(self.n):
            if shards[w]:
                self.in_qs[w].put((w, shards[w]))
                pending += 1

        results = [None] * len(list_of_messages)
        for _ in range(pending):
            tag, out, err = self._get_live(self.out_q)
            if err is not None:
                raise RuntimeError(
                    f"GPU replica {self.gpu_ids[tag]} generate failed:\n{err}")
            for local_i, text in enumerate(out):
                results[idx_map[tag][local_i]] = text
        return results

    def close(self):
        for in_q in getattr(self, "in_qs", []):
            try:
                in_q.put(None)
            except Exception:
                pass
        for p in getattr(self, "procs", []):
            try:
                p.join(timeout=5)
            except Exception:
                pass


# ============================================================ TRAINABLE (RL)
class TrainableLLM(BaseLLM):
    """One base model, one LoRA adapter per band, one optimizer per adapter.

    Used when cfg.rl_enabled is True (make_llm returns this instead of the frozen
    UnslothLLM / MultiGPUUnslothLLM). It is SINGLE-PROCESS / SINGLE-GPU: training a
    shared base with per-band adapters cannot be data-parallel across the frozen
    replicas, so the multi-GPU path is not used under RL.

    Adapters live on the same wrapped LoRA layers (get_peft_model sets them up;
    add_adapter adds each band's params), so all adapters share Unsloth's
    gradient-checkpointing patch. set_adapter(band) flips which adapter the forward
    uses; disable_adapter() drops to the shared frozen base, the fixed KL reference
    for every band.

    Adapter scope (which bands actually get a trainable adapter):
      * per_band=False                -> ["shared"] (one adapter for all bands)
      * per_band=True, adapter_bands  -> just that subset (e.g. ["good"]); bands
                                         NOT listed run on the frozen base and are
                                         never trained (good-band-only mode)
      * per_band=True, adapter_bands=None -> one adapter per band (full default)

    Correctness notes:
    * No for_inference/for_training toggling. Built training-capable; generate
      under model.eval()+inference_mode, train under model.train(). LoRA dropout is
      forced to 0 so eval and train give identical mappings; the only behavior/
      update logprob difference is the weight change GRPO corrects via the ratio.
    * token_logprobs divides logits by the generation temperature, matching the
      sampling distribution exactly when top_p == 1.0 (the RL config).
    * Memory: never materialize an [L, vocab] tensor. _completion_logits runs the
      trunk and applies the LM head ONLY to the ~comp_len positions predicting
      completion tokens; logsumexp gives the normalizer without a second alloc.
    * Length safety: (prompt + completion) is clamped to max_seq_length, truncating
      the PROMPT from the LEFT so the scored completion is preserved, and targets
      are aligned to the returned logit rows so the gather never size-mismatches.
    """

    def __init__(self, model_name, max_seq_length=8192, load_in_4bit=False,
                 temperature=1.0, top_p=1.0, max_new_tokens=2048,
                 enable_thinking=False, gen_batch_size=8,
                 lora_r=16, lora_alpha=32, lora_dropout=0.0, lr=1e-6, seed=42,
                 bands=None, per_band=True, adapter_bands=None,
                 build_optimizers=True, use_value_head=False, vf_lr=1e-5):
        from unsloth import FastLanguageModel        # MUST precede transformers
        import torch
        from peft import LoraConfig
        self.torch = torch
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.max_seq_length = int(max_seq_length)
        self.enable_thinking = bool(enable_thinking)
        self.gen_batch_size = int(gen_batch_size)
        self.lr = float(lr)

        if bands is None:
            from bands import WEAK, GOOD, ELITE, NEAR_SOTA
            bands = [WEAK, GOOD, ELITE, NEAR_SOTA]
        self.bands = list(bands)
        self.per_band = bool(per_band)
        # which bands actually get a LoRA adapter (see class docstring)
        if not self.per_band:
            self.adapter_names = ["shared"]
        elif adapter_bands:
            allow = set(adapter_bands)
            self.adapter_names = [b for b in self.bands if b in allow]
            if not self.adapter_names:
                raise ValueError(
                    f"adapter_bands={list(adapter_bands)} matched none of the "
                    f"known bands {self.bands}")
        else:
            self.adapter_names = list(self.bands)

        if self.enable_thinking:
            print("[llm][rl] WARNING: enable_thinking=True under RL. The action is "
                  "the ENTIRE generation (think + code); outcome-only GRPO over "
                  "thousands of CoT tokens is memory-prohibitive at 8B and gives "
                  "poor credit assignment, and the code lives at the END so a front "
                  "cap would drop it. Strongly recommend enable_thinking: false for "
                  "RL. The loss is capped at rl_max_completion_tokens regardless.",
                  flush=True)

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        print(f"[llm][rl] loading {model_name} via Unsloth (4bit={load_in_4bit}) "
              f"+ LoRA(r={lora_r}, alpha={lora_alpha}) x {len(self.adapter_names)} "
              f"adapter(s): {self.adapter_names} ...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit, dtype=torch.bfloat16,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=int(lora_r), lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=int(seed), bias="none",
        )
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

        # multi-GPU generation plumbing (set via attach_gen_pool). With no pool,
        # generation runs locally on this GPU. With a pool, the trainer keeps THIS
        # model for the GRPO update/logprobs and generation fans out to the worker
        # GPUs, which reload the adapter whenever _sync_version bumps.
        self.gen_pool = None
        self._adapter_path = None
        self._adapters_dirty = True
        self._sync_version = 0

        # one optimizer per adapter (training process only). Generation-only worker
        # replicas pass build_optimizers=False: they never train, they just load
        # adapter weights and decode.
        self._params = {}
        self.optimizers = {}
        if build_optimizers:
            for name in self.adapter_names:
                ps = self._adapter_params(name)
                if not ps:
                    raise RuntimeError(f"no trainable params for adapter '{name}'")
                self._params[name] = ps
                self.optimizers[name] = torch.optim.AdamW(ps, lr=self.lr)
            model.train()
            tot = sum(sum(p.numel() for p in ps)
                      for ps in self._params.values()) / 1e6
            print(f"[llm][rl] trainable params: {tot:.1f}M across "
                  f"{len(self.adapter_names)} adapter(s) on {self.device}  "
                  f"lr={self.lr}", flush=True)
        else:
            model.eval()
            print(f"[llm][rl] generation replica ready on {self.device} "
                  f"(adapters: {self.adapter_names})", flush=True)

        # ---- A2C critic: a scalar value head over the FROZEN backbone ----
        # Only the training process (build_optimizers) and only rl_algo: a2c build
        # it. The base model is a frozen feature extractor (no grad flows into it);
        # just this linear head trains, so it is cheap and cannot OOM the 8B. V(s)
        # reads the hidden state at the last prompt token (state = mutation prompt).
        self.value_head = None
        self.value_optimizer = None
        if use_value_head and build_optimizers:
            cfg_obj = (getattr(self.model, "config", None)
                       or getattr(getattr(self.model, "base_model", None), "config", None))
            hsz = int(cfg_obj.hidden_size)
            self.value_head = torch.nn.Linear(hsz, 1).to(self.device).float()
            self.value_optimizer = torch.optim.AdamW(
                self.value_head.parameters(), lr=float(vf_lr))
            print(f"[llm][rl] A2C value head (critic): Linear({hsz}->1) on "
                  f"{self.device}  vf_lr={vf_lr}", flush=True)

    # --------------------------------------------------------- adapters
    def _adapter_for(self, band: Optional[str]) -> Optional[str]:
        if not self.per_band:
            return "shared"
        if band in self.adapter_names:
            return band
        return None             # no adapter for this band -> frozen base, untrained

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
        # cap the prompt so prompt + max_new_tokens fits the window; left
        # truncation keeps the most recent prompt tokens, so the captured
        # prompt_ids match what the model conditioned on and token_logprobs can
        # reconstruct the same sequence.
        max_prompt = max(8, self.max_seq_length - self.max_new_tokens)
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_prompt).to(self.device)
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
        """Generate under the band's adapter. band=None (explore/reflection) OR a
        band with NO adapter (good-only mode) => the frozen base.

        If a multi-GPU generation pool is attached (RL training on one GPU, the
        REST of the GPUs generating in parallel), the work is sharded across the
        worker replicas after syncing the current adapter to them; otherwise it
        runs locally on this (training) GPU."""
        if not list_of_messages:
            return []
        if self.gen_pool is not None:
            self._ensure_synced()
            return self.gen_pool.generate(list_of_messages, band,
                                          self._sync_version, self._adapter_path)
        return self._generate_local(list_of_messages, band)

    def _generate_local(self, list_of_messages, band=None):
        prompts = [self._render(m) for m in list_of_messages]
        bs = max(1, self.gen_batch_size)
        chunks = [prompts[s:s + bs] for s in range(0, len(prompts), bs)]
        out = []
        adapter = None if band is None else self._adapter_for(band)
        if adapter is None:
            with self.model.disable_adapter():
                for ch in chunks:
                    out.extend(self._generate_meta_chunks(ch))
        else:
            self._set_adapter(adapter)
            for ch in chunks:
                out.extend(self._generate_meta_chunks(ch))
        return out

    # ---------------------------- multi-GPU generation: adapter sync ----------
    def attach_gen_pool(self, pool, adapter_path):
        """Route generation through `pool` (worker GPUs). adapter_path is the
        scratch file the trainer writes the current adapter weights to whenever
        they change; the workers reload it when the sync version bumps."""
        self.gen_pool = pool
        self._adapter_path = adapter_path
        self._adapters_dirty = True       # force a sync before the first generation

    def _ensure_synced(self):
        if self._adapters_dirty and self._adapter_path is not None:
            self._sync_version += 1
            self.save_adapter_state(self._adapter_path)
            self._adapters_dirty = False

    def save_adapter_state(self, path):
        """Dump every trainable adapter's LoRA weights to `path` (CPU tensors)."""
        from peft import get_peft_model_state_dict
        blob = {}
        for name in self.adapter_names:
            sd = get_peft_model_state_dict(self.model, adapter_name=name)
            blob[name] = {k: v.detach().to("cpu") for k, v in sd.items()}
        self.torch.save(blob, path)

    def load_adapter_state(self, path):
        """Load adapter weights written by save_adapter_state into this model's
        adapters, in place (used by the generation worker replicas)."""
        from peft import set_peft_model_state_dict
        blob = self.torch.load(path, map_location="cpu")
        for name in self.adapter_names:
            if name not in blob:
                continue
            sd = {k: v.to(self.device) for k, v in blob[name].items()}
            set_peft_model_state_dict(self.model, sd, adapter_name=name)

    def complete_batch(self, list_of_messages):
        # reflection / bootstrap: use the base model (band=None)
        return [m.text for m in self.generate_with_meta(list_of_messages, band=None)]

    # --------------------------------------------------------- logprobs
    def _clamp_sequence(self, prompt_ids, completion_ids):
        """Clamp prompt+completion to max_seq_length, truncating the PROMPT from
        the LEFT so the whole completion (the action being scored) is preserved.
        Returns (full_ids_list, prompt_len)."""
        cap = int(self.max_seq_length)
        comp = list(completion_ids)
        prm = list(prompt_ids)
        if len(comp) >= cap:                      # pathological: action longer than window
            comp = comp[:cap - 1] if cap > 1 else comp[:1]
            prm = prm[-1:]                        # keep at least one prompt token
        elif len(prm) + len(comp) > cap:
            keep_prompt = cap - len(comp)
            prm = prm[-keep_prompt:] if keep_prompt > 0 else prm[-1:]
        return prm + comp, len(prm)

    def _completion_logits(self, full_ids, prompt_len, comp_len, with_grad):
        """Logits for exactly the positions that PREDICT completion tokens.
        full_ids: 1D LongTensor [L] = prompt_ids ++ completion_ids. Returns
        [<=comp_len, vocab]; row k is the distribution over token (prompt_len+k).
        Applies the LM head to only those positions, never all L."""
        torch = self.torch
        from contextlib import nullcontext
        ids = full_ids.unsqueeze(0)
        attn = torch.ones_like(ids)
        L = full_ids.shape[0]
        start = max(0, prompt_len - 1)
        end = min(L - 1, start + comp_len)
        sl = slice(start, end)
        ctx = nullcontext() if with_grad else torch.no_grad()
        with ctx:
            try:
                inner = self.model.model              # PEFT CausalLM
                trunk = getattr(inner, "model", inner)  # CausalLM.model = trunk
                head = self.model.get_output_embeddings()
                hidden = trunk(input_ids=ids, attention_mask=attn).last_hidden_state[0]
                return head(hidden[sl])               # [<=comp_len, vocab]
            except Exception:
                logits = self.model(input_ids=ids, attention_mask=attn).logits[0]
                return logits[sl]

    def token_logprobs(self, prompt_ids, completion_ids, band, with_grad=False,
                       use_reference=False):
        """Per-token logprob of each completion token. use_reference => the shared
        base (all adapters disabled), always no-grad. Otherwise the band's adapter
        (or the frozen base if the band has no adapter)."""
        torch = self.torch
        if len(prompt_ids) < 1 or len(completion_ids) < 1:
            return torch.zeros(0, device=self.device,
                               requires_grad=bool(with_grad))
        full_ids, prompt_len = self._clamp_sequence(prompt_ids, completion_ids)
        comp_len = len(full_ids) - prompt_len
        if comp_len < 1:
            return torch.zeros(0, device=self.device,
                               requires_grad=bool(with_grad))
        full = torch.tensor(full_ids, dtype=torch.long, device=self.device)

        def _logp():
            logits = self._completion_logits(full, prompt_len, comp_len, with_grad).float()
            logits = logits / self.temperature
            rows = logits.shape[0]                 # actual number of predicted positions
            tgt_tokens = full[prompt_len:prompt_len + rows]
            tgt = logits.gather(-1, tgt_tokens.unsqueeze(-1)).squeeze(-1)
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
        adapter = self._adapter_for(band)
        if adapter is None:
            # band has no adapter (e.g. good-only mode): score under the frozen
            # base. The GRPO trainer skips these bands, so this is only reached
            # defensively / for non-grad scoring.
            try:
                with self.model.disable_adapter():
                    return _logp() if with_grad else self._no_grad(_logp)
            except (AttributeError, RuntimeError):
                return _logp() if with_grad else self._no_grad(_logp)
        self._set_adapter(adapter)
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
        self._adapters_dirty = True       # weights changed -> re-sync to gen workers
        return float(gn)

    def set_train(self, mode=True):
        self.model.train(bool(mode))

    # ------------------------------------------------- A2C critic (value head)
    def state_value(self, prompt_ids, with_grad=False):
        """Scalar critic value V(s) of a state s = prompt. Reads the hidden state
        at the LAST prompt token from the FROZEN base (adapters disabled, no grad
        into the 8B) and applies the trainable value head. with_grad=True keeps
        the head's graph for the critic's MSE step; otherwise fully detached."""
        torch = self.torch
        if self.value_head is None:
            raise RuntimeError("state_value() needs a value head "
                               "(set rl_algo: a2c so make_llm builds one)")
        cap = int(self.max_seq_length)
        ids_list = list(prompt_ids)[-cap:] or [self.pad_id]
        ids = torch.tensor(ids_list, dtype=torch.long,
                           device=self.device).unsqueeze(0)
        attn = torch.ones_like(ids)
        inner = self.model.model
        trunk = getattr(inner, "model", inner)
        with torch.no_grad():                       # base = frozen feature extractor
            try:
                with self.model.disable_adapter():
                    feat = trunk(input_ids=ids,
                                 attention_mask=attn).last_hidden_state[0, -1]
            except (AttributeError, RuntimeError):
                feat = trunk(input_ids=ids,
                             attention_mask=attn).last_hidden_state[0, -1]
        feat = feat.float()
        if with_grad:
            return self.value_head(feat).squeeze(-1)
        with torch.no_grad():
            return self.value_head(feat).squeeze(-1)

    def value_zero_grad(self):
        if self.value_optimizer is not None:
            self.value_optimizer.zero_grad(set_to_none=True)

    def value_step(self, grad_clip=1.0):
        if self.value_optimizer is None:
            return 0.0
        torch = self.torch
        gn = torch.nn.utils.clip_grad_norm_(self.value_head.parameters(),
                                            float(grad_clip))
        self.value_optimizer.step()
        self.value_optimizer.zero_grad(set_to_none=True)
        return float(gn)


# ---- multi-GPU generation for RL: train on one GPU, generate on the rest ----
def _rl_gen_worker(gpu_id, model_kwargs, in_q, out_q, ready_q):
    """One generation replica pinned to a single physical GPU. Builds a
    generation-only TrainableLLM (no optimizer), then loops: reload the adapter
    when the sync version changes, generate the shard, return the GenMeta list.
    CUDA_VISIBLE_DEVICES is set BEFORE unsloth/torch import (TrainableLLM's ctor
    imports them), so this replica sees exactly one device as cuda:0."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        llm = TrainableLLM(build_optimizers=False, **model_kwargs)
    except BaseException:
        import traceback
        ready_q.put(("error", gpu_id, traceback.format_exc()))
        return
    ready_q.put(("ok", gpu_id, None))
    last_version = -1
    while True:
        job = in_q.get()
        if job is None:                     # shutdown sentinel
            break
        tag, messages, band, version, adapter_path = job
        try:
            if adapter_path and version != last_version:
                llm.load_adapter_state(adapter_path)
                last_version = version
            metas = llm._generate_local(messages, band)
            out_q.put((tag, metas, None))
        except BaseException:
            import traceback
            out_q.put((tag, None, traceback.format_exc()))


class RLGenerationPool:
    """Data-parallel rollout generation for RL: one generation replica per worker
    GPU, each holding the same LoRA adapters as the trainer. The trainer keeps its
    own model on a SEPARATE GPU and does the GRPO update; this pool only generates.

    Each generate() shards the prompts round-robin across the workers; when the
    trainer's adapter has changed (new sync version) the workers reload it from
    `adapter_path` before decoding. The trainer recomputes behavior/reference
    logprobs itself, so GRPO stays exact regardless of which GPU generated."""

    def __init__(self, gpu_ids, model_kwargs):
        import atexit
        import multiprocessing as mp
        self.gpu_ids = [int(g) for g in gpu_ids]
        self.n = len(self.gpu_ids)
        ctx = mp.get_context("spawn")        # CUDA requires spawn, not fork
        self.out_q = ctx.Queue()
        ready_q = ctx.Queue()
        self.in_qs, self.procs = [], []
        print(f"[llm][rl] spawning {self.n} generation replica(s) on GPUs "
              f"{self.gpu_ids} ...", flush=True)
        for gid in self.gpu_ids:
            in_q = ctx.Queue()
            p = ctx.Process(target=_rl_gen_worker,
                            args=(gid, model_kwargs, in_q, self.out_q, ready_q),
                            daemon=True)
            p.start()
            self.in_qs.append(in_q)
            self.procs.append(p)
        for _ in range(self.n):
            status, gid, err = self._get_live(ready_q, timeout=600.0)
            if status == "error":
                self.shutdown()
                raise RuntimeError(f"GPU {gid} generation replica failed:\n{err}")
            print(f"[llm][rl] generation replica ready on GPU {gid}", flush=True)
        atexit.register(self.shutdown)

    def _get_live(self, q, timeout=120.0):
        import queue as _queue
        while True:
            try:
                return q.get(timeout=timeout)
            except _queue.Empty:
                dead = [g for g, p in zip(self.gpu_ids, self.procs)
                        if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"generation replica(s) {dead} died")

    def generate(self, list_of_messages, band, version, adapter_path):
        """Shard prompts round-robin, decode on each worker, reassemble in order.
        Returns a list of GenMeta aligned to list_of_messages."""
        if not list_of_messages:
            return []
        shards = [[] for _ in range(self.n)]
        idx_map = [[] for _ in range(self.n)]
        for i, m in enumerate(list_of_messages):
            w = i % self.n
            shards[w].append(m)
            idx_map[w].append(i)
        pending = 0
        for w in range(self.n):
            if shards[w]:
                self.in_qs[w].put((w, shards[w], band, version, adapter_path))
                pending += 1
        results = [None] * len(list_of_messages)
        for _ in range(pending):
            tag, metas, err = self._get_live(self.out_q)
            if err is not None:
                raise RuntimeError(
                    f"generation replica {self.gpu_ids[tag]} failed:\n{err}")
            for local_i, meta in enumerate(metas):
                results[idx_map[tag][local_i]] = meta
        return results

    def shutdown(self):
        for in_q in getattr(self, "in_qs", []):
            try:
                in_q.put(None)
            except Exception:
                pass
        for p in getattr(self, "procs", []):
            try:
                p.join(timeout=5)
            except Exception:
                pass


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

    def generate_with_meta(self, list_of_messages, band=None):
        # empty token-id lists: lets the RL loop STRUCTURE run offline; the engine
        # skips the GRPO weight update when the model is not a TrainableLLM.
        return [GenMeta(text=t, prompt_ids=[], completion_ids=[])
                for t in self.complete_batch(list_of_messages)]


def make_llm(cfg):
    backend = (getattr(cfg, "llm_backend", "unsloth") or "unsloth").lower()
    if backend in ("dummy", "offline"):
        return DummyLLM(entrypoint=getattr(cfg, "_entrypoint", "run_packing"))

    # RL flips the policy from the frozen UnslothLLM to a trainable model with
    # per-band LoRA adapters. Training (GRPO update + logprobs) runs on gpu_ids[0];
    # any remaining gpu_ids run data-parallel rollout-generation replicas with the
    # adapter synced each iteration. The frozen multi-GPU path below is not used.
    if getattr(cfg, "rl_enabled", False):
        # adapter scope: an explicit band allowlist (e.g. [good, near_sota]) trains
        # only those bands; the rest route to the frozen base. A non-empty allowlist
        # implies per-band routing.
        from config import resolve_adapter_bands
        per_band = getattr(cfg, "rl_adapter_per_band", True)
        adapter_bands = resolve_adapter_bands(cfg)
        if adapter_bands is not None:
            per_band = True
        tll_kwargs = dict(
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
            per_band=per_band,
            adapter_bands=adapter_bands,
            # a2c needs the learned critic; only the trainer (build_optimizers=True)
            # actually builds it, so the generation workers ignore this.
            use_value_head=(str(getattr(cfg, "rl_algo", "grpo")).lower() == "a2c"),
            vf_lr=getattr(cfg, "rl_vf_lr", 1e-5),
        )
        # RL trains ONE model on gpu_ids[0]; if more GPUs are listed, the REST run
        # data-parallel generation replicas (adapter synced each iteration). The
        # workers pin their own device, so spawn them BEFORE this process claims
        # the training GPU.
        gpu_ids = _resolve_gpu_ids(cfg)
        pool = None
        train_gpu = gen_gpus = None
        if len(gpu_ids) >= 2:
            train_gpu, gen_gpus = gpu_ids[0], gpu_ids[1:]
            pool = RLGenerationPool(gen_gpus, tll_kwargs)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(train_gpu)
        elif len(gpu_ids) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        llm = TrainableLLM(**tll_kwargs)
        if pool is not None:
            import tempfile
            adapter_path = os.path.join(
                tempfile.mkdtemp(prefix="rl_adapter_"), "adapter_sync.pt")
            llm.attach_gen_pool(pool, adapter_path)
            print(f"[llm][rl] training on GPU {train_gpu}; rollout generation "
                  f"parallel across GPUs {gen_gpus}", flush=True)
        return llm

    if backend in ("unsloth", "local", "hf"):
        model_kwargs = dict(
            model_name=cfg.llm_model,
            max_seq_length=getattr(cfg, "max_seq_length", 32000),
            load_in_4bit=getattr(cfg, "load_in_4bit", False),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            enable_thinking=getattr(cfg, "enable_thinking", False),
            gen_batch_size=getattr(cfg, "gen_batch_size", 8),
        )
        gpu_ids = _resolve_gpu_ids(cfg)
        if len(gpu_ids) >= 2:
            return MultiGPUUnslothLLM(gpu_ids=gpu_ids, **model_kwargs)
        if len(gpu_ids) == 1:
            # pin the single replica before unsloth/torch import in UnslothLLM
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return UnslothLLM(**model_kwargs)
    raise ValueError(f"unknown llm_backend '{backend}'")