"""
The band-bandit evolutionary search engine.

This is the whole algorithm. Each numbered block below maps to one of the nine
steps. By default the model is frozen and all adaptation lives in two bandits:

  - UCT over parents              (archive.select_parents)
  - per-band Thompson sampling    (PromptBandit) over mutation prompts

With cfg.rl_enabled the mutator LLM is ALSO fine-tuned online with GRPO, using a
SEPARATE LoRA adapter per band (see grpo.py and llm.TrainableLLM). The frozen path
(_iteration) is untouched. The RL path (_iteration_rl) changes only the rollout
sampling so that each parent's generations form a clean same-prompt GRPO group,
routes generation through the parent-band's adapter, and trains each band's
adapter on its own groups. Bandit, reflection, UCT, gate, and archive are
unchanged.

SPEED (the TTT way):
  1. Generations are planned first, then issued as batched generate() calls (in RL
     mode, one batch per band under that band's adapter; one base batch for
     explore). The GPU is never idled on a single-sequence call.
  2. The slow part is the sandbox: candidates run a real optimizer in a
     subprocess, so all survivors are evaluated on a CPU ThreadPoolExecutor while
     bookkeeping stays single-threaded.
"""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np # type: ignore

from archive import Archive, State, child_lineage, STERILE, INVALID
from bands import BandAssigner, BandStats, WEAK, GOOD, ELITE, NEAR_SOTA
from evaluation import evaluate_candidate, paired_delta
from grpo import GRPOTrainer, rollout_reward
from mutation import build_mutation_messages
from progress import make_bar
from prompt_bandit import PromptBandit
from prompts import SEED_TEMPLATES
from reflection import reflect
from reward import extract_python_code
from validation import quick_gate, validate_child
from problems.base import ParentContext


@dataclass
class RolloutRecord:
    kind: str                      # "explore" | "mutate"
    band: Optional[str] = None
    arm_idx: Optional[int] = None
    arm_source: Optional[str] = None
    outcome: str = ""
    value: Optional[float] = None
    dmu: Optional[float] = None
    dsigma: Optional[float] = None


def _parent_ctx(s: State) -> ParentContext:
    return ParentContext(code=s.code, value=s.value, raw_score=s.raw_score,
                         construction=s.construction)


class Engine:
    def __init__(self, cfg, problem, llm):
        self.cfg = cfg
        self.problem = problem
        self.llm = llm
        self.archive = Archive(uct_c=cfg.uct_c, max_size=cfg.max_archive,
                               topk_children=cfg.topk_children)
        self.bander = BandAssigner(cfg.q_good, cfg.q_elite, cfg.q_near)
        self.band_stats = BandStats()
        self.bandit = PromptBandit(SEED_TEMPLATES,
                                   rng=np.random.default_rng(cfg.seed))
        self.eval_seeds = list(range(cfg.num_eval_seeds))
        self.rng = random.Random(cfg.seed)
        self.goal = getattr(problem, "goal", problem.metric_name)

        nw = int(getattr(cfg, "reward_workers", 0) or 0)
        if nw <= 0:
            nw = max(2, (os.cpu_count() or 8))
        self.reward_workers = nw
        print(f"[init] reward pool: {self.reward_workers} parallel sandboxes",
              flush=True)

        # ---- RL (GRPO) setup ----
        self.rl_enabled = bool(getattr(cfg, "rl_enabled", False))
        self.trainer = None
        self.rl_buffer = []
        self.rl_group_size = int(getattr(cfg, "rl_group_size", 8))
        self.rl_train_every = int(getattr(cfg, "rl_train_every", 1))
        if self.rl_enabled:
            from llm import TrainableLLM
            if isinstance(llm, TrainableLLM):
                self.trainer = GRPOTrainer(llm, cfg)
                mode = ("per-band adapters" if getattr(cfg, "rl_adapter_per_band", True)
                        else "one shared adapter")
                print(f"[init] RL (GRPO) ON [{mode}]: group_size={self.rl_group_size} "
                      f"train_every={self.rl_train_every} "
                      f"ppo_epochs={cfg.rl_ppo_epochs} lr={cfg.rl_lr} "
                      f"kl={cfg.rl_kl_coef} clip={cfg.rl_clip_eps} "
                      f"max_comp_tok={cfg.rl_max_completion_tokens}", flush=True)
            else:
                print("[init] rl_enabled but the LLM is not trainable "
                      "(dummy backend?). Running the RL loop STRUCTURE with NO "
                      "weight updates.", flush=True)

    # ---- existing arm templates per band, for reflection dedup (step 9) ----
    def _existing_arms_by_band(self):
        return {b: [a.template for a in arms]
                for b, arms in self.bandit.pools.items()}

    # ---------------------------------------------------------------- run
    def run(self) -> Optional[State]:
        self._bootstrap_seeds()                          # step 1
        for it in range(self.cfg.num_iters):
            if self.rl_enabled:
                self._iteration_rl(it)                   # steps 2-9 + GRPO
            else:
                self._iteration(it)                      # steps 2-9
        return self.archive.best_state()

    # --------------------------------------------- parallel eval helper
    def _eval_many(self, jobs):
        results = {}
        if not jobs:
            return results
        with ThreadPoolExecutor(max_workers=self.reward_workers) as pool:
            futs = {}
            for key, code, seeds, pctx in jobs:
                fut = pool.submit(evaluate_candidate, self.problem, code,
                                  seeds, self.cfg.timeout, pctx)
                futs[fut] = key
            for fut in futs:
                key = futs[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    from evaluation import EvalResult
                    results[key] = EvalResult(valid=False, msg=f"eval_exc:{e}")
        return results

    # ----------------------------------------------------- step 1: seeds
    def _bootstrap_seeds(self):
        target = self.cfg.num_seeds
        made = 0
        print(f"[init] bootstrapping {target} seeds "
              f"({len(self.eval_seeds)}-seed eval, parallel) ...", flush=True)

        canned = [ss for ss in self.problem.seed_states()
                  if ss.code and ss.code.strip()]
        if canned:
            jobs = [(i, ss.code, self.eval_seeds, None)
                    for i, ss in enumerate(canned)]
            evs = self._eval_many(jobs)
            for i, ss in enumerate(canned):
                if made >= target:
                    break
                ev = evs.get(i)
                if ev and ev.valid:
                    self.archive.add_seed(State.make(
                        ss.code, ev.value, 0, is_seed=True, raw_score=ev.raw,
                        per_seed=ev.per_seed, construction=ss.construction))
                    made += 1
                    print(f"[init] seed {made}/{target}: canned, "
                          f"value={ev.value:.4f}", flush=True)

        seed_prompt = self.problem.build_seed_prompt()
        batch = max(1, int(getattr(self.cfg, "gen_batch_size", 8)))
        max_rounds = max(target, 1) * self.cfg.seed_max_attempts
        rounds = 0
        while made < target and rounds < max_rounds:
            rounds += 1
            need = target - made
            n = min(batch, max(need, 1))
            print(f"[init] seeds {made}/{target}: gen batch {n} "
                  f"(round {rounds}/{max_rounds}) ...", flush=True)
            completions = self.llm.complete_batch([seed_prompt for _ in range(n)])

            from collections import Counter
            fails = Counter()
            jobs, codes = [], {}
            for j, raw in enumerate(completions):
                code = extract_python_code(raw)
                if code is None:
                    fails["no_code"] += 1
                    snippet = (raw or "").strip().replace("\n", " ")[:160]
                    print(f"[init]   no_code; raw[:160]={snippet!r}", flush=True)
                    continue
                gate = quick_gate(code, self.problem)
                if not gate.ok:
                    fails[f"gate:{gate.reason}"] += 1
                    continue
                jobs.append((j, code, self.eval_seeds, None))
                codes[j] = code

            evs = self._eval_many(jobs)
            for j, code in codes.items():
                if made >= target:
                    break
                ev = evs.get(j)
                if ev and ev.valid:
                    self.archive.add_seed(State.make(
                        code, ev.value, 0, is_seed=True, raw_score=ev.raw,
                        per_seed=ev.per_seed))
                    made += 1
                    print(f"[init]   -> seed {made}/{target} valid, "
                          f"value={ev.value:.4f}", flush=True)
                elif ev is not None:
                    fails[f"eval:{ev.msg[:40]}"] += 1
            if fails:
                print(f"[init]   round {rounds} rejections: {dict(fails)}",
                      flush=True)

        print(f"[init] seeds: {made}/{target} valid "
              f"(archive {self.archive.size()})", flush=True)
        if made == 0:
            raise RuntimeError("could not produce any valid seed; check the "
                               "LLM endpoint and the problem definition")

    # ------------------------------------------------- steps 2-9: one pass
    def _iteration(self, it: int):
        t0 = time.time()
        parents = self.archive.select_parents(self.cfg.num_parents)   # step 2
        valid_values = self.archive.valid_values()
        rollouts: List[RolloutRecord] = []

        print(f"\n[iter {it}] parents picked: {len(parents)}  "
              f"(T={self.archive.T} archive={self.archive.size()})")
        bands_for = []
        for i, (p, info) in enumerate(zip(parents, self.archive.last_picks_info)):
            band = self.bander.assign(p.value, valid_values)
            bands_for.append(band)
            tag = "seed" if info["is_seed"] else "expanded"
            print(f"  parent {i} [{tag:8s}] value={info['value']:.4f} "
                  f"band={band:9s} n={info['n']:<4d} Q={info['Q']:.4f} "
                  f"bonus={info['bonus']:.4f} score={info['score']:.4f}")

        plans = []
        for i, parent in enumerate(parents):
            self.archive.record_expansion(parent, count=self.cfg.rollouts_per_parent)
            band = bands_for[i]
            parent.band = band
            for k in range(self.cfg.rollouts_per_parent):             # step 3
                if self.rng.random() < self.cfg.explore_eps:          # step 4a
                    plans.append({
                        "pidx": i, "k": k, "kind": "explore", "band": None,
                        "parent": None, "arm_idx": None, "arm": None,
                        "messages": self.problem.build_seed_prompt(),
                    })
                else:                                                 # step 4b
                    idx, arm = self.bandit.sample(band)               # step 5
                    gp_code = self.archive.grandparent_code_of(parent)   # step 6
                    best = self.archive.best_state()
                    worst = self.archive.worst_valid_state()
                    messages = build_mutation_messages(
                        self.problem, _parent_ctx(parent), gp_code,
                        best.code if best else "", worst.code if worst else "",
                        band, arm.template, max_chars=self.cfg.max_code_chars)
                    plans.append({
                        "pidx": i, "k": k, "kind": "mutate", "band": band,
                        "parent": parent, "arm_idx": idx, "arm": arm,
                        "messages": messages,
                    })

        completions = self.llm.complete_batch([p["messages"] for p in plans])

        recs = [None] * len(plans)
        eval_jobs = []
        meta = {}
        for pi, (plan, raw) in enumerate(zip(plans, completions)):
            code = extract_python_code(raw)
            if plan["kind"] == "explore":
                rec = RolloutRecord(kind="explore")
                if code is None:
                    rec.outcome = "no_code"
                    recs[pi] = rec
                    continue
                gate = quick_gate(code, self.problem)
                if not gate.ok:
                    self.archive.add_nonparent(
                        State.make(code, 0.0, it, status=INVALID), INVALID)
                    rec.outcome = f"invalid:{gate.reason}"
                    recs[pi] = rec
                    continue
                recs[pi] = rec
                meta[pi] = code
                eval_jobs.append((pi, code, self.eval_seeds, None))
            else:
                parent, band, arm = plan["parent"], plan["band"], plan["arm"]
                rec = RolloutRecord(kind="mutate", band=band,
                                    arm_idx=plan["arm_idx"],
                                    arm_source=arm.source)
                if code is None:
                    self.archive.add_nonparent(
                        State.make("", 0.0, it, parents=child_lineage(parent),
                                   status=INVALID), INVALID)
                    rec.outcome = "no_code"
                    recs[pi] = rec
                    continue
                gp_code = self.archive.grandparent_code_of(parent)
                gate = validate_child(code, parent, gp_code, self.archive,
                                      self.problem, self.cfg)
                if gate.sterile:
                    self.archive.add_nonparent(
                        State.make(code, 0.0, it, parents=child_lineage(parent),
                                   status=STERILE), STERILE)
                    rec.outcome = f"sterile:{gate.reason}"
                    recs[pi] = rec
                    continue
                if gate.invalid:
                    self.archive.add_nonparent(
                        State.make(code, 0.0, it, parents=child_lineage(parent),
                                   status=INVALID), INVALID)
                    rec.outcome = f"invalid:{gate.reason}"
                    recs[pi] = rec
                    continue
                recs[pi] = rec
                meta[pi] = code
                seeds = list(parent.per_seed.keys()) or self.eval_seeds
                eval_jobs.append((pi, code, seeds, _parent_ctx(parent)))

        evs = self._eval_many(eval_jobs)

        for pi, plan in enumerate(plans):
            rec = recs[pi]
            if pi not in meta:
                continue
            code = meta[pi]
            ev = evs.get(pi)
            if plan["kind"] == "explore":
                if ev and ev.valid:
                    self.archive.add_root(State.make(
                        code, ev.value, it, parents=[], raw_score=ev.raw,
                        per_seed=ev.per_seed))
                    rec.outcome, rec.value = "valid", ev.value
                else:
                    self.archive.add_nonparent(
                        State.make(code, 0.0, it, status=INVALID), INVALID)
                    rec.outcome = "invalid:eval"
            else:
                parent, band, idx = plan["parent"], plan["band"], plan["arm_idx"]
                if not (ev and ev.valid):
                    self.archive.add_nonparent(
                        State.make(code, 0.0, it, parents=child_lineage(parent),
                                   status=INVALID), INVALID)
                    rec.outcome = "invalid:eval"
                    continue
                dmu, dsigma = paired_delta(ev.per_seed, parent.per_seed)
                child = State.make(code, ev.value, it,
                                   parents=child_lineage(parent),
                                   raw_score=ev.raw, per_seed=ev.per_seed)
                self.archive.add_child(child)
                self.archive.record_child_reward(parent, ev.value)
                self.band_stats.update(band, dmu)
                self.bandit.update(band, idx, self.band_stats.normalize(band, dmu))
                rec.outcome, rec.value = "valid", ev.value
                rec.dmu, rec.dsigma = dmu, dsigma

        groups = {}
        for plan, rec in zip(plans, recs):
            rollouts.append(rec)
            groups.setdefault(plan["pidx"], []).append(rec)
            print(self._fmt_rollout(plan["pidx"], plan["k"], rec), flush=True)
        for pidx, group in groups.items():
            band = bands_for[pidx] if pidx < len(bands_for) else "?"
            print(self._fmt_group(pidx, band, group), flush=True)

        print("  reflecting on iteration ...", flush=True)
        existing = self._existing_arms_by_band()
        ref = reflect(self.llm, self.goal, rollouts,
                      existing_arms_by_band=existing)
        if ref is not None:
            self.bandit.add_arm(ref["band"], ref["prompt"], source="reflection")
            print(f"  reflection -> +arm[{ref['band']}]  "
                  f"failure='{ref['failure_mode'][:70]}'")
        else:
            print("  reflection -> no parseable suggestion")

        self._log_iter(it, rollouts, time.time() - t0)

    # ------------------------------------------- steps 2-9 + GRPO: RL pass
    def _iteration_rl(self, it: int):
        """RL variant. Structural changes vs _iteration:
          * rollout sampling: per parent we Thompson-sample ONE arm and generate G
            completions from that single prompt -> a clean same-prompt GRPO group
            (correct group-relative baseline, never a size-1 group);
          * generation is batched PER BAND under that band's LoRA adapter (explore
            runs on the shared base);
          * each band's adapter is trained on its own groups.
        Bandit / reflection / UCT / gate / archive are unchanged. Explore rollouts
        stay 1-per-parent and are NOT trained (no delta reward)."""
        t0 = time.time()
        parents = self.archive.select_parents(self.cfg.num_parents)   # step 2
        valid_values = self.archive.valid_values()
        rollouts: List[RolloutRecord] = []
        G = self.rl_group_size

        print(f"\n[iter {it}] (RL) parents picked: {len(parents)}  "
              f"(T={self.archive.T} archive={self.archive.size()})")
        bands_for = []
        for i, (p, info) in enumerate(zip(parents, self.archive.last_picks_info)):
            band = self.bander.assign(p.value, valid_values)
            bands_for.append(band)
            tag = "seed" if info["is_seed"] else "expanded"
            print(f"  parent {i} [{tag:8s}] value={info['value']:.4f} "
                  f"band={band:9s} n={info['n']:<4d} Q={info['Q']:.4f} "
                  f"bonus={info['bonus']:.4f} score={info['score']:.4f}")

        # ---- PLAN: per parent, ONE explore OR a G-sized mutation group ----
        plans = []
        for i, parent in enumerate(parents):
            band = bands_for[i]
            parent.band = band
            if self.rng.random() < self.cfg.explore_eps:              # step 4a
                self.archive.record_expansion(parent, count=1)
                plans.append({
                    "pidx": i, "g": 0, "kind": "explore", "band": None,
                    "parent": None, "arm_idx": None, "arm": None, "gid": None,
                    "messages": self.problem.build_seed_prompt(),
                })
            else:                                                     # step 4b
                self.archive.record_expansion(parent, count=G)
                idx, arm = self.bandit.sample(band)                   # step 5 (one arm)
                gp_code = self.archive.grandparent_code_of(parent)    # step 6
                best = self.archive.best_state()
                worst = self.archive.worst_valid_state()
                messages = build_mutation_messages(
                    self.problem, _parent_ctx(parent), gp_code,
                    best.code if best else "", worst.code if worst else "",
                    band, arm.template, max_chars=self.cfg.max_code_chars)
                for g in range(G):                                    # one GRPO group
                    plans.append({
                        "pidx": i, "g": g, "kind": "mutate", "band": band,
                        "parent": parent, "arm_idx": idx, "arm": arm, "gid": i,
                        "messages": messages,
                    })

        # ---- generation: batch PER BAND under its adapter (None => base/explore) ----
        metas = [None] * len(plans)
        gen_buckets = {}        # band (or None for explore) -> [plan_idx]
        for pi, plan in enumerate(plans):
            key = plan["band"] if plan["kind"] == "mutate" else None
            gen_buckets.setdefault(key, []).append(pi)
        for band_key, idxs in gen_buckets.items():
            msgs = [plans[pi]["messages"] for pi in idxs]
            gm = self.llm.generate_with_meta(msgs, band=band_key)
            for pi, m in zip(idxs, gm):
                metas[pi] = m

        # ---- Phase A: extract + gate, collect eval jobs ----
        recs = [None] * len(plans)
        eval_jobs = []
        code_of = {}
        for pi, (plan, gm) in enumerate(zip(plans, metas)):
            code = extract_python_code(gm.text)
            if plan["kind"] == "explore":
                rec = RolloutRecord(kind="explore")
                if code is None:
                    rec.outcome = "no_code"; recs[pi] = rec; continue
                gate = quick_gate(code, self.problem)
                if not gate.ok:
                    self.archive.add_nonparent(
                        State.make(code, 0.0, it, status=INVALID), INVALID)
                    rec.outcome = f"invalid:{gate.reason}"; recs[pi] = rec; continue
                recs[pi] = rec; code_of[pi] = code
                eval_jobs.append((pi, code, self.eval_seeds, None))
            else:
                parent, band, arm = plan["parent"], plan["band"], plan["arm"]
                rec = RolloutRecord(kind="mutate", band=band,
                                    arm_idx=plan["arm_idx"], arm_source=arm.source)
                if code is None:
                    self.archive.add_nonparent(
                        State.make("", 0.0, it, parents=child_lineage(parent),
                                   status=INVALID), INVALID)
                    rec.outcome = "no_code"; recs[pi] = rec; continue
                gp_code = self.archive.grandparent_code_of(parent)
                gate = validate_child(code, parent, gp_code, self.archive,
                                      self.problem, self.cfg)
                if gate.sterile:
                    self.archive.add_nonparent(
                        State.make(code, 0.0, it, parents=child_lineage(parent),
                                   status=STERILE), STERILE)
                    rec.outcome = f"sterile:{gate.reason}"; recs[pi] = rec; continue
                if gate.invalid:
                    self.archive.add_nonparent(
                        State.make(code, 0.0, it, parents=child_lineage(parent),
                                   status=INVALID), INVALID)
                    rec.outcome = f"invalid:{gate.reason}"; recs[pi] = rec; continue
                recs[pi] = rec; code_of[pi] = code
                seeds = list(parent.per_seed.keys()) or self.eval_seeds
                eval_jobs.append((pi, code, seeds, _parent_ctx(parent)))

        # ---- Phase B: parallel sandbox evaluation ----
        evs = self._eval_many(eval_jobs)

        # ---- Phase C: bookkeeping single-threaded (archive + bandit) ----
        for pi, plan in enumerate(plans):
            rec = recs[pi]
            if pi not in code_of:
                continue
            code = code_of[pi]
            ev = evs.get(pi)
            if plan["kind"] == "explore":
                if ev and ev.valid:
                    self.archive.add_root(State.make(
                        code, ev.value, it, parents=[], raw_score=ev.raw,
                        per_seed=ev.per_seed))
                    rec.outcome, rec.value = "valid", ev.value
                else:
                    self.archive.add_nonparent(
                        State.make(code, 0.0, it, status=INVALID), INVALID)
                    rec.outcome = "invalid:eval"
            else:
                parent, band, idx = plan["parent"], plan["band"], plan["arm_idx"]
                if not (ev and ev.valid):
                    self.archive.add_nonparent(
                        State.make(code, 0.0, it, parents=child_lineage(parent),
                                   status=INVALID), INVALID)
                    rec.outcome = "invalid:eval"
                    continue
                dmu, dsigma = paired_delta(ev.per_seed, parent.per_seed)
                child = State.make(code, ev.value, it,
                                   parents=child_lineage(parent),
                                   raw_score=ev.raw, per_seed=ev.per_seed)
                self.archive.add_child(child)
                self.archive.record_child_reward(parent, ev.value)
                self.band_stats.update(band, dmu)
                self.bandit.update(band, idx, self.band_stats.normalize(band, dmu))
                rec.outcome, rec.value = "valid", ev.value
                rec.dmu, rec.dsigma = dmu, dsigma

        # ---- assemble BANDED GRPO groups, cache logp under each band's adapter ----
        if self.trainer is not None:
            by_gid = {}             # gid -> (band, [(meta, reward), ...])
            for pi, plan in enumerate(plans):
                if plan["kind"] != "mutate":
                    continue
                gid = plan["gid"]
                if gid not in by_gid:
                    by_gid[gid] = (plan["band"], [])
                by_gid[gid][1].append((metas[pi], rollout_reward(recs[pi], self.cfg)))
            banded_groups = [(b, g) for (b, g) in by_gid.values() if len(g) >= 2]
            if banded_groups:
                new_samples = self.trainer.build_samples(banded_groups)
                self.rl_buffer.extend(new_samples)
                bands_seen = sorted({b for (b, _) in banded_groups})
                print(f"  RL: assembled {len(new_samples)} samples from "
                      f"{len(banded_groups)} group(s) over bands {bands_seen}; "
                      f"buffer={len(self.rl_buffer)}", flush=True)

        # ---- logging ----
        by_parent = {}
        for plan, rec in zip(plans, recs):
            rollouts.append(rec)
            by_parent.setdefault(plan["pidx"], []).append(rec)
            print(self._fmt_rollout(plan["pidx"], plan["g"], rec), flush=True)
        for pidx, group in by_parent.items():
            band = bands_for[pidx] if pidx < len(bands_for) else "?"
            print(self._fmt_group(pidx, band, group), flush=True)

        # ---- reflection (step 9), with existing-arm dedup context ----
        print("  reflecting on iteration ...", flush=True)
        existing = self._existing_arms_by_band()
        ref = reflect(self.llm, self.goal, rollouts, existing_arms_by_band=existing)
        if ref is not None:
            self.bandit.add_arm(ref["band"], ref["prompt"], source="reflection")
            print(f"  reflection -> +arm[{ref['band']}]  "
                  f"failure='{ref['failure_mode'][:70]}'")
        else:
            print("  reflection -> no parseable suggestion")

        # ---- GRPO update every train_every iterations (per-band adapters) ----
        if (self.trainer is not None and self.rl_buffer
                and (it + 1) % self.rl_train_every == 0):
            print(f"  GRPO update on {len(self.rl_buffer)} action sequences ...",
                  flush=True)
            stats = self.trainer.update(self.rl_buffer)
            self.rl_buffer = []
            print(f"  GRPO: adapters={stats.get('adapters', 0)} "
                  f"pg={stats.get('pg', 0):+.5f} kl={stats.get('kl', 0):.5f} "
                  f"ratio={stats.get('ratio', 1):.4f} n={stats.get('n', 0)} "
                  f"tok={stats.get('tokens', 0)}", flush=True)
            for adapter, st in stats.get("per", {}).items():
                print(f"    [{adapter:9s}] n={st['n']:<3d} pg={st['pg']:+.5f} "
                      f"kl={st['kl']:.5f} ratio={st['ratio']:.4f} "
                      f"|adv|={st['adv_abs']:.4f} gnorm={st['grad_norm']:.3f}",
                      flush=True)

        self._log_iter(it, rollouts, time.time() - t0)

    # ------------------------------------------------------------- logging
    def _fmt_rollout(self, pidx, k, rec):
        if rec.kind == "explore":
            v = f" value={rec.value:.4f}" if rec.value is not None else ""
            return f"    p{pidx} k{k} explore  -> {rec.outcome}{v}"
        extra = ""
        if rec.dmu is not None:
            extra = (f" dmu={rec.dmu:+.4f} dsigma={rec.dsigma:.4f} "
                     f"value={rec.value:.4f}")
        return (f"    p{pidx} k{k} mutate band={rec.band:9s} "
                f"arm={rec.arm_idx}({rec.arm_source[:4]}) -> {rec.outcome}{extra}")

    def _fmt_group(self, pidx, band, group):
        from collections import Counter
        oc = Counter(r.outcome.split(":")[0] for r in group)
        dmus = [r.dmu for r in group if r.dmu is not None]
        dmu_mean = (sum(dmus) / len(dmus)) if dmus else 0.0
        return (f"  group p{pidx} [{band}]: {dict(oc)}  "
                f"mean_dmu={dmu_mean:+.4f} over {len(dmus)} valid mutations")

    def _log_iter(self, it, rollouts, dt):
        from collections import Counter
        oc = Counter(r.outcome.split(":")[0] for r in rollouts)
        reasons = Counter(r.outcome.split(":", 1)[1] for r in rollouts
                          if ":" in r.outcome)
        best = self.archive.best_state()
        raw = (f" raw={best.raw_score:.6f}"
               if best and best.raw_score is not None else "")
        arms = {b: len(a) for b, a in self.bandit.pools.items()}
        print(f"[iter {it}] outcomes={dict(oc)}  reasons={dict(reasons)}")
        print(f"[iter {it}] best={best.value:.6f}{raw}  archive={self.archive.size()}  "
              f"sterile={self.archive.counts[STERILE]} "
              f"invalid={self.archive.counts[INVALID]}  arms={arms}  "
              f"({dt:.1f}s)")