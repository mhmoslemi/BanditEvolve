"""
The band-bandit evolutionary search engine.

This is the whole algorithm. Each numbered block below maps to one of the nine
steps. The model is frozen throughout; all adaptation lives in two bandits:

  - UCT over parents              (archive.select_parents)
  - per-band Thompson sampling    (PromptBandit) over mutation prompts

SPEED (the TTT way):
  1. Every generation in an iteration is planned first, then issued as ONE
     batched generate() through llm.complete_batch. The GPU is never idled on a
     single-sequence call inside a loop.
  2. The slow part is the sandbox: each candidate runs a real optimizer in a
     subprocess. TTT hid this by evaluating rewards on a CPU ThreadPoolExecutor
     while the GPU worked, so dozens of sandboxes run at once instead of one at
     a time. We do the same: extract + gate in-process (cheap), then fan ALL
     surviving candidates' evaluations out across the pool, collect, then do the
     archive / bandit bookkeeping single-threaded (so the archive is never
     mutated concurrently). run_code writes unique temp files, so concurrent
     sandboxes are safe.
"""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from archive import Archive, State, child_lineage, STERILE, INVALID
from bands import BandAssigner, BandStats
from evaluation import evaluate_candidate, paired_delta
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

        # CPU threadpool for parallel sandbox evaluation (the TTT reward pool).
        nw = int(getattr(cfg, "reward_workers", 0) or 0)
        if nw <= 0:
            nw = max(2, (os.cpu_count() or 8))
        self.reward_workers = nw
        print(f"[init] reward pool: {self.reward_workers} parallel sandboxes",
              flush=True)

    # ---------------------------------------------------------------- run
    def run(self) -> Optional[State]:
        self._bootstrap_seeds()                          # step 1
        for it in range(self.cfg.num_iters):
            self._iteration(it)                          # steps 2-9
        return self.archive.best_state()

    # --------------------------------------------- parallel eval helper
    def _eval_many(self, jobs):
        """jobs: list of (key, code, seeds, parent_ctx).
        Returns {key: EvalResult}, all sandboxes run concurrently."""
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
                except Exception as e:  # a single bad sandbox must not kill the run
                    from evaluation import EvalResult
                    results[key] = EvalResult(valid=False, msg=f"eval_exc:{e}")
        return results

    # ----------------------------------------------------- step 1: seeds
    def _bootstrap_seeds(self):
        """Generate `num_seeds` valid seeds. Problem code seeds first (free).
        The rest are generated in BATCHES and EVALUATED IN PARALLEL, so the
        bootstrap is not a per-seed serial crawl."""
        target = self.cfg.num_seeds
        made = 0
        print(f"[init] bootstrapping {target} seeds "
              f"({len(self.eval_seeds)}-seed eval, parallel) ...", flush=True)

        # canned code seeds (still parallel-evaluated)
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

        # generated seeds, batched gen + parallel eval
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

            evs = self._eval_many(jobs)                     # PARALLEL
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

        # ---- PLAN every rollout up front (steps 3-6), NO generation yet ----
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

        # ---- ONE batched generate() for the whole iteration ----
        completions = self.llm.complete_batch([p["messages"] for p in plans])

        # ---- Phase A: extract + gate in-process (cheap), collect eval jobs ----
        recs = [None] * len(plans)         # final RolloutRecord per plan
        eval_jobs = []                     # (plan_idx, code, seeds, parent_ctx)
        meta = {}                          # plan_idx -> (code, gate stuff)
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

        # ---- Phase B: evaluate ALL survivors in parallel (the slow part) ----
        evs = self._eval_many(eval_jobs)

        # ---- Phase C: bookkeeping single-threaded (archive + bandit) ----
        for pi, plan in enumerate(plans):
            rec = recs[pi]
            if pi not in meta:             # already finalized (no_code / gate)
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

        # ---- logging ----
        groups = {}
        for plan, rec in zip(plans, recs):
            rollouts.append(rec)
            groups.setdefault(plan["pidx"], []).append(rec)
            print(self._fmt_rollout(plan["pidx"], plan["k"], rec), flush=True)
        for pidx, group in groups.items():
            band = bands_for[pidx] if pidx < len(bands_for) else "?"
            print(self._fmt_group(pidx, band, group), flush=True)

        # ---- reflection (step 9) ----
        print("  reflecting on iteration ...", flush=True)
        ref = reflect(self.llm, self.goal, rollouts)
        if ref is not None:
            self.bandit.add_arm(ref["band"], ref["prompt"], source="reflection")
            print(f"  reflection -> +arm[{ref['band']}]  "
                  f"failure='{ref['failure_mode'][:70]}'")
        else:
            print("  reflection -> no parseable suggestion")

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