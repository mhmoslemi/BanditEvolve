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

import datetime
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from artifacts import save_rollout
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
        self.bander = BandAssigner(cfg.q_good, cfg.q_elite, cfg.q_near,
                                   absolute=getattr(cfg, "bands_absolute", False),
                                   target=getattr(problem, "target", None))
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

        # per-rollout artifacts root (one dir per run). "" disables it.
        adir = getattr(cfg, "artifacts_dir", "runs")
        if adir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(adir, f"{cfg.problem}_{ts}")
            os.makedirs(self.run_dir, exist_ok=True)
            print(f"[init] saving per-rollout artifacts under {self.run_dir}/",
                  flush=True)
        else:
            self.run_dir = None

        # ---- RL (GRPO) setup; all inert unless cfg.rl_enabled ----
        self.rl_enabled = bool(getattr(cfg, "rl_enabled", False))
        self.trainer = None
        self.rl_buffer = []
        self.rl_group_size = int(getattr(cfg, "rl_group_size", 8))
        self.rl_train_every = int(getattr(cfg, "rl_train_every", 1))
        if self.rl_enabled:
            from llm import TrainableLLM
            if isinstance(llm, TrainableLLM):
                self.trainer = GRPOTrainer(llm, cfg)
                names = getattr(llm, "adapter_names", [])
                if not getattr(llm, "per_band", True):
                    mode = "one shared adapter"
                else:
                    mode = f"adapters on bands {names}"
                print(f"[init] RL (GRPO) ON [{mode}]: group_size={self.rl_group_size} "
                      f"train_every={self.rl_train_every} "
                      f"ppo_epochs={cfg.rl_ppo_epochs} lr={cfg.rl_lr} "
                      f"kl={cfg.rl_kl_coef} clip={cfg.rl_clip_eps} "
                      f"max_comp_tok={cfg.rl_max_completion_tokens}", flush=True)
            else:
                print("[init] rl_enabled but the LLM is not trainable "
                      "(dummy backend?). Running the RL loop STRUCTURE with NO "
                      "weight updates.", flush=True)

    # ---------------------------------------------------------------- run
    def run(self) -> Optional[State]:
        if not self._maybe_resume():                     # warm-start, or...
            self._bootstrap_seeds()                      # step 1 (fresh seeds)
        for it in range(self.cfg.num_iters):
            if self.rl_enabled:
                self._iteration_rl(it)                   # steps 2-9 + GRPO
            else:
                self._iteration(it)                      # steps 2-9
        return self.archive.best_state()

    # ------------------------------------------------- resume / warm-start
    def _maybe_resume(self) -> bool:
        """If cfg.resume, rebuild the archive from a prior run's saved programs and
        skip bootstrap. Returns True iff the archive was warm-started."""
        if not getattr(self.cfg, "resume", False):
            return False
        prior = self._resume_dir()
        if not prior:
            print("[init] resume requested but no prior run found; "
                  "bootstrapping fresh seeds instead", flush=True)
            return False
        n = self._load_archive(prior)
        if n == 0:
            print(f"[init] resume: no valid programs under {prior}; "
                  "bootstrapping fresh seeds instead", flush=True)
            return False
        best = self.archive.best_state()
        bestv = f"{best.value:.4f}" if best else "n/a"
        print(f"[init] RESUME from {prior}: loaded {n} valid program(s) "
              f"(archive {self.archive.size()}, best={bestv})", flush=True)
        return True

    def _resume_dir(self):
        """The prior run dir to resume from: cfg.resume_from if set, else the most
        recent <artifacts_dir>/<problem>_* dir that is not the current run."""
        explicit = getattr(self.cfg, "resume_from", "") or ""
        if explicit:
            return explicit if os.path.isdir(explicit) else None
        import glob
        adir = getattr(self.cfg, "artifacts_dir", "runs") or "runs"
        if not os.path.isdir(adir):
            return None
        cur = os.path.abspath(self.run_dir) if self.run_dir else None
        cands = [d for d in glob.glob(os.path.join(adir, f"{self.cfg.problem}_*"))
                 if os.path.isdir(d) and os.path.abspath(d) != cur]
        cands.sort(key=os.path.getmtime, reverse=True)
        return cands[0] if cands else None

    def _load_archive(self, prior_dir) -> int:
        """Add VALID saved programs under prior_dir to the archive as roots (deduped
        by code). With cfg.resume_top_k > 0, keep only the top-K by value. Returns
        the number loaded. Trusts the saved eval values; does NOT re-run the
        sandbox."""
        import glob
        import json
        seen, items = set(), []          # items: (value, code, raw, per_seed)
        code_files = sorted(glob.glob(os.path.join(prior_dir, "**", "code.py"),
                                      recursive=True))
        for cf in code_files:
            try:
                with open(cf) as f:
                    code = f.read()
            except OSError:
                continue
            if not code.strip() or code in seen:
                continue
            ev = None
            ev_path = os.path.join(os.path.dirname(cf), "eval.json")
            if os.path.exists(ev_path):
                try:
                    with open(ev_path) as f:
                        ev = json.load(f)
                except (OSError, ValueError):
                    ev = None
            evd = (ev or {}).get("eval") or {}
            if not evd.get("valid") or evd.get("value") is None:
                continue
            seen.add(code)
            per_seed = {int(k): float(v)
                        for k, v in (evd.get("per_seed") or {}).items()}
            items.append((float(evd["value"]), code, evd.get("raw"), per_seed))

        top_k = int(getattr(self.cfg, "resume_top_k", 0) or 0)
        if top_k > 0 and len(items) > top_k:
            items.sort(key=lambda t: t[0], reverse=True)
            items = items[:top_k]
        for value, code, raw, per_seed in items:
            self.archive.add_root(State.make(code, value, 0, parents=[],
                                             raw_score=raw, per_seed=per_seed))
        return len(items)

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

    # ------------------------------------------------- artifact dumping
    def _save_iteration(self, it, plans, completions, recs, evs, extracted):
        """Write one dir per rollout for iteration `it`: prompt, response,
        extracted code, and sandbox evaluation. No-ops if artifacts disabled."""
        if not self.run_dir:
            return
        it_dir = os.path.join(self.run_dir, f"iter_{it:03d}")
        for pi, plan in enumerate(plans):
            rollout_dir = os.path.join(
                it_dir, f"rollout_p{plan['pidx']}_k{plan['k']}")
            save_rollout(
                rollout_dir,
                messages=plan["messages"],
                response=completions[pi],
                rec=recs[pi],
                ev=evs.get(pi),
                code=extracted.get(pi),
            )

    def _save_seed_round(self, round_idx, seed_prompt, completions, codes, evs):
        """Capture a bootstrap seed-generation round under seeds/round_<n>/."""
        if not self.run_dir:
            return
        base = os.path.join(self.run_dir, "seeds", f"round_{round_idx:03d}")
        for j, resp in enumerate(completions):
            save_rollout(os.path.join(base, f"gen_{j}"),
                         messages=seed_prompt, response=resp,
                         rec=None, ev=evs.get(j), code=codes.get(j))

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
            self._save_seed_round(rounds, seed_prompt, completions, codes, evs)
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
        meta = {}                          # plan_idx -> code (survivors -> eval)
        extracted = {}                     # plan_idx -> code for EVERY plan w/ code
        for pi, (plan, raw) in enumerate(zip(plans, completions)):
            code = extract_python_code(raw)
            if code is not None:
                extracted[pi] = code
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

        # ---- per-rollout artifacts (prompt + response + sandbox eval) ----
        self._save_iteration(it, plans, completions, recs, evs, extracted)

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
        existing_arms = {b: [a.template for a in arms]
                         for b, arms in self.bandit.pools.items()}
        ref = reflect(self.llm, self.goal, rollouts,
                      existing_arms_by_band=existing_arms)
        if ref is not None:
            self.bandit.add_arm(ref["band"], ref["prompt"], source="reflection")
            print(f"  reflection -> +arm[{ref['band']}]  "
                  f"failure='{ref['failure_mode'][:70]}'")
        else:
            print("  reflection -> no parseable suggestion")

        self._log_iter(it, rollouts, time.time() - t0)

    # ------------------------------------------- steps 2-9 + GRPO: RL pass
    def _iteration_rl(self, it: int):
        """RL variant of _iteration. Structural changes:
          * rollout sampling: per parent we Thompson-sample ONE arm and generate G
            completions from that single prompt -> a clean same-prompt GRPO group;
          * generation is batched PER BAND under that band's LoRA adapter (explore,
            and any band with no adapter, run on the frozen base);
          * each band's adapter trains on its own groups. In good-band-only mode
            only the 'good' band has an adapter, so only it is ever trained.
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
                bands_seen = sorted({s.band for s in new_samples})
                print(f"  RL: assembled {len(new_samples)} samples from "
                      f"{len(banded_groups)} group(s); trained bands {bands_seen}; "
                      f"buffer={len(self.rl_buffer)}", flush=True)

        # ---- per-rollout artifacts (prompt + response + sandbox eval) ----
        if self.run_dir:
            it_dir = os.path.join(self.run_dir, f"iter_{it:03d}")
            for pi, plan in enumerate(plans):
                m = metas[pi]
                save_rollout(
                    os.path.join(it_dir, f"rollout_p{plan['pidx']}_g{plan['g']}"),
                    messages=plan["messages"],
                    response=(m.text if m is not None else ""),
                    rec=recs[pi], ev=evs.get(pi), code=code_of.get(pi))

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
        existing_arms = {b: [a.template for a in arms]
                         for b, arms in self.bandit.pools.items()}
        ref = reflect(self.llm, self.goal, rollouts,
                      existing_arms_by_band=existing_arms)
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