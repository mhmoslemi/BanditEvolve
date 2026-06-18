"""
The band-bandit evolutionary search engine.

This is the whole algorithm. Each numbered block below maps to one of the nine
steps. The model is frozen throughout; all adaptation lives in two bandits:

  - UCT over parents              (archive.select_parents)
  - per-band Thompson sampling    (PromptBandit) over mutation prompts

A note on the interaction, since it is the subtle part: UCT estimates which
parents are worth expanding while the prompt bandit is simultaneously changing
the conditional reward of every parent (the prompt determines child quality).
That is a non-stationary bandit feeding a non-stationary bandit. Q(s) here is the
max child reward (best-outcome, not average), which is the right target for a
discovery objective, but be aware that early UCT statistics are computed over a
mutation operator that is still being tuned.
"""

import random
import time
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

    # ---------------------------------------------------------------- run
    def run(self) -> Optional[State]:
        self._bootstrap_seeds()                          # step 1
        for it in range(self.cfg.num_iters):
            self._iteration(it)                          # steps 2-9
        return self.archive.best_state()

    # ----------------------------------------------------- step 1: seeds
    def _gen_fresh(self) -> Optional[str]:
        messages = self.problem.build_seed_prompt()
        return extract_python_code(self.llm.complete(messages))

    def _bootstrap_seeds(self):
        target = self.cfg.num_seeds
        made = 0
        print(f"[init] bootstrapping {target} seeds "
              f"(each = 1 generation + {len(self.eval_seeds)}-seed eval) ...",
              flush=True)

        # problem-provided code seeds first (free, no generation)
        for ss in self.problem.seed_states():
            if made >= target:
                break
            if not (ss.code and ss.code.strip()):
                continue
            ev = evaluate_candidate(self.problem, ss.code, self.eval_seeds,
                                    self.cfg.timeout)
            if ev.valid:
                self.archive.add_seed(State.make(
                    ss.code, ev.value, 0, is_seed=True, raw_score=ev.raw,
                    per_seed=ev.per_seed, construction=ss.construction))
                made += 1
                print(f"[init] seed {made}/{target}: canned, value={ev.value:.4f}",
                      flush=True)

        attempts = 0
        max_attempts = max(target, 1) * self.cfg.seed_max_attempts
        while made < target and attempts < max_attempts:
            attempts += 1
            print(f"[init] seed {made + 1}/{target}: generating "
                  f"(attempt {attempts}/{max_attempts}) ...", flush=True)
            code = self._gen_fresh()
            if code is None:
                print("[init]   -> no code extracted, retrying", flush=True)
                continue
            gate = quick_gate(code, self.problem)
            if not gate.ok:
                print(f"[init]   -> rejected ({gate.reason}), retrying", flush=True)
                continue
            ev = evaluate_candidate(self.problem, code, self.eval_seeds,
                                    self.cfg.timeout)
            if ev.valid:
                self.archive.add_seed(State.make(
                    code, ev.value, 0, is_seed=True, raw_score=ev.raw,
                    per_seed=ev.per_seed))
                made += 1
                print(f"[init]   -> valid, value={ev.value:.4f}", flush=True)
            else:
                print(f"[init]   -> invalid eval ({ev.msg[:50]}), retrying",
                      flush=True)

        print(f"[init] seeds: {made}/{target} valid "
              f"(archive size {self.archive.size()})", flush=True)
        if made == 0:
            raise RuntimeError("could not produce any valid seed; check the "
                               "LLM endpoint and the problem definition")

    # ------------------------------------------------- steps 2-9: one pass
    def _iteration(self, it: int):
        t0 = time.time()
        parents = self.archive.select_parents(self.cfg.num_parents)   # step 2
        valid_values = self.archive.valid_values()
        rollouts: List[RolloutRecord] = []

        # ---- parent pick table (UCT internals) ----
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

        # ---- rollouts ----
        total = len(parents) * self.cfg.rollouts_per_parent
        bar = make_bar(total, f"iter {it} rollouts")
        try:
            for i, parent in enumerate(parents):
                self.archive.record_expansion(parent, count=self.cfg.rollouts_per_parent)
                band = bands_for[i]
                parent.band = band
                group = []
                for k in range(self.cfg.rollouts_per_parent):         # step 3
                    if self.rng.random() < self.cfg.explore_eps:      # step 4a
                        rec = self._do_explore(it)
                    else:                                             # step 4b
                        rec = self._do_mutate(it, parent, band)
                    group.append(rec)
                    rollouts.append(rec)
                    bar.update(1)
                    bar.write(self._fmt_rollout(i, k, rec))
                bar.write(self._fmt_group(i, band, group))
        finally:
            bar.close()

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

    # ----------------------------------------------------- step 4a: explore
    def _do_explore(self, it: int) -> RolloutRecord:
        rec = RolloutRecord(kind="explore")
        code = self._gen_fresh()
        if code is None:
            rec.outcome = "no_code"
            return rec
        gate = quick_gate(code, self.problem)
        if not gate.ok:
            self.archive.add_nonparent(State.make(code, 0.0, it, status=INVALID),
                                       INVALID)
            rec.outcome = f"invalid:{gate.reason}"
            return rec
        ev = evaluate_candidate(self.problem, code, self.eval_seeds, self.cfg.timeout)
        if ev.valid:
            self.archive.add_root(State.make(
                code, ev.value, it, parents=[], raw_score=ev.raw,
                per_seed=ev.per_seed))
            rec.outcome, rec.value = "valid", ev.value
        else:
            self.archive.add_nonparent(State.make(code, 0.0, it, status=INVALID),
                                       INVALID)
            rec.outcome = "invalid:eval"
        return rec

    # ----------------------------------------------------- step 4b: mutate
    def _do_mutate(self, it: int, parent: State, band: str) -> RolloutRecord:
        idx, arm = self.bandit.sample(band)                            # step 5
        rec = RolloutRecord(kind="mutate", band=band, arm_idx=idx,
                            arm_source=arm.source)

        gp_code = self.archive.grandparent_code_of(parent)             # step 6
        best = self.archive.best_state()
        worst = self.archive.worst_valid_state()
        messages = build_mutation_messages(
            self.problem, _parent_ctx(parent), gp_code,
            best.code if best else "", worst.code if worst else "",
            band, arm.template, max_chars=self.cfg.max_code_chars)

        code = extract_python_code(self.llm.complete(messages))
        if code is None:
            self.archive.add_nonparent(
                State.make("", 0.0, it, parents=child_lineage(parent),
                           status=INVALID), INVALID)
            rec.outcome = "no_code"
            return rec

        gate = validate_child(code, parent, gp_code, self.archive,        # steps 6-7
                              self.problem, self.cfg)
        if gate.sterile:
            self.archive.add_nonparent(
                State.make(code, 0.0, it, parents=child_lineage(parent),
                           status=STERILE), STERILE)
            rec.outcome = f"sterile:{gate.reason}"
            return rec
        if gate.invalid:
            self.archive.add_nonparent(
                State.make(code, 0.0, it, parents=child_lineage(parent),
                           status=INVALID), INVALID)
            rec.outcome = f"invalid:{gate.reason}"
            return rec

        # step 8: evaluate the child on the SAME seeds the parent used
        seeds = list(parent.per_seed.keys()) or self.eval_seeds
        ev = evaluate_candidate(self.problem, code, seeds, self.cfg.timeout,
                                parent_ctx=_parent_ctx(parent))
        if not ev.valid:
            self.archive.add_nonparent(
                State.make(code, 0.0, it, parents=child_lineage(parent),
                           status=INVALID), INVALID)
            rec.outcome = "invalid:eval"
            return rec

        dmu, dsigma = paired_delta(ev.per_seed, parent.per_seed)
        child = State.make(code, ev.value, it, parents=child_lineage(parent),
                           raw_score=ev.raw, per_seed=ev.per_seed)
        self.archive.add_child(child)
        self.archive.record_child_reward(parent, ev.value)

        # bandit reward = within-band standardized improvement (see bands.py)
        self.band_stats.update(band, dmu)
        reward = self.band_stats.normalize(band, dmu)
        self.bandit.update(band, idx, reward)

        rec.outcome, rec.value = "valid", ev.value
        rec.dmu, rec.dsigma = dmu, dsigma
        return rec

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