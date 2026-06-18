"""
Multi-seed evaluation and paired deltas (step 8).

A candidate is scored on a fixed list of evaluation seeds. The aggregate `value`
(the reward used everywhere else, higher is better) is the mean per-seed reward;
a seed that errors contributes the problem's fail score, so a child that crashes
on some seeds is correctly penalized rather than silently dropped.

A child is always evaluated on the SAME seeds its parent was evaluated on, which
makes the comparison paired: the per-seed delta is child[seed] - parent[seed],
and we report dmu (mean improvement) and dsigma (spread of the improvement). A
healthy dmu with a large dsigma means the child helps on some seeds and hurts on
others, exactly the signal the prompt bandit and the design should care about.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from problems.base import ParentContext


@dataclass
class EvalResult:
    per_seed: Dict[int, float] = field(default_factory=dict)
    value: float = 0.0
    valid: bool = False
    raw: Optional[float] = None
    msg: str = ""


def evaluate_candidate(problem, code: str, seeds: List[int], timeout: float,
                       parent_ctx: Optional[ParentContext] = None) -> EvalResult:
    pc = parent_ctx or ParentContext()
    per_seed: Dict[int, float] = {}
    raws: List[float] = []
    n_valid = 0
    last_msg = ""
    for sd in seeds:
        sr = problem.run_one(code, pc, sd, timeout)
        last_msg = sr.msg
        if sr.valid:
            per_seed[sd] = float(sr.reward)
            if sr.raw is not None:
                raws.append(float(sr.raw))
            n_valid += 1
        else:
            per_seed[sd] = float(problem.fail_score)
    value = float(np.mean(list(per_seed.values()))) if per_seed else float(problem.fail_score)
    return EvalResult(
        per_seed=per_seed,
        value=value,
        valid=(n_valid > 0),
        raw=(float(np.mean(raws)) if raws else None),
        msg=last_msg,
    )


def paired_delta(child_per_seed: Dict[int, float],
                 parent_per_seed: Dict[int, float]) -> Tuple[float, float]:
    shared = [s for s in child_per_seed if s in parent_per_seed]
    if not shared:
        return 0.0, 0.0
    deltas = np.array([child_per_seed[s] - parent_per_seed[s] for s in shared],
                      dtype=float)
    dmu = float(deltas.mean())
    dsigma = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    return dmu, dsigma
