"""
Problem abstraction for the band-bandit search.

This is the TTT problem ABC reshaped for a frozen-model evolutionary loop. The
reward convention is unchanged: every problem reports higher-is-better, so the
archive, bands, and bandit never need to know whether the underlying metric is
minimized or maximized.

Differences from the TTT version:
  - build_seed_prompt():           used by the explore branch (step 4a) and by
                                    bootstrap (step 1) to write a program from
                                    scratch.
  - preprocess(code, parent, seed): the evaluation seed is threaded in, so a
                                    stochastic search produces genuine per-seed
                                    variation and dsigma is meaningful.
  - static_check(code):            the problem's hard structural constraints,
                                    checked by the validation gate before any
                                    code is run (step 6).
  - run_one(code, parent, seed):   single-seed score; evaluation.py loops it over
                                    the seed set and forms the paired deltas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from reward import extract_python_code
from sandbox import run_code


@dataclass
class ParentContext:
    code: str = ""
    value: float = 0.0
    raw_score: Optional[float] = None
    construction: Optional[list] = None


@dataclass
class Seed:
    code: str = ""
    construction: Optional[list] = None


@dataclass
class ScoreResult:
    valid: bool = False
    reward: float = 0.0          # higher is better
    raw: Optional[float] = None  # true metric (for display)
    msg: str = ""
    stdout: str = ""
    construction: Optional[list] = None


def render_state_context(metric_name, target, parent: ParentContext,
                         maximize=True) -> str:
    direction = "higher is better" if maximize else "lower is better"
    if parent.code and parent.code.strip():
        shown = parent.raw_score if parent.raw_score is not None else parent.value
        return (f"Target {metric_name}: {target} ({direction}).\n"
                f"The current program achieved {metric_name} = {shown:.6f}.\n")
    return (f"Target {metric_name}: {target} ({direction}).\n"
            f"No previous program. Write one from scratch.\n")


class Problem(ABC):
    name = "base"
    entrypoint = "run"
    metric_name = "score"
    maximize = True

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.target = self.cfg.get("target")
        self.fail_score = float(self.cfg.get("fail_score", 0.0))
        self.num_seed_states = int(self.cfg.get("num_seeds", 8))
        self.seed = int(self.cfg.get("seed", 42))

    @property
    def goal(self) -> str:
        d = "maximize" if self.maximize else "minimize"
        return f"{d} the metric '{self.metric_name}' (target {self.target})"

    # ---- subclasses implement ----
    @abstractmethod
    def build_prompt(self, parent: ParentContext) -> List[dict]: ...

    @abstractmethod
    def build_seed_prompt(self) -> List[dict]: ...

    @abstractmethod
    def preprocess(self, code: str, parent: ParentContext, seed: int) -> str: ...

    @abstractmethod
    def score(self, value: Any, stdout: str) -> ScoreResult: ...

    @abstractmethod
    def seed_states(self) -> List[Seed]: ...

    def static_check(self, code: str) -> Tuple[bool, str]:
        """Hard structural constraints checked before running. Default: none."""
        return True, "ok"

    # ---- shared single-seed runner ----
    def run_one(self, code: str, parent: ParentContext, seed: int,
                timeout: float) -> ScoreResult:
        # code is already extracted (no fences) by the caller, but be defensive.
        clean = code if code is not None else ""
        if "```" in clean:
            clean = extract_python_code(clean) or clean
        full = self.preprocess(clean, parent, seed)
        out = run_code(full, entrypoint=self.entrypoint, timeout_s=timeout)
        if not out.get("ok"):
            return ScoreResult(valid=False, reward=self.fail_score,
                               msg=f"run_failed:{out.get('error', '?')}",
                               stdout=out.get("stdout", ""))
        sr = self.score(out.get("value"), out.get("stdout", ""))
        if not sr.stdout:
            sr.stdout = out.get("stdout", "")
        return sr
