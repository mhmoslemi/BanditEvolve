"""
Circle packing: pack n circles in the unit square to maximize the sum of radii.

Ported from the TTT codebase to the band-bandit Problem ABC. The validator is
kept byte-for-byte so results stay comparable. The seed is injected into the
sandbox prelude (numpy / random seeded) so a stochastic search produces genuine
per-seed variation, which makes the paired dsigma in step 8 meaningful even
though the scoring of a fixed packing is itself deterministic.
"""

from __future__ import annotations

import inspect
from typing import Any, List

import numpy as np

from problems.base import (
    Problem, ParentContext, ScoreResult, Seed, render_state_context,
)


def validate_packing(centers, radii):
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values present"
    for i in range(n):
        if radii[i] < 0:
            return False, f"Circle {i} negative radius"
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12
                or y - r < -1e-12 or y + r > 1 + 1e-12):
            return False, f"Circle {i} outside unit square"
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False, f"Circles {i},{j} overlap"
    return True, "ok"


_VALIDATOR_SRC = inspect.getsource(validate_packing)


class CirclePacking(Problem):
    name = "circle_packing"
    entrypoint = "run_packing"
    metric_name = "sum of radii"
    maximize = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.num_circles = int(cfg.get("num_circles", 26))
        if self.target is None:
            self.target = 2.636 if self.num_circles == 26 else 2.940

    # ----------------------------------------------------------- prompts
    def _rules(self) -> str:
        n = self.num_circles
        return f"""Rules:
- Define run_packing() -> tuple[np.ndarray, np.ndarray, float] returning
  (centers, radii, sum_radii) with centers shape ({n}, 2) and radii shape ({n},).
- Centers in [0,1]^2, radii nonnegative, no overlaps, inside the unit square.
- scipy, numpy, cvxpy, math are available. Top-level helpers only, no closures,
  no lambdas. No filesystem or network IO.
- Use print() to log progress; your stdout is shown back to you.
Return the final program between ```python and ```."""

    def build_prompt(self, parent: ParentContext) -> List[dict]:
        n = self.num_circles
        ctx = render_state_context(self.metric_name, self.target, parent,
                                   maximize=self.maximize)
        body = (parent.code if (parent.code and parent.code.strip())
                else "(no current program)")
        user = f"""You are an expert in circle packing and computational geometry.
Pack {n} circles in the unit square [0,1]x[0,1] to maximize the sum of radii.

We run this validator (read-only):
```python
{_VALIDATOR_SRC}
```

{ctx}
Current program:
```python
{body}
```

{self._rules()}"""
        return [{"role": "user", "content": user}]

    def build_seed_prompt(self) -> List[dict]:
        return self.build_prompt(ParentContext())

    # ------------------------------------------------------- sandbox glue
    def preprocess(self, code: str, parent: ParentContext, seed: int) -> str:
        prelude = (
            "import numpy as np\n"
            "import math\n"
            "import random\n"
            f"SEED = {int(seed)}\n"
            "np.random.seed(SEED)\n"
            "random.seed(SEED)\n"
            "try:\n"
            "    from scipy.optimize import minimize\n"
            "except ImportError:\n"
            "    minimize = None\n\n"
            + _VALIDATOR_SRC + "\n"
        )
        return prelude + "\n# ---- model code below ----\n" + code

    def score(self, value: Any, stdout: str) -> ScoreResult:
        res = ScoreResult(reward=self.fail_score)
        if not (isinstance(value, tuple) and len(value) == 3):
            res.msg = "bad_return_shape"
            return res
        centers, radii, _ = value
        try:
            centers = np.asarray(centers, dtype=float)
            radii = np.asarray(radii, dtype=float).ravel()
        except (ValueError, TypeError) as e:
            res.msg = f"bad_array:{e}"
            return res
        if (centers.ndim != 2 or centers.shape != (self.num_circles, 2)
                or radii.shape != (self.num_circles,)):
            res.msg = f"bad_shape:{centers.shape},{radii.shape}"
            return res
        valid, msg = validate_packing(centers, radii)
        res.valid, res.msg = valid, msg
        if valid:
            s = float(np.sum(radii))
            res.reward = s
            res.raw = s
        return res

    # --------------------------------------------------------- static gate
    def static_check(self, code: str) -> tuple:
        # cheap structural constraints; the entrypoint presence is checked by
        # the gate itself, so just guard against obviously degenerate output.
        if "return" not in code:
            return False, "no_return"
        return True, "ok"

    # -------------------------------------------------------------- seeds
    def seed_states(self) -> List[Seed]:
        # No canned code: bootstrap will ask the LLM to write seeds from scratch.
        return [Seed(code="") for _ in range(self.num_seed_states)]
