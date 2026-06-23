"""
Circle packing: pack n circles in the unit square to maximize the sum of radii.

Ported from the TTT codebase to the band-bandit Problem ABC. The validator is
kept byte-for-byte so results stay comparable. The seed is injected into the
sandbox prelude (numpy / random seeded) so a stochastic search produces genuine
per-seed variation, which makes the paired dsigma in step 8 meaningful even
though the scoring of a fixed packing is itself deterministic.

The prompt is deliberately blunt: one job, one function, one fenced block. The
model is told to emit NOTHING but the code block, because anything that wraps
the program in prose or never closes the fence shows up as "no_code" and wastes
a whole generation.

Robustness fixes vs the original:
  1. score() rejects a DEGENERATE packing (sum of radii ~ 0). A program that
     returns all-zero radii passes validate_packing (zero-radius circles never
     overlap and are trivially in-bounds) but is worthless, and a flood of these
     value=0.0 "valid" programs collapses the score bands (every quantile becomes
     0, so every parent lands in one band) and gives the search nothing to climb.
     Treat it as invalid so it never enters the archive.
  2. seed_states() returns ONE canned valid packing (a 6x5 grid, sum ~ 2.16) plus
     empty slots. Bootstrap no longer depends entirely on cold one-shot
     generation clearing the validator, so the archive always has a real nonzero
     anchor to band against and to mutate from. The remaining seeds are still
     generated independently by the LLM (step 1).
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


# A guaranteed-valid canned seed: 6x5 grid, radii = half the smallest spacing
# minus a margin, so nothing overlaps and nothing leaves the square. Sum of radii
# is ~2.16, a real nonzero anchor for the archive / bands / mutations.
_GRID_SEED = '''
import numpy as np

def run_packing():
    n = 26
    cols, rows = 6, 5
    xs = [(i + 0.5) / cols for i in range(cols)]
    ys = [(j + 0.5) / rows for j in range(rows)]
    pts = [(x, y) for y in ys for x in xs][:n]
    centers = np.array(pts, dtype=float)
    r = min(1.0 / (2 * cols), 1.0 / (2 * rows)) - 1e-4
    radii = np.full(n, r, dtype=float)
    return centers, radii, float(radii.sum())
'''


class CirclePacking(Problem):
    name = "circle_packing"
    entrypoint = "run_packing"
    metric_name = "sum of radii"
    maximize = True

    # a packing whose radii sum below this is degenerate (treated as invalid)
    min_sum_radii = 1e-3

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.num_circles = int(cfg.get("num_circles", 26))
        if self.target is None:
            self.target = 2.636 if self.num_circles == 26 else 2.940

    # ----------------------------------------------------------- prompts
    def build_prompt(self, parent: ParentContext) -> List[dict]:
        n = self.num_circles
        has_parent = bool(parent.code and parent.code.strip())

        if has_parent:
            shown = (parent.raw_score if parent.raw_score is not None
                     else parent.value)
            state = (f"Here is the current best program "
                     f"(sum of radii = {shown:.6f}). Improve it:\n\n"
                     f"```python\n{parent.code}\n```\n")
            task = "Write an improved version that achieves a larger sum of radii."
        else:
            state = ""
            task = ("Write a Python program from scratch that finds a good "
                    "packing.")

        user = f"""You are an expert in circle packing and numerical optimization.

Task: pack {n} non-overlapping circles inside the unit square [0,1]x[0,1] and
maximize the sum of their radii. {task}

Hard requirements for your program:
- Define exactly this function: def run_packing() -> tuple[np.ndarray, np.ndarray, float]
- It returns (centers, radii, sum_radii): centers has shape ({n}, 2),
  radii has shape ({n},), sum_radii is float(radii.sum()).
- N IS EXACTLY {n}. Every array you build and every bounds list you pass to an
  optimizer must have a length consistent with {n} circles. Do not hardcode a
  different count and do not let the variable vector and the bounds list disagree.
- Every circle must lie inside the unit square and no two may overlap. The
  result is checked by this exact validator (do not redefine it):

```python
{_VALIDATOR_SRC}
```

- numpy (as np) and math are already imported. scipy.optimize.minimize is
  available as the name `minimize` (already imported for you). Do NOT re-import
  scipy; just use `minimize(...)`.
- Top-level helper functions only. No lambdas, no nested closures, no classes.
- No file or network IO.

A correct, self-consistent optimizer pattern you may adapt (note how the decision
vector v has length 3*{n} and the bounds list has the SAME length 3*{n}; this is
the mismatch that most commonly breaks programs):

```python
def run_packing():
    n = {n}
    # decision vector v = [x0,y0,r0, x1,y1,r1, ...], length 3*n
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]   # length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({{"type": "ineq", "fun": (lambda i: lambda v: v[3*i]   - v[3*i+2])(i)}})
        cons.append({{"type": "ineq", "fun": (lambda i: lambda v: 1.0 - v[3*i]   - v[3*i+2])(i)}})
        cons.append({{"type": "ineq", "fun": (lambda i: lambda v: v[3*i+1] - v[3*i+2])(i)}})
        cons.append({{"type": "ineq", "fun": (lambda i: lambda v: 1.0 - v[3*i+1] - v[3*i+2])(i)}})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({{"type": "ineq",
                          "fun": (lambda i, j: lambda v:
                                  (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2
                                  - (v[3*i+2]+v[3*j+2])**2)(i, j)}})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={{"maxiter": 500, "ftol": 1e-9}})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())
```

You do not have to use this exact method, but whatever you do, keep the variable
vector and the bounds list the same length, and return exactly {n} circles.

{state}
Output format (CRITICAL): respond with ONE Python code block and NOTHING else.
Start your reply with ```python on its own line and end it with ```. Do not
write any explanation before or after the code block.

```python
import numpy as np

def run_packing():
    ...
    return centers, radii, float(radii.sum())
```"""
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
            "from scipy.optimize import minimize\n\n"
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
        if valid:
            s = float(np.sum(radii))
            # Degenerate guard: a zero-radius packing is "valid" but worthless and
            # collapses the score bands. Reject it so it never enters the archive.
            if s < self.min_sum_radii:
                res.valid = False
                res.msg = f"degenerate_sum_radii:{s:.3e}"
                return res
            res.valid, res.msg = True, msg
            res.reward = s
            res.raw = s
        else:
            res.valid, res.msg = False, msg
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
        # One guaranteed-valid canned packing as a nonzero anchor; the remaining
        # slots are generated independently from scratch by the LLM (step 1).
        n = max(1, self.num_seed_states)
        seeds = [Seed(code=_GRID_SEED)]
        seeds += [Seed(code="") for _ in range(n - 1)]
        return seeds