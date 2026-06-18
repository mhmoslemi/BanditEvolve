"""
Problem registry.

Only circle_packing is implemented here as the worked, dependency-light example.
The other TTT problems (erdos, ac1/ac2, denoising, gpu_mode) port over by
copying their build_prompt / score / seed_states from the TTT codebase into a
subclass of the new Problem ABC and adding two things:

  - build_seed_prompt():  the from-scratch variant (usually build_prompt with an
                          empty ParentContext, which most of them already handle)
  - static_check(code):   the hard constraints they already enforce inline, e.g.
                          gpu_mode requires '@triton.jit' and forbids 'identity';
                          surface those here so the gate rejects them before any
                          run instead of wasting an evaluation.

Their preprocess already threads a seed where the entrypoint takes one (erdos
run(seed=...), denoising eval_seed), so they satisfy the per-seed contract.
"""

from __future__ import annotations


def available_problems():
    return ["circle_packing"]


def get_problem(name: str, cfg: dict):
    key = (name or "").strip().lower()
    if key in ("circle_packing", "circle", "circles"):
        from problems.circle_packing import CirclePacking
        return CirclePacking(cfg)
    raise ValueError(
        f"Unknown / not-yet-ported problem '{name}'. Implemented: "
        f"{', '.join(available_problems())}. See the module docstring for how "
        f"to port the other TTT problems."
    )
