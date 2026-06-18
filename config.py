"""
Config: dataclass defaults < configs/<problem>.yaml < a few CLI overrides.

The dataclass holds ONLY engine-level knobs that every problem shares. Problem-
specific values (num_circles, target, problem_type sub-mode, budget_s, ...) live
in the YAML and flow through the returned `merged` dict to get_problem(). They
are deliberately not dataclass fields: a circle-packing default like
target=2.635983 must never sit in the generic config, because Problem.__init__
reads `self.cfg.get("target")` and a non-None value there silences each problem's
own target-defaulting logic, so the wrong target would leak into every other
problem's run.
"""

import argparse
import os
from dataclasses import dataclass, fields

import yaml


@dataclass
class Config:
    # problem selector (the problem-specific values live in the YAML)
    problem: str = "circle_packing"
    problem_type: str = ""

    # search budget
    num_iters: int = 50
    num_parents: int = 4               # n parents sampled per iteration (UCT)
    rollouts_per_parent: int = 4       # k rollouts per parent  (n*k per iter)
    num_seeds: int = 8                 # initial independent seeds (step 1)
    seed_max_attempts: int = 6         # LLM tries per seed before giving up
    num_eval_seeds: int = 5            # evaluation seeds for paired deltas
    explore_eps: float = 0.1           # probability a rollout makes a fresh seed
    timeout: float = 100.0             # per-evaluation sandbox timeout (s)

    # UCT / archive
    uct_c: float = 1.0
    max_archive: int = 2000
    topk_children: int = 3

    # bands (quantile thresholds over valid archive rewards)
    q_good: float = 0.30
    q_elite: float = 0.70
    q_near: float = 0.90

    # gate thresholds
    parent_sim_threshold: float = 0.97  # >= this vs parent -> sterile (too similar)
    novelty_threshold: float = 0.95     # >= this vs ref set -> sterile (not novel)
    novelty_topk: int = 10
    max_code_chars: int = 4000

    # llm (frozen policy, loaded in-process via Unsloth)
    llm_backend: str = "unsloth"        # "unsloth" | "dummy"
    llm_model: str = "Qwen/Qwen3-8B"
    max_seq_length: int = 32000
    load_in_4bit: bool = False
    enable_thinking: bool = False
    gen_batch_size: int = 8             # prompts per batched generate() call
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 4000

    # misc
    seed: int = 42


def _parser():
    # argparse turns --foo-bar into args.foo_bar, which already matches the
    # dataclass field names, so CLI overrides apply by name with no remapping.
    p = argparse.ArgumentParser(description="Band-bandit evolutionary search")
    p.add_argument("--problem", default="circle_packing")
    p.add_argument("--config", default=None)
    p.add_argument("--llm-backend", default=None, choices=["unsloth", "dummy"])
    p.add_argument("--num-iters", type=int, default=None)
    p.add_argument("--num-parents", type=int, default=None)
    p.add_argument("--rollouts-per-parent", type=int, default=None)
    p.add_argument("--explore-eps", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def load_config():
    """Returns (cfg, merged).

    cfg    is the engine Config (only the fields it declares).
    merged is the full dict: engine defaults < YAML < CLI, INCLUDING problem-only
           YAML keys (num_circles, target, problem_type, ...) that the problem
           registry consumes.
    """
    args = _parser().parse_args()

    # 1) dataclass defaults
    merged = {f.name: getattr(Config(), f.name) for f in fields(Config)}

    # 2) YAML overlay (adds problem-only keys not present on Config)
    path = args.config or os.path.join("configs", f"{args.problem}.yaml")
    ydict = {}
    if os.path.exists(path):
        with open(path) as f:
            ydict = yaml.safe_load(f) or {}
        merged.update(ydict)
        print(f"[config] loaded {path}")
    elif args.config is not None:
        raise FileNotFoundError(f"--config path not found: {path}")
    else:
        print(f"[config] no YAML at {path}; using defaults + CLI")
    merged["problem"] = ydict.get("problem", args.problem)

    # 3) CLI overlay (only explicitly-provided flags)
    for name, val in vars(args).items():
        if name in ("problem", "config") or val is None:
            continue
        merged[name] = val

    # 4) build Config from the fields it knows; problem-only keys stay in merged
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in merged.items() if k in known})
    return cfg, merged