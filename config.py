"""
Config: dataclass defaults < configs/<problem>.yaml < a few CLI overrides.
"""

import argparse
import os
from dataclasses import dataclass, fields
from typing import Tuple

import yaml


@dataclass
class Config:
    # problem
    problem: str = "circle_packing"
    problem_type: str = ""
    target: float = 2.635983
    num_circles: int = 26

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

    # llm (frozen policy, served externally)
    llm_backend: str = "openai"         # "openai" | "dummy"
    llm_model: str = "openai/gpt-oss-120b"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key_env: str = "LLM_API_KEY"
    llm_concurrency: int = 8
    llm_timeout_s: float = 600.0
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 26000

    # misc
    seed: int = 42


def _parser():
    p = argparse.ArgumentParser(description="Band-bandit evolutionary search")
    p.add_argument("--problem", default="circle_packing")
    p.add_argument("--config", default=None)
    p.add_argument("--llm-backend", default=None, choices=["openai", "dummy"])
    p.add_argument("--num-iters", type=int, default=None)
    p.add_argument("--num-parents", type=int, default=None)
    p.add_argument("--rollouts-per-parent", type=int, default=None)
    p.add_argument("--explore-eps", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


_CLI = {"llm_backend": "llm_backend", "num_iters": "num_iters",
        "num_parents": "num_parents", "rollouts_per_parent": "rollouts_per_parent",
        "explore_eps": "explore_eps", "seed": "seed"}


def load_config():
    args = _parser().parse_args()
    merged = {f.name: getattr(Config(), f.name) for f in fields(Config)}

    path = args.config or os.path.join("configs", f"{args.problem}.yaml")
    ydict = {}
    if os.path.exists(path):
        with open(path) as f:
            ydict = yaml.safe_load(f) or {}
        merged.update(ydict)
        print(f"[config] loaded {path}")
    else:
        print(f"[config] no YAML at {path}; using defaults + CLI")
    merged["problem"] = ydict.get("problem", args.problem)

    for name, val in vars(args).items():
        if name in ("problem", "config") or val is None:
            continue
        merged[_CLI.get(name, name)] = val

    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in merged.items() if k in known})
    return cfg, merged
