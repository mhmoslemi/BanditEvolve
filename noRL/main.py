"""
Entry point.

    python main.py --problem circle_packing
    python main.py --problem circle_packing --llm-backend dummy   # offline wiring test
"""

import os
import warnings

# transformers emits FutureWarnings (attention-mask API) on every generate; mute.
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from BanditEvolve.noRL.config import load_config
from BanditEvolve.noRL.llm import make_llm
from BanditEvolve.noRL.engine import Engine
from problems.registry import get_problem


def main():
    cfg, merged = load_config()
    problem = get_problem(cfg.problem, merged)

    # let the dummy LLM know the entrypoint so it returns a parseable stub
    setattr(cfg, "_entrypoint", getattr(problem, "entrypoint", "run"))
    llm = make_llm(cfg)

    print("=" * 64)
    print("Band-bandit evolutionary search")
    print(f"  problem        : {cfg.problem}"
          + (f" ({cfg.problem_type})" if cfg.problem_type else ""))
    print(f"  metric         : {problem.metric_name} "
          f"({'maximize' if problem.maximize else 'minimize'})")
    print(f"  llm            : {cfg.llm_backend}:{cfg.llm_model}")
    print(f"  iters          : {cfg.num_iters}")
    print(f"  parents x roll : {cfg.num_parents} x {cfg.rollouts_per_parent} "
          f"= {cfg.num_parents * cfg.rollouts_per_parent}/iter")
    print(f"  eval seeds     : {cfg.num_eval_seeds}   explore eps: {cfg.explore_eps}")
    print("=" * 64)

    engine = Engine(cfg, problem, llm)
    best = engine.run()

    print("\n" + "=" * 64)
    if best is not None:
        raw = (f"  (raw {problem.metric_name} = {best.raw_score:.6f})"
               if best.raw_score is not None else "")
        print(f"best reward: {best.value:.6f}{raw}   found at iter {best.timestep}")
        print("band arms:", engine.bandit.summary())
        print("band delta stats:", engine.band_stats.summary())
        print("\n--- best code ---\n" + best.code + "\n--- end ---")
    else:
        print("no valid solution produced")


if __name__ == "__main__":
    main()