"""
Entry point.

    python main.py --problem circle_packing                     # frozen search
    python main.py --problem circle_packing --llm-backend dummy # offline wiring test
    python main.py --config configs/circle_packing_rl.yaml      # GRPO fine-tuning
    python main.py --problem circle_packing --rl                # GRPO via CLI flag
    python main.py --config configs/circle_packing_rl.yaml --llm-backend dummy
                                                                # offline RL wire-test
"""

import os
import warnings

# transformers emits FutureWarnings (attention-mask API) on every generate; mute.
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import load_config
from llm import make_llm
from engine import Engine
from problems.registry import get_problem # type: ignore


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
    if getattr(cfg, "rl_enabled", False):
        print("  --- RL (GRPO) ---")
        print(f"  adapters       : {'per-band (weak/good/elite/near_sota)' if cfg.rl_adapter_per_band else 'one shared'}")
        print(f"  policy         : LoRA r={cfg.rl_lora_r} alpha={cfg.rl_lora_alpha} "
              f"dropout={cfg.rl_lora_dropout}")
        print(f"  group size G   : {cfg.rl_group_size}  "
              f"(per parent: 1 arm -> G completions = one GRPO group)")
        print(f"  gen / iter     : up to {cfg.num_parents} x {cfg.rl_group_size} "
              f"= {cfg.num_parents * cfg.rl_group_size} mutations")
        print(f"  train_every    : {cfg.rl_train_every}   ppo_epochs: {cfg.rl_ppo_epochs}")
        print(f"  lr / kl / clip : {cfg.rl_lr} / {cfg.rl_kl_coef} / {cfg.rl_clip_eps}")
        print(f"  max comp tok   : {cfg.rl_max_completion_tokens}   "
              f"thinking: {cfg.enable_thinking}")
        if cfg.enable_thinking:
            print("  WARNING        : enable_thinking=True under RL is memory-heavy "
                  "and credit-poor; prefer false.")
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
