"""
GRPO fine-tuning of the mutator (the RL upgrade), with a SEPARATE LoRA adapter
per band.

WHY GRPO AND NOT PPO HERE.
A mutation is a one-shot contextual bandit. The policy sees a state (the mutation
prompt: parent code + exemplars + band directive + the sampled tactic), emits a
single action (the child program), and receives one terminal reward (the
sandboxed paired delta). There is no temporal structure inside an episode, so
PPO's value function has nothing to bootstrap over and GAE degenerates: the critic
would be a learned constant-per-state baseline trained on a single sample. GRPO
gives that baseline for free. Sample a GROUP of children from the SAME prompt, use
the group statistics as the baseline. The advantage is purely group-relative,
with no critic to train and no second network to OOM an 8B model on. "PPO without
a critic, with a group baseline" IS GRPO.

ADVANTAGE: MAX-SEEKING (group-max anchored).
The discovery objective rewards finding the single best child, not lifting the
group average. So the advantage is anchored to the GROUP MAX rather than the
group mean:

    let  m  = max_g r,   mu = mean_g r,   sd = std_g r
    argmax member i*:   A_i* = + (m - second_g) / (sd + eps)     (> 0, reinforced
                                  in proportion to how far the best beat the rest)
    every other member: A_i  =   (r_i - m) / (sd + eps)          (<= 0, suppressed
                                  toward the group's best action)

This pushes probability mass onto the single best action in each group and pushes
it off everything below it, in proportion to the margin. It deliberately drops
GRPO's mean-baseline (an unbiased, low-variance estimator) for a max-anchored one:
the trade is a biased, higher-variance signal that chases the extreme instead of
the average. The KL-to-reference leash and the importance ratio still apply.

NOTE the failure mode this AVOIDS: the naive max-baseline A = (r - max)/sd gives
the best member advantage EXACTLY 0, i.e. zero gradient on the one action you most
want to reinforce, so the policy only ever learns to push losers down and tends to
collapse toward the reference. The argmax-margin term above fixes that by giving
the winner a positive signal.

WHY ONE ADAPTER PER BAND.
The four bands demand opposite behavior: a weak parent should be rewritten from
scratch, a near_sota parent should be nudged additively. A single adapter receives
the gradient from a weak-band "paradigm shift" win and a near_sota "tighten one
tolerance" win at once, and those pull in conflicting directions. Training one
LoRA adapter per band lets each specialize. It also mirrors the existing design:
the prompt bandit is already per-band (which instruction to try); now the policy
is per-band too (how to execute it). The KL reference is the SHARED frozen base
(adapter disabled) for every band, so all adapters are leashed to the same
pretrained model.

This file is the ALGORITHM. The model mechanics (multi-adapter generation,
per-token logprobs under a chosen adapter, the per-adapter optimizer step, the
shared reference via adapter-disable) live on TrainableLLM in llm.py. GRPOTrainer
composes those primitives into the loss and the per-band update.

STATE / ACTION / REWARD.
    state  s_i = prompt token ids       (TrainableLLM tokenizes the rendered chat)
    action a_i = completion token ids    (the generated program, incl. EOS)
    reward r_i = rollout_reward(...)      below

Reward shaping covers the FULL outcome space, not only valid deltas, because the
failure modes the search keeps logging (no_code, invalid, sterile) ARE actions the
policy chose and should be pushed down relative to valid improvements:

    valid    ->  raw dmu              (paired improvement over parent; may be < 0)
    sterile  ->  rl_reward_sterile    (ran but redundant / not novel)
    invalid  ->  rl_reward_invalid    (broke a hard rule or crashed in the sandbox)
    no_code  ->  rl_reward_nocode     (worst: a whole generation with no program)

The defaults give the coherent total order
    no_code < invalid < sterile < (valid, ranked by dmu).

LOSS (token-level, PPO-clipped surrogate + KL-to-reference):
    ratio_{i,t} = exp(logp_theta - logp_behavior)
    L_pg        = - mean_t min(ratio * A_i, clip(ratio, 1-e, 1+e) * A_i)
    L_kl        =   mean_t [ exp(ref - theta) - (ref - theta) - 1 ]      (k3, >= 0)
    L           =   mean_i ( L_pg + beta * L_kl )

The BEHAVIOR logprobs are captured at generation time under the band's adapter, so
reusing a buffer across iterations stays valid off-policy (the ratio corrects for
drift). The REFERENCE logprobs come from the base model with all adapters disabled
and are fixed. With rl_ppo_epochs=1 on fresh data the ratio is ~1 and this reduces
to REINFORCE with the max-anchored advantage and a KL leash; rl_ppo_epochs>1 turns
on genuine PPO-style clipped reuse of the batch.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np # type: ignore


def rollout_reward(rec, cfg) -> float:
    """Map a RolloutRecord outcome to its scalar GRPO reward (see module docstring)."""
    head = (rec.outcome or "").split(":", 1)[0]
    if head == "valid":
        return float(rec.dmu) if rec.dmu is not None else 0.0
    if head == "sterile":
        return float(cfg.rl_reward_sterile)
    if head == "invalid":
        return float(cfg.rl_reward_invalid)
    return float(cfg.rl_reward_nocode)          # "no_code" / anything with no program


def max_anchored_advantages(rewards: np.ndarray, eps: float) -> np.ndarray:
    """Group-max anchored advantages (max-seeking).

    Non-argmax members: (r - max) / (std + eps)   -> <= 0, suppressed toward best.
    The single argmax member: + (max - second) / (std + eps)  -> > 0, reinforced in
    proportion to its margin over the runner-up. With a unique max this gives the
    winner a strictly positive signal (avoiding the zero-gradient-on-winner bug of
    the naive max baseline). Ties at the top split the positive signal evenly so
    the gradient is not double-counted.
    """
    r = np.asarray(rewards, dtype=float)
    n = r.shape[0]
    std = float(r.std())
    denom = std + eps
    m = float(r.max())
    # runner-up: the largest value strictly below m; if all tied, fall back to m
    below = r[r < m]
    second = float(below.max()) if below.size else m

    adv = (r - m) / denom                      # <= 0 everywhere; 0 at the max
    top_mask = r >= m                           # all members tied at the max
    n_top = int(top_mask.sum())
    margin = (m - second) / denom               # >= 0
    # give the winner(s) a positive signal; split evenly across ties
    adv[top_mask] = margin / max(n_top, 1)
    return adv


@dataclass
class RLSample:
    band: str               # which band -> which LoRA adapter this sample trains
    prompt_ids: List[int]
    completion_ids: List[int]
    old_logp: "object"      # 1D float tensor [T] on CPU (behavior policy, detached)
    ref_logp: "object"      # 1D float tensor [T] on CPU (shared reference, detached)
    advantage: float
    value_target: float = 0.0   # A2C only: critic regression target (group max); unused by GRPO


# a banded group: (band, [(GenMeta, reward), ...])  -> all members share one prompt
BandedGroup = Tuple[str, List[Tuple[object, float]]]


class GRPOTrainer:
    """Owns the GRPO update. Pulls all model-touching primitives off `llm`
    (a TrainableLLM): token_logprobs(band, ...), zero_grad(band), step(band),
    set_train(), _adapter_for(band), .device. The trainer never names adapters
    directly; it speaks in bands and lets the LLM resolve band -> adapter (so the
    rl_adapter_per_band=False shared-adapter mode works with no change here)."""

    def __init__(self, llm, cfg):
        self.llm = llm
        self.clip_eps = float(cfg.rl_clip_eps)
        self.kl_coef = float(cfg.rl_kl_coef)
        self.ppo_epochs = int(cfg.rl_ppo_epochs)
        self.grad_clip = float(cfg.rl_grad_clip)
        self.max_comp = int(cfg.rl_max_completion_tokens)
        self.adv_eps = float(cfg.rl_adv_eps)
        self.min_group_std = float(cfg.rl_min_group_std)

    # ---- experience assembly: banded groups -> RLSamples (with behavior+ref) ----
    def build_samples(self, banded_groups: List[BandedGroup]) -> List[RLSample]:
        """banded_groups: list of (band, group); each group is a list of
        (GenMeta, reward) that all came from ONE prompt (one parent, one arm).

        Computes group-MAX-anchored advantages (max-seeking; see
        max_anchored_advantages) and caches behavior + reference per-token
        logprobs (both no-grad, on CPU). Behavior logp is taken under the band's
        adapter; reference logp under the shared base. Skips groups that cannot
        form a baseline (< 2 members) or carry no signal (zero variance). Caching
        behavior logp HERE keeps a multi-iteration buffer valid off-policy."""
        samples: List[RLSample] = []
        for band, group in banded_groups:
            if len(group) < 2:
                continue
            if self.llm._adapter_for(band) is None:
                continue            # band has no adapter (e.g. good-only mode): never trained
            rewards = np.asarray([r for (_, r) in group], dtype=float)
            std = float(rewards.std())                      # population std over the group
            if std < self.min_group_std:
                continue                                    # degenerate: no relative signal
            advs = max_anchored_advantages(rewards, self.adv_eps)
            for (meta, _), adv in zip(group, advs):
                comp = list(meta.completion_ids[: self.max_comp])
                if not comp:
                    continue
                old = self.llm.token_logprobs(meta.prompt_ids, comp, band=band,
                                              with_grad=False, use_reference=False)
                ref = self.llm.token_logprobs(meta.prompt_ids, comp, band=band,
                                              with_grad=False, use_reference=True)
                samples.append(RLSample(
                    band=band,
                    prompt_ids=list(meta.prompt_ids),
                    completion_ids=comp,
                    old_logp=old.detach().to("cpu"),
                    ref_logp=ref.detach().to("cpu"),
                    advantage=float(adv),
                ))
        return samples

    # ------------------------------- the update ------------------------------
    def update(self, samples: List[RLSample]) -> dict:
        """Bucket samples by adapter (band, or 'shared') and run an independent
        GRPO update on each adapter over its own samples."""
        import torch
        if not samples:
            return {"n": 0, "per": {}}

        buckets = {}
        for s in samples:
            buckets.setdefault(self.llm._adapter_for(s.band), []).append(s)

        per = {}
        tot_n = 0
        tot_tok = 0
        self.llm.set_train(True)
        try:
            for adapter, items in buckets.items():
                n = len(items)
                band0 = items[0].band
                st = {"n": n, "pg": 0.0, "kl": 0.0, "ratio": 1.0, "grad_norm": 0.0,
                      "adv_abs": float(np.mean([abs(s.advantage) for s in items])),
                      "tokens": int(sum(len(s.completion_ids) for s in items))}
                for _ in range(self.ppo_epochs):
                    self.llm.zero_grad(band0)
                    ep_pg = ep_kl = ep_ratio = 0.0
                    for s in items:
                        cur = self.llm.token_logprobs(
                            s.prompt_ids, s.completion_ids, band=s.band,
                            with_grad=True, use_reference=False)            # [T], grad
                        old = s.old_logp.to(self.llm.device)
                        ref = s.ref_logp.to(self.llm.device)
                        T = min(cur.shape[0], old.shape[0], ref.shape[0])
                        if T == 0:
                            continue
                        cur, old, ref = cur[:T], old[:T], ref[:T]
                        adv = float(s.advantage)

                        logratio = cur - old
                        ratio = torch.exp(logratio)
                        pg1 = ratio * adv
                        pg2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                          1.0 + self.clip_eps) * adv
                        pg = -torch.minimum(pg1, pg2)                        # [T]
                        kl = torch.exp(ref - cur) - (ref - cur) - 1.0        # [T] k3, >= 0
                        per_tok = pg + self.kl_coef * kl
                        loss = per_tok.mean() / n                            # per-seq mean, 1/N outer
                        loss.backward()

                        ep_pg += float(pg.mean().detach()) / n
                        ep_kl += float(kl.mean().detach()) / n
                        ep_ratio += float(ratio.mean().detach()) / n
                    gnorm = self.llm.step(band0, self.grad_clip)
                    st["pg"], st["kl"], st["ratio"] = ep_pg, ep_kl, ep_ratio
                    st["grad_norm"] = float(gnorm)
                per[adapter] = st
                tot_n += n
                tot_tok += st["tokens"]
        finally:
            self.llm.set_train(False)

        return {"n": tot_n, "tokens": tot_tok, "per": per,
                "adapters": len(per),
                "pg": float(np.mean([v["pg"] for v in per.values()])) if per else 0.0,
                "kl": float(np.mean([v["kl"] for v in per.values()])) if per else 0.0,
                "ratio": float(np.mean([v["ratio"] for v in per.values()])) if per else 1.0}




# """
# GRPO fine-tuning of the mutator (the RL upgrade), with a SEPARATE LoRA adapter
# per band.

# WHY GRPO AND NOT PPO HERE.
# A mutation is a one-shot contextual bandit. The policy sees a state (the mutation
# prompt: parent code + exemplars + band directive + the sampled tactic), emits a
# single action (the child program), and receives one terminal reward (the
# sandboxed paired delta). There is no temporal structure inside an episode, so
# PPO's value function has nothing to bootstrap over and GAE degenerates: the critic
# would be a learned constant-per-state baseline trained on a single sample. GRPO
# gives that baseline for free. Sample a GROUP of children from the SAME prompt, use
# the group mean as the baseline and the group std to scale. The advantage is purely
# group-relative, with no critic to train and no second network to OOM an 8B model
# on. "PPO without a critic, with a group baseline" IS GRPO.

# WHY ONE ADAPTER PER BAND.
# The four bands demand opposite behavior: a weak parent should be rewritten from
# scratch, a near_sota parent should be nudged additively. A single adapter receives
# the gradient from a weak-band "paradigm shift" win and a near_sota "tighten one
# tolerance" win at once, and those pull in conflicting directions. Training one
# LoRA adapter per band lets each specialize. It also mirrors the existing design:
# the prompt bandit is already per-band (which instruction to try); now the policy
# is per-band too (how to execute it). The KL reference is the SHARED frozen base
# (adapter disabled) for every band, so all adapters are leashed to the same
# pretrained model.

# This file is the ALGORITHM. The model mechanics (multi-adapter generation,
# per-token logprobs under a chosen adapter, the per-adapter optimizer step, the
# shared reference via adapter-disable) live on TrainableLLM in llm.py. GRPOTrainer
# composes those primitives into the loss and the per-band update.

# STATE / ACTION / REWARD.
#     state  s_i = prompt token ids       (TrainableLLM tokenizes the rendered chat)
#     action a_i = completion token ids    (the generated program, incl. EOS)
#     reward r_i = rollout_reward(...)      below

# Reward shaping covers the FULL outcome space, not only valid deltas, because the
# failure modes the search keeps logging (no_code, invalid, sterile) ARE actions the
# policy chose and should be pushed down relative to valid improvements:

#     valid    ->  raw dmu              (paired improvement over parent; may be < 0)
#     sterile  ->  rl_reward_sterile    (ran but redundant / not novel)
#     invalid  ->  rl_reward_invalid    (broke a hard rule or crashed in the sandbox)
#     no_code  ->  rl_reward_nocode     (worst: a whole generation with no program)

# GRPO normalizes within each group, so the ABSOLUTE scale of these constants is
# washed out; only their ORDER and spacing inside a group matter. The defaults give
# the coherent total order  no_code < invalid < sterile < (valid, ranked by dmu).

# ADVANTAGE.   A_i = (r_i - mean_g) / (std_g + eps),  broadcast to every action token.

# LOSS (token-level, PPO-clipped surrogate + KL-to-reference):
#     ratio_{i,t} = exp(logp_theta - logp_behavior)
#     L_pg        = - mean_t min(ratio * A_i, clip(ratio, 1-e, 1+e) * A_i)
#     L_kl        =   mean_t [ exp(ref - theta) - (ref - theta) - 1 ]      (k3, >= 0)
#     L           =   mean_i ( L_pg + beta * L_kl )

# The BEHAVIOR logprobs are captured at generation time under the band's adapter, so
# reusing a buffer across iterations stays valid off-policy (the ratio corrects for
# drift). The REFERENCE logprobs come from the base model with all adapters disabled
# and are fixed. With rl_ppo_epochs=1 on fresh data the ratio is ~1 and this reduces
# to REINFORCE with a group baseline and a KL leash; rl_ppo_epochs>1 turns on genuine
# PPO-style clipped reuse of the batch.
# """

# from dataclasses import dataclass
# from typing import List, Tuple

# import numpy as np # type: ignore


# def rollout_reward(rec, cfg) -> float:
#     """Map a RolloutRecord outcome to its scalar GRPO reward (see module docstring)."""
#     head = (rec.outcome or "").split(":", 1)[0]
#     if head == "valid":
#         return float(rec.dmu) if rec.dmu is not None else 0.0
#     if head == "sterile":
#         return float(cfg.rl_reward_sterile)
#     if head == "invalid":
#         return float(cfg.rl_reward_invalid)
#     return float(cfg.rl_reward_nocode)          # "no_code" / anything with no program


# @dataclass
# class RLSample:
#     band: str               # which band -> which LoRA adapter this sample trains
#     prompt_ids: List[int]
#     completion_ids: List[int]
#     old_logp: "object"      # 1D float tensor [T] on CPU (behavior policy, detached)
#     ref_logp: "object"      # 1D float tensor [T] on CPU (shared reference, detached)
#     advantage: float


# # a banded group: (band, [(GenMeta, reward), ...])  -> all members share one prompt
# BandedGroup = Tuple[str, List[Tuple[object, float]]]


# class GRPOTrainer:
#     """Owns the GRPO update. Pulls all model-touching primitives off `llm`
#     (a TrainableLLM): token_logprobs(band, ...), zero_grad(band), step(band),
#     set_train(), _adapter_for(band), .device. The trainer never names adapters
#     directly; it speaks in bands and lets the LLM resolve band -> adapter (so the
#     rl_adapter_per_band=False shared-adapter mode works with no change here)."""

#     def __init__(self, llm, cfg):
#         self.llm = llm
#         self.clip_eps = float(cfg.rl_clip_eps)
#         self.kl_coef = float(cfg.rl_kl_coef)
#         self.ppo_epochs = int(cfg.rl_ppo_epochs)
#         self.grad_clip = float(cfg.rl_grad_clip)
#         self.max_comp = int(cfg.rl_max_completion_tokens)
#         self.adv_eps = float(cfg.rl_adv_eps)
#         self.min_group_std = float(cfg.rl_min_group_std)

#     # ---- experience assembly: banded groups -> RLSamples (with behavior+ref) ----
#     def build_samples(self, banded_groups: List[BandedGroup]) -> List[RLSample]:
#         """banded_groups: list of (band, group); each group is a list of
#         (GenMeta, reward) that all came from ONE prompt (one parent, one arm).

#         Computes group-relative advantages and caches behavior + reference
#         per-token logprobs (both no-grad, on CPU). Behavior logp is taken under
#         the band's adapter; reference logp under the shared base. Skips groups
#         that cannot form a baseline (< 2 members) or carry no signal (zero
#         variance). Caching behavior logp HERE keeps a multi-iteration buffer valid
#         off-policy."""
#         samples: List[RLSample] = []
#         for band, group in banded_groups:
#             if len(group) < 2:
#                 continue
#             rewards = np.asarray([r for (_, r) in group], dtype=float)
#             std = float(rewards.std())                      # population std over the group
#             if std < self.min_group_std:
#                 continue                                    # degenerate: no relative signal
#             mean = float(rewards.mean())
#             advs = (rewards - mean) / (std + self.adv_eps)
#             for (meta, _), adv in zip(group, advs):
#                 comp = list(meta.completion_ids[: self.max_comp])
#                 if not comp:
#                     continue
#                 old = self.llm.token_logprobs(meta.prompt_ids, comp, band=band,
#                                               with_grad=False, use_reference=False)
#                 ref = self.llm.token_logprobs(meta.prompt_ids, comp, band=band,
#                                               with_grad=False, use_reference=True)
#                 samples.append(RLSample(
#                     band=band,
#                     prompt_ids=list(meta.prompt_ids),
#                     completion_ids=comp,
#                     old_logp=old.detach().to("cpu"),
#                     ref_logp=ref.detach().to("cpu"),
#                     advantage=float(adv),
#                 ))
#         return samples

#     # ------------------------------- the update ------------------------------
#     def update(self, samples: List[RLSample]) -> dict:
#         """Bucket samples by adapter (band, or 'shared') and run an independent
#         GRPO update on each adapter over its own samples."""
#         import torch
#         if not samples:
#             return {"n": 0, "per": {}}

#         buckets = {}
#         for s in samples:
#             buckets.setdefault(self.llm._adapter_for(s.band), []).append(s)

#         per = {}
#         tot_n = 0
#         tot_tok = 0
#         self.llm.set_train(True)
#         try:
#             for adapter, items in buckets.items():
#                 n = len(items)
#                 band0 = items[0].band
#                 st = {"n": n, "pg": 0.0, "kl": 0.0, "ratio": 1.0, "grad_norm": 0.0,
#                       "adv_abs": float(np.mean([abs(s.advantage) for s in items])),
#                       "tokens": int(sum(len(s.completion_ids) for s in items))}
#                 for _ in range(self.ppo_epochs):
#                     self.llm.zero_grad(band0)
#                     ep_pg = ep_kl = ep_ratio = 0.0
#                     for s in items:
#                         cur = self.llm.token_logprobs(
#                             s.prompt_ids, s.completion_ids, band=s.band,
#                             with_grad=True, use_reference=False)            # [T], grad
#                         old = s.old_logp.to(self.llm.device)
#                         ref = s.ref_logp.to(self.llm.device)
#                         T = min(cur.shape[0], old.shape[0], ref.shape[0])
#                         if T == 0:
#                             continue
#                         cur, old, ref = cur[:T], old[:T], ref[:T]
#                         adv = float(s.advantage)

#                         logratio = cur - old
#                         ratio = torch.exp(logratio)
#                         pg1 = ratio * adv
#                         pg2 = torch.clamp(ratio, 1.0 - self.clip_eps,
#                                           1.0 + self.clip_eps) * adv
#                         pg = -torch.minimum(pg1, pg2)                        # [T]
#                         kl = torch.exp(ref - cur) - (ref - cur) - 1.0        # [T] k3, >= 0
#                         per_tok = pg + self.kl_coef * kl
#                         loss = per_tok.mean() / n                            # per-seq mean, 1/N outer
#                         loss.backward()

#                         ep_pg += float(pg.mean().detach()) / n
#                         ep_kl += float(kl.mean().detach()) / n
#                         ep_ratio += float(ratio.mean().detach()) / n
#                     gnorm = self.llm.step(band0, self.grad_clip)
#                     st["pg"], st["kl"], st["ratio"] = ep_pg, ep_kl, ep_ratio
#                     st["grad_norm"] = float(gnorm)
#                 per[adapter] = st
#                 tot_n += n
#                 tot_tok += st["tokens"]
#         finally:
#             self.llm.set_train(False)

#         return {"n": tot_n, "tokens": tot_tok, "per": per,
#                 "adapters": len(per),
#                 "pg": float(np.mean([v["pg"] for v in per.values()])) if per else 0.0,
#                 "kl": float(np.mean([v["kl"] for v in per.values()])) if per else 0.0,
#                 "ratio": float(np.mean([v["ratio"] for v in per.values()])) if per else 1.0}
