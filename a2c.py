"""
A2C (advantage actor-critic) fine-tuning of the mutator -- the alternative to
grpo.py, selected with rl_algo: a2c.

WHAT DIFFERS FROM GRPO.
GRPO is critic-free: the per-prompt GROUP is the baseline (its own statistics
give the advantage). A2C adds a LEARNED critic V_phi(s): a scalar value head on
the frozen backbone, where the state s is the mutation prompt (parent code +
exemplars + band directive + sampled tactic). The head reads the hidden state at
the last prompt token and is trained by regression. The model mechanics live on
TrainableLLM (state_value / value_zero_grad / value_step); this file is the
algorithm, mirroring how GRPOTrainer relates to TrainableLLM.

MAX-SEEKING IS PRESERVED. The discovery objective rewards finding the single best
child, so the critic is OPTIMISTIC: it regresses toward the GROUP MAX m (the best
reward the state produced), not the mean. The actor advantage is then anchored to
that max:

    let  m = max_g r,   V = V_phi(s),   sd = std_g r,   denom = sd + eps
    every member i:     A_i  = (r_i - V) / denom                  (reward - value)
    the argmax  i*:     A_i* = max( (m - V), (m - second) ) / denom   (> 0)

(r - V) is the textbook actor-critic advantage: members that beat the critic's
prediction are reinforced, those below it are suppressed. The winner override
keeps a strictly positive, max-anchored signal even if the critic overshoots
(V >= m), so the single best action never gets a zero/negative gradient, and
since m >= r_i it stays the most-reinforced action. As the critic learns to
predict the max (V -> m) the winner term collapses to GRPO's runner-up margin
(m - second)/denom, so A2C degrades gracefully into the max-anchored GRPO it
generalizes (identical for a unique max).

CRITIC LOSS.  L_v = vf_coef * (V_phi(s) - m)^2, one optimizer step per update
over the UNIQUE states in the buffer (each group is one state). Only the linear
value head trains; the 8B backbone is frozen, so this adds no backprop through
the base model.

ACTOR LOSS is identical to grpo.py (PPO-clipped surrogate + KL-to-reference);
A2CTrainer reuses GRPOTrainer.update for it and just feeds the critic-relative
advantage above, then runs the extra critic step.
"""

from typing import List

import numpy as np # type: ignore

from grpo import GRPOTrainer, RLSample, BandedGroup


def max_anchored_critic_advantages(rewards: np.ndarray, value: float,
                                   eps: float) -> np.ndarray:
    """Max-anchored, critic-relative advantages (see module docstring).

    A_i = (r_i - V)/denom for every member; the argmax member(s) are overridden
    to max((m - V), (m - second))/denom so the best action keeps a strictly
    positive signal even if the critic overshoots (V >= m). Because m >= r_i for
    all i, that winner value is also >= every member's (r_i - V)/denom, so the
    group max is always the most-reinforced action (the max-anchored property).
    Tied winners each get the full winner value (not split): every best sequence
    is reinforced, which suits the max-seeking objective.
    """
    r = np.asarray(rewards, dtype=float)
    denom = float(r.std()) + eps
    m = float(r.max())
    below = r[r < m]
    second = float(below.max()) if below.size else m

    adv = (r - float(value)) / denom              # reward - value (A2C advantage)
    margin = (m - second) / denom                 # >= 0 (runner-up gap)
    adv[r >= m] = max((m - float(value)) / denom, margin)
    return adv


class A2CTrainer(GRPOTrainer):
    """GRPO's PPO-clip + KL loss machinery, plus a learned value-head critic and
    a max-anchored, critic-relative advantage. Requires the LLM to carry a value
    head (TrainableLLM built with use_value_head=True, i.e. rl_algo: a2c)."""

    def __init__(self, llm, cfg):
        super().__init__(llm, cfg)
        self.vf_coef = float(getattr(cfg, "rl_vf_coef", 0.5))
        if getattr(llm, "value_head", None) is None:
            raise RuntimeError(
                "A2CTrainer requires a value head. make_llm builds one when "
                "rl_algo: a2c; got an LLM without value_head.")

    # ---- experience assembly: same as GRPO but with the critic-relative adv ----
    def build_samples(self, banded_groups: List[BandedGroup]) -> List[RLSample]:
        samples: List[RLSample] = []
        for band, group in banded_groups:
            if len(group) < 2:
                continue
            if self.llm._adapter_for(band) is None:
                continue
            rewards = np.asarray([r for (_, r) in group], dtype=float)
            if float(rewards.std()) < self.min_group_std:
                continue                                    # no relative signal

            # every member of a group shares ONE prompt (one parent+arm) = one state
            state_prompt = list(group[0][0].prompt_ids)
            value = float(self.llm.state_value(state_prompt, with_grad=False))
            group_max = float(rewards.max())
            advs = max_anchored_critic_advantages(rewards, value, self.adv_eps)

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
                    value_target=group_max,
                ))
        return samples

    # ---- update: GRPO actor step, then one critic (value) regression step ----
    def update(self, samples: List[RLSample]) -> dict:
        stats = super().update(samples)
        stats.update(self._update_critic(samples))
        return stats

    def _update_critic(self, samples: List[RLSample]) -> dict:
        if not samples:
            return {"vf_loss": 0.0, "vf_grad_norm": 0.0, "vf_states": 0}
        # one regression target per unique state (group): V(s) -> group max
        targets = {}
        for s in samples:
            targets[tuple(s.prompt_ids)] = s.value_target

        self.llm.value_zero_grad()
        tot = 0.0
        for key, tgt in targets.items():
            v = self.llm.state_value(list(key), with_grad=True)
            loss = self.vf_coef * (v - float(tgt)) ** 2
            loss.backward()
            tot += float((v.detach() - float(tgt)) ** 2)
        gnorm = self.llm.value_step(self.grad_clip)
        n = len(targets)
        return {"vf_loss": tot / max(n, 1), "vf_grad_norm": float(gnorm),
                "vf_states": n}
