"""
Per-band prompt pools as Normal-Inverse-Gamma Thompson-sampling bandits (step 5).

Each band owns a list of arms. An arm is a mutation-prompt template paired with
an NIG posterior over its reward, where the reward is the within-band
standardized mutation delta (see bands.py). Thompson sampling draws one believed
mean per arm from its posterior and expands the argmax arm.

Rewards can be negative (a mutation can make things worse), so the conjugate
family is Normal, not Beta. The NIG prior models unknown mean mu and variance
sigma^2:

    sigma^2 ~ InvGamma(alpha, beta),   mu | sigma^2 ~ Normal(mu0, sigma^2 / lam)

Posterior after observing n samples summarized by (xbar, S = sum (x - xbar)^2):

    lam_n   = lam0 + n
    mu_n    = (lam0*mu0 + n*xbar) / lam_n
    alpha_n = alpha0 + n/2
    beta_n  = beta0 + S/2 + lam0*n*(xbar - mu0)^2 / (2*lam_n)

A fresh arm (n = 0) samples from the wide prior, so newly added reflection arms
get explored before the bandit decides whether they help.
"""

import numpy as np


class _Arm:
    def __init__(self, template: str, source: str = "seed"):
        self.template = template
        self.source = source          # "seed" or "reflection"
        self.n = 0
        self.mean = 0.0               # running sample mean
        self.m2 = 0.0                 # running sum of squared deviations (= S)

    def update(self, reward: float):
        self.n += 1
        d = reward - self.mean
        self.mean += d / self.n
        self.m2 += d * (reward - self.mean)


class PromptBandit:
    def __init__(self, seed_templates: dict, mu0=0.0, lam0=1.0,
                 alpha0=2.0, beta0=1.0, rng=None):
        # seed_templates: {band -> [template, ...]}
        self.mu0, self.lam0 = float(mu0), float(lam0)
        self.alpha0, self.beta0 = float(alpha0), float(beta0)
        self.rng = rng or np.random.default_rng(0)
        self.pools = {b: [_Arm(t) for t in tmpls]
                      for b, tmpls in seed_templates.items()}

    def _sample_mean(self, arm: _Arm) -> float:
        n, xbar = arm.n, arm.mean
        lam_n = self.lam0 + n
        mu_n = (self.lam0 * self.mu0 + n * xbar) / lam_n
        alpha_n = self.alpha0 + n / 2.0
        beta_n = (self.beta0 + 0.5 * arm.m2
                  + (self.lam0 * n * (xbar - self.mu0) ** 2) / (2.0 * lam_n))
        # sigma^2 ~ InvGamma(alpha_n, beta_n): if g ~ Gamma(alpha_n, rate=beta_n)
        # then 1/g ~ InvGamma(alpha_n, beta_n). numpy uses scale = 1/rate.
        g = self.rng.gamma(shape=alpha_n, scale=1.0 / beta_n)
        var = 1.0 / max(g, 1e-12)
        return float(self.rng.normal(mu_n, np.sqrt(var / lam_n)))

    def sample(self, band: str):
        arms = self.pools[band]
        draws = [self._sample_mean(a) for a in arms]
        idx = int(np.argmax(draws))
        return idx, arms[idx]

    def update(self, band: str, idx: int, reward: float):
        self.pools[band][idx].update(float(reward))

    def add_arm(self, band: str, template: str, source: str = "reflection") -> int:
        self.pools[band].append(_Arm(template, source=source))
        return len(self.pools[band]) - 1

    def summary(self):
        out = {}
        for b, arms in self.pools.items():
            out[b] = [
                {"src": a.source, "n": a.n, "mean": round(a.mean, 4)}
                for a in arms
            ]
        return out
