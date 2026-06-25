"""
Score bands (step 5) and within-band delta normalization.

Bands are assigned by quantile of the current VALID archive rewards, so 'weak'
vs 'near_sota' is relative to what the search has found so far rather than a
fixed absolute threshold.

Within-band normalization is the load-bearing fix for the prompt bandit's credit
assignment. A raw mutation delta of +0.01 means something completely different
coming off a near_sota parent (where headroom is tiny) than off a weak parent
(where it is large). If we fed raw deltas to the bandit, the near_sota pool would
look uniformly terrible and the search would never tune it. So each band keeps a
running mean/std of the deltas it observes and reports a standardized z-score as
the bandit reward. Each band's arms are then compared on that band's own scale.
"""

import numpy as np # type: ignore

WEAK, GOOD, ELITE, NEAR_SOTA = "weak", "good", "elite", "near_sota"
BANDS = [WEAK, GOOD, ELITE, NEAR_SOTA]


class BandStats:
    """Welford running mean/std of mutation deltas, one accumulator per band."""

    def __init__(self):
        self._n = {b: 0 for b in BANDS}
        self._mean = {b: 0.0 for b in BANDS}
        self._m2 = {b: 0.0 for b in BANDS}

    def update(self, band: str, delta: float):
        self._n[band] += 1
        d = delta - self._mean[band]
        self._mean[band] += d / self._n[band]
        self._m2[band] += d * (delta - self._mean[band])

    def normalize(self, band: str, delta: float) -> float:
        n = self._n[band]
        if n < 2:
            return float(delta)            # not enough history: pass through raw
        std = max((self._m2[band] / (n - 1)) ** 0.5, 1e-6)
        return float((delta - self._mean[band]) / std)

    def summary(self):
        return {b: (self._n[b], round(self._mean[b], 5)) for b in BANDS}


class BandAssigner:
    """Assign a value to a score band, in one of two modes.

    quantile (default, absolute=False): thresholds are quantiles of the current
        valid archive rewards, so a band is RELATIVE to what the search has found
        so far. With only a few tied top seeds, the 99.5th percentile is just the
        running max, so the best-so-far seeds land in near_sota regardless of how
        far they are from the real target.

    absolute (absolute=True, target set): q_good/q_elite/q_near are read as
        FRACTIONS OF THE KNOWN TARGET ("known best result"). near_sota means
        value >= q_near * target, etc., so a band means the same absolute score
        for the whole run. Falls back to quantile mode if no target is given.
    """

    def __init__(self, q_good=0.30, q_elite=0.70, q_near=0.90,
                 absolute=False, target=None):
        self.q = (q_good, q_elite, q_near)
        self.target = float(target) if target is not None else None
        self.absolute = bool(absolute) and self.target is not None

    def assign(self, value: float, valid_values) -> str:
        if self.absolute:
            lo, mid, hi = (c * self.target for c in self.q)
        else:
            if valid_values is None or len(valid_values) < 4:
                return GOOD                # cold start
            lo, mid, hi = np.quantile(np.asarray(valid_values, dtype=float), self.q)
        if value >= hi:
            return NEAR_SOTA
        if value >= mid:
            return ELITE
        if value >= lo:
            return GOOD
        return WEAK
