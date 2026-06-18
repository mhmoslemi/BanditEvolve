"""
State archive with UCT parent selection and lineage bookkeeping.

Only VALID states (initial seeds and explore roots are VALID too) are selectable
as parents. STERILE children (ran-but-redundant: failed novelty) and INVALID
children (broke a hard rule) are recorded separately for diagnostics, can never
be selected, and are never used as the 'worst-but-valid' exemplar in a mutation
prompt. Because only VALID states ever become parents, every ancestor in a
lineage is resolvable from the selectable set.

UCT selection score for a candidate s:

    score(s) = Qn(s) + c * sqrt( ln(1 + T) / (1 + n(s)) )

  Q(s)   = max child reward seen from s (or s.value if never expanded)
  Qn(s)  = Q(s) min-max normalized across the current candidates, so the
           exploration term is on a comparable scale across problems whose
           rewards differ in magnitude (runtime us vs sum-of-radii vs 1/MSE)
  n(s)   = expansions of s or any descendant
  T      = total expansions

Batch selection blocks the full ancestor+descendant lineage of each pick, so a
single iteration's parents do not all collapse onto one promising thread.
"""

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

VALID = "valid"
STERILE = "sterile"
INVALID = "invalid"


@dataclass
class State:
    id: str
    code: str
    value: float                                   # aggregate reward, higher better
    raw_score: Optional[float] = None
    per_seed: Dict[int, float] = field(default_factory=dict)   # seed -> reward
    parents: List[dict] = field(default_factory=list)          # [{id,timestep}], [0]=immediate parent
    timestep: int = 0
    is_seed: bool = False
    status: str = VALID
    band: Optional[str] = None
    construction: Optional[list] = None

    @staticmethod
    def make(code, value, timestep, parents=None, is_seed=False, status=VALID,
             raw_score=None, per_seed=None, construction=None):
        return State(
            id=str(uuid.uuid4()), code=code, value=value, raw_score=raw_score,
            per_seed=dict(per_seed or {}), parents=list(parents or []),
            timestep=timestep, is_seed=is_seed, status=status,
            construction=construction,
        )

    @property
    def selectable(self) -> bool:
        return self.status == VALID


def child_lineage(parent: State) -> List[dict]:
    """Lineage list a child of `parent` should carry: immediate parent first."""
    return [{"id": parent.id, "timestep": parent.timestep}] + list(parent.parents or [])


class Archive:
    def __init__(self, uct_c=1.0, max_size=2000, topk_children=3,
                 max_nonparents=4000):
        self.uct_c = float(uct_c)
        self.max_size = int(max_size)
        self.topk_children = int(topk_children)

        self._by_id: Dict[str, State] = {}     # selectable states only
        self._order: List[str] = []
        self._n: Dict[str, int] = {}           # visit counts (incl. descendants)
        self._m: Dict[str, float] = {}         # best child reward per state
        self._T = 0

        # sterile + invalid live here (never selectable); kept for diagnostics
        self.nonparents = deque(maxlen=int(max_nonparents))
        self.counts = {VALID: 0, STERILE: 0, INVALID: 0}
        self.last_picks_info = []

    # ---------------------------------------------------------------- inserts
    def _insert_selectable(self, s: State):
        self._by_id[s.id] = s
        self._order.append(s.id)
        self.counts[VALID] += 1

    def add_seed(self, s: State):
        s.is_seed, s.status = True, VALID
        self._insert_selectable(s)

    def add_root(self, s: State):
        s.status = VALID
        self._insert_selectable(s)
        self._prune()

    def add_child(self, s: State):
        s.status = VALID
        self._insert_selectable(s)
        self._prune()

    def add_nonparent(self, s: State, status: str):
        s.status = status
        self.nonparents.append(s)
        self.counts[status] = self.counts.get(status, 0) + 1

    def record_child_reward(self, parent: State, child_value: float):
        self._m[parent.id] = max(self._m.get(parent.id, -math.inf), float(child_value))

    def record_expansion(self, parent: State, count: int = 1):
        for aid in [parent.id] + [p["id"] for p in (parent.parents or [])]:
            self._n[aid] = self._n.get(aid, 0) + count
        self._T += count

    # ---------------------------------------------------------------- lineage
    def code_of(self, sid: Optional[str]) -> str:
        s = self._by_id.get(sid) if sid else None
        return s.code if s else ""

    def grandparent_code_of(self, parent: State) -> str:
        # parent.parents[0] is the parent's immediate parent == the child's grandparent
        if parent.parents:
            return self.code_of(parent.parents[0]["id"])
        return ""

    def _children_map(self) -> Dict[str, set]:
        cm: Dict[str, set] = {}
        for sid in self._order:
            for p in self._by_id[sid].parents:
                cm.setdefault(p["id"], set()).add(sid)
        return cm

    def _lineage(self, sid: str, cm: Dict[str, set]) -> set:
        lin = {sid}
        s = self._by_id.get(sid)
        if s:
            for p in s.parents:
                lin.add(p["id"])
        q = [sid]
        while q:
            cur = q.pop()
            for c in cm.get(cur, ()):
                if c not in lin:
                    lin.add(c)
                    q.append(c)
        return lin

    # ------------------------------------------------------------- selection
    def _selectable(self) -> List[State]:
        return [self._by_id[i] for i in self._order]

    def select_parents(self, num: int) -> List[State]:
        cands = self._selectable()
        self.last_picks_info = []
        if not cands:
            return []

        Q = []
        for s in cands:
            n = self._n.get(s.id, 0)
            Q.append(self._m[s.id] if (n > 0 and s.id in self._m) else s.value)
        qmin, qmax = min(Q), max(Q)
        span = (qmax - qmin) or 1.0
        logT = math.log(1.0 + self._T)

        scored = []
        for s, q in zip(cands, Q):
            n = self._n.get(s.id, 0)
            qn = (q - qmin) / span
            bonus = self.uct_c * math.sqrt(logT / (1.0 + n))
            info = {"value": s.value, "n": n, "Q": q, "qn": qn,
                    "bonus": bonus, "score": qn + bonus, "is_seed": s.is_seed}
            scored.append((qn + bonus, s.value, s, info))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        cm = self._children_map()
        blocked, picks, info_out = set(), [], []
        for _, _, s, info in scored:
            if s.id in blocked:
                continue
            picks.append(s)
            info_out.append(info)
            blocked |= self._lineage(s.id, cm)
            if len(picks) >= num:
                break
        if len(picks) < num:                       # top up without blocking
            have = {s.id for s in picks}
            for _, _, s, info in scored:
                if len(picks) >= num:
                    break
                if s.id not in have:
                    picks.append(s)
                    info_out.append(info)
                    have.add(s.id)
        self.last_picks_info = info_out
        return picks

    # ----------------------------------------------------------- exemplars
    def _non_seed_valid(self) -> List[State]:
        return [s for s in self._selectable() if not s.is_seed]

    def best_state(self) -> Optional[State]:
        pool = self._non_seed_valid() or self._selectable()
        return max(pool, key=lambda s: s.value) if pool else None

    def worst_valid_state(self) -> Optional[State]:
        pool = self._non_seed_valid() or self._selectable()
        return min(pool, key=lambda s: s.value) if pool else None

    def top_k_codes(self, k: int) -> List[str]:
        pool = sorted(self._selectable(), key=lambda s: s.value, reverse=True)
        return [s.code for s in pool[:k] if s.code]

    def valid_values(self) -> List[float]:
        return [s.value for s in self._selectable()]

    def size(self) -> int:
        return len(self._order)

    @property
    def T(self) -> int:
        return self._T

    # -------------------------------------------------------------- pruning
    def _prune(self):
        # Keep top-K children per immediate parent (seeds / roots always kept).
        if self.topk_children > 0:
            by_parent: Dict[str, List[State]] = {}
            keep_ids = set()
            for sid in self._order:
                s = self._by_id[sid]
                if s.is_seed or not s.parents:
                    keep_ids.add(sid)
                    continue
                by_parent.setdefault(s.parents[0]["id"], []).append(s)
            for kids in by_parent.values():
                kids.sort(key=lambda s: s.value, reverse=True)
                for s in kids[: self.topk_children]:
                    keep_ids.add(s.id)
            if len(keep_ids) < len(self._order):
                self._order = [i for i in self._order if i in keep_ids]
                self._by_id = {i: self._by_id[i] for i in self._order}

        # Global cap, never dropping seeds / roots.
        if len(self._order) > self.max_size:
            roots = [i for i in self._order
                     if self._by_id[i].is_seed or not self._by_id[i].parents]
            rest = [i for i in self._order if i not in set(roots)]
            rest.sort(key=lambda i: self._by_id[i].value, reverse=True)
            keep = roots + rest[: max(0, self.max_size - len(roots))]
            keep_set = set(keep)
            self._order = [i for i in self._order if i in keep_set]
            self._by_id = {i: self._by_id[i] for i in self._order}