"""
The child validation gate (steps 6 and 7).

Outcomes:
  ok=True                  -> proceed to evaluation
  invalid=True             -> broke a hard rule (unparseable, missing entrypoint,
                              problem constraint violation). Archived as INVALID,
                              never a parent.
  sterile=True             -> ran-but-redundant: a cosmetic/no-op edit of the
                              parent, or fails the novelty check against
                              {parent, grandparent, best, top-10}. Archived as
                              STERILE, never a parent.

Both invalid and sterile children are non-parents; the distinction is purely
diagnostic. Novelty is checked against VALID states only, so sterile/invalid
near-duplicates never pollute the comparison set, and the worst-but-valid
exemplar in the mutation prompt is always a real, valid program.
"""

from dataclasses import dataclass
from typing import Optional

import codetools


@dataclass
class GateResult:
    ok: bool
    reason: str = ""
    invalid: bool = False
    sterile: bool = False
    sim_parent: float = 0.0
    novelty_sim: float = 0.0          # max similarity to the reference set


def _invalid(reason, sim_parent=0.0):
    return GateResult(ok=False, invalid=True, reason=reason, sim_parent=sim_parent)


def _sterile(reason, sim_parent=0.0, novelty_sim=0.0):
    return GateResult(ok=False, sterile=True, reason=reason,
                      sim_parent=sim_parent, novelty_sim=novelty_sim)


def quick_gate(code: str, problem) -> GateResult:
    """Lighter gate for explore seeds and bootstrap (no parent, no novelty)."""
    if not code or codetools.parse(code) is None:
        return _invalid("unparseable")
    if problem.entrypoint not in codetools.defined_functions(code):
        return _invalid(f"missing_entrypoint:{problem.entrypoint}")
    ok, msg = problem.static_check(code)
    if not ok:
        return _invalid(f"constraint:{msg}")
    return GateResult(ok=True)


def validate_child(code, parent, grandparent_code, archive, problem, cfg) -> GateResult:
    # 1) parseable
    if not code or codetools.parse(code) is None:
        return _invalid("unparseable")

    # 2) required functions / entrypoint
    if problem.entrypoint not in codetools.defined_functions(code):
        return _invalid(f"missing_entrypoint:{problem.entrypoint}")

    # 3) problem-specific output / config constraints
    ok, msg = problem.static_check(code)
    if not ok:
        return _invalid(f"constraint:{msg}")

    # 4) no-op / cosmetic change vs parent
    sim_parent = codetools.similarity(parent.code, code)
    if codetools.is_cosmetic_change(parent.code, code):
        return _sterile("cosmetic_noop", sim_parent=sim_parent, novelty_sim=1.0)
    if sim_parent >= cfg.parent_sim_threshold:
        return _sterile("too_similar_to_parent", sim_parent=sim_parent,
                        novelty_sim=sim_parent)

    # 5) novelty vs {parent, grandparent, best, top-10}  (valid states only)
    refs = [parent.code]
    if grandparent_code:
        refs.append(grandparent_code)
    best = archive.best_state()
    if best is not None:
        refs.append(best.code)
    refs.extend(archive.top_k_codes(cfg.novelty_topk))

    nov = 0.0
    for r in refs:
        if not r:
            continue
        s = codetools.similarity(r, code)
        if s > nov:
            nov = s
    if nov >= cfg.novelty_threshold:
        return _sterile("not_novel", sim_parent=sim_parent, novelty_sim=nov)

    return GateResult(ok=True, sim_parent=sim_parent, novelty_sim=nov)
