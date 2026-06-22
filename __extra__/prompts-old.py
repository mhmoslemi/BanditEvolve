"""
All prompt text in one place.

  - BAND_FREEDOM:   the per-band mutation-freedom clause injected into every
                    mutation prompt for that band, independent of which arm is
                    sampled. This is what makes 'not all bands get the same
                    mutation freedom' (step 5) true structurally, not just by
                    which template happened to be sampled.
  - SEED_TEMPLATES: the initial arms in each band's pool (step 5). Reflection
                    adds more arms over time (step 9).
  - EVOLUTION_BLOCK: the parent / grandparent / best / worst-but-valid exemplar
                    block appended to the problem's own task prompt (step 6).
  - REFLECTION_*:   end-of-iteration meta prompt and its JSON parser (step 9).
"""

import json
import re
from typing import Optional

from bands import WEAK, GOOD, ELITE, NEAR_SOTA


BAND_FREEDOM = {
    WEAK: (
        "This program is weak relative to the rest of the search. You have full "
        "freedom: discard its approach if needed and try a fundamentally "
        "different algorithm or formulation."
    ),
    GOOD: (
        "This program is mid-pack. Keep what works but make a substantial "
        "change: swap a core subroutine, change the optimizer, or restructure "
        "the search, not just constants."
    ),
    ELITE: (
        "This program is among the strongest found. Make a targeted improvement "
        "to a specific bottleneck. Preserve the overall structure that makes it "
        "strong."
    ),
    NEAR_SOTA: (
        "This program is at or near the best known. Make a conservative, "
        "low-risk refinement only. Do not restructure it. A micro-optimization "
        "that holds up across all evaluation seeds is the goal."
    ),
}


SEED_TEMPLATES = {
    WEAK: [
        "Replace the search strategy entirely with a different family of method "
        "(for example switch between gradient-based, combinatorial, and "
        "sampling-based search).",
        "Rebuild the initialization from first principles, then attach a simple "
        "local optimizer.",
    ],
    GOOD: [
        "Identify the single weakest component and replace it with a stronger "
        "alternative, keeping the rest intact.",
        "Add a second optimization stage that polishes the output of the first.",
    ],
    ELITE: [
        "Profile the dominant cost or error source from the printed logs and "
        "attack only that, leaving everything else unchanged.",
        "Tighten the numerical tolerances and the convergence criteria of the "
        "existing optimizer.",
    ],
    NEAR_SOTA: [
        "Sweep the existing hyperparameters in a narrow band around their "
        "current values and keep the best, changing nothing structural.",
        "Add a final cleanup or projection pass that only ever improves a valid "
        "solution and never breaks validity.",
    ],
}


def evolution_block(parent_code, grandparent_code, best_code, worst_code,
                    band, arm_template, max_chars=4000):
    def trunc(c):
        c = c or ""
        return c if len(c) <= max_chars else c[:max_chars] + "\n# ... [truncated] ...\n"

    parts = [f"## Mutation directive (score band: {band})",
             BAND_FREEDOM.get(band, ""), "", "Specific tactic to apply:",
             arm_template, "", "## Current program to improve (the parent)",
             f"```python\n{trunc(parent_code)}\n```"]
    if grandparent_code:
        parts += ["", "## The program this parent came from (the grandparent), "
                  "for context on what changed",
                  f"```python\n{trunc(grandparent_code)}\n```"]
    if best_code:
        parts += ["", "## Best program found so far in the whole search "
                  "(reference, do not copy verbatim)",
                  f"```python\n{trunc(best_code)}\n```"]
    if worst_code:
        parts += ["", "## A valid but weak program (a negative exemplar: avoid "
                  "what makes this one weak)",
                  f"```python\n{trunc(worst_code)}\n```"]
    parts += [
        "",
        "Return ONE improved, complete program and NOTHING else. It must define "
        "the same entrypoint function, must be genuinely different from the "
        "parent (not a reformatting or comment change), and must still satisfy "
        "every hard requirement and the validator above.",
        "",
        "Output format (CRITICAL): start your reply with ```python on its own "
        "line and end it with ```. No prose before or after the code block.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------- reflection
REFLECTION_SYSTEM = (
    "You are analyzing one iteration of an evolutionary program-search loop. "
    "You will see a compact log of every rollout in the iteration: which score "
    "band the parent was in, what kind of mutation was attempted, and the "
    "outcome (invalid, sterile/redundant, or valid with its measured "
    "improvement). Your job is to find the single most dominant failure mode "
    "across these rollouts and propose one new mutation prompt that would "
    "counter it."
)

REFLECTION_USER_TEMPLATE = """Goal: {goal}

Rollout log for this iteration:
{rollout_log}

Score bands you may target: weak, good, elite, near_sota.

Identify the ONE dominant failure mode, decide which band it most belongs to,
and write a new, concrete mutation prompt (an instruction to give the model when
mutating a program in that band) that directly addresses it.

Respond with ONLY a JSON object, no prose and no markdown fences:
{{"failure_mode": "<short description>", "band": "<weak|good|elite|near_sota>", "prompt": "<the new mutation instruction>"}}"""


_BAND_SET = {WEAK, GOOD, ELITE, NEAR_SOTA}


def parse_reflection(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)      # first {...} blob
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    band = str(obj.get("band", "")).strip().lower()
    prompt = str(obj.get("prompt", "")).strip()
    if band not in _BAND_SET or not prompt:
        return None
    return {"failure_mode": str(obj.get("failure_mode", "")).strip(),
            "band": band, "prompt": prompt}