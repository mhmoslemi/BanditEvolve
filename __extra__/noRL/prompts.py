"""
All prompt text in one place.

  - BAND_FREEDOM:   the per-band mutation-freedom clause injected into every
                    mutation prompt for that band, independent of which arm is
                    sampled.
  - SEED_TEMPLATES: the initial arms in each band's pool (step 5). Reflection
                    adds more arms over time (step 9).
  - evolution_block: the parent / grandparent / best / worst-but-valid exemplar
                    block appended to the problem's own task prompt (step 6).
  - REFLECTION_*:   end-of-iteration meta prompt, its builder, and JSON parser
                    (step 9).

Call-site changes vs the previous version:
  * Build the reflection user message with build_reflection_user(...) instead
    of REFLECTION_USER_TEMPLATE.format(...). The template now expects extra
    context (existing arms, optional target band) that the builder fills in.
  * evolution_block gained two optional kwargs (include_strategy_scaffold,
    worst_diagnosis). Existing calls keep working unchanged.
"""

import json
from typing import Optional

from bands import WEAK, GOOD, ELITE, NEAR_SOTA


BAND_FREEDOM = {
    WEAK: (
        "CURRENT STATUS: WEAK.\n"
        "This program is fundamentally underperforming compared to the rest of the search archive. "
        "DIRECTIVE: You have absolute freedom to rewrite. Discard the core algorithm if necessary. "
        "Shift the paradigm completely (e.g., if gradient-based, pivot to simulated annealing, basin hopping, or greedy packing). "
        "Do not waste time tweaking constants; rethink the mathematical foundation."
    ),
    GOOD: (
        "CURRENT STATUS: GOOD (MID-PACK).\n"
        "This program has a viable foundation but lacks the sophistication to reach elite performance. "
        "DIRECTIVE: Keep the core architecture but replace or overhaul a major sub-component. "
        "Introduce advanced heuristics, hybridize the optimizer (e.g., add a local polish step after a global search), "
        "or fundamentally change how boundary conditions are enforced."
    ),
    ELITE: (
        "CURRENT STATUS: ELITE.\n"
        "This program is highly competitive. Structural overhauls are too risky and will likely degrade performance. "
        "DIRECTIVE: Execute a targeted, surgical strike on computational bottlenecks or numerical instability. "
        "Focus on vectorization, tightening solver tolerances, improving gradient approximations, or "
        "adding targeted heuristics for edge-case circles."
    ),
    NEAR_SOTA: (
        "CURRENT STATUS: NEAR STATE-OF-THE-ART (SOTA).\n"
        "This program is at the absolute frontier of known solutions and is almost certainly the best program in the archive. "
        "DIRECTIVE: STRICT CONSERVATISM. Do not alter the algorithmic structure or rename the entrypoints. "
        "Only propose ADDITIVE, REVERSIBLE changes layered on top of the existing solution: a final projection pass, "
        "infinitesimal radius inflation that keeps centers fixed, deterministic edge-case cleanup, or a precise "
        "hyperparameter sweep around existing values. "
        "The new program MUST first reproduce the parent's result, then attempt the refinement, and return the refined "
        "result only if it is valid and does not regress; otherwise it must fall back to the parent's result. "
        "Treat the parent as a baseline you are guaranteed to keep, never one you might lose."
    ),
}


SEED_TEMPLATES = {
    WEAK: [
        "Discard the current initialization strategy. Implement a completely different geometric approach (e.g., hexagonal grid seeding, spiral placement) before applying the optimizer.",
        "The current solver approach is failing. Switch to a completely different optimization library or algorithm (e.g., `scipy.optimize.differential_evolution` or a physics-based spring repulsion model).",
    ],
    GOOD: [
        "Identify the primary source of overlapping or out-of-bounds errors. Introduce a penalty function or a dedicated subroutine to resolve these constraints more gracefully.",
        "Add a two-stage optimization pipeline: a coarse global search to find the general layout, followed by a tight local optimization (e.g., L-BFGS-B) to maximize radii.",
    ],
    ELITE: [
        "Look at the mathematical bottleneck. Vectorize the overlap constraint calculations to allow the optimizer to run more iterations in the same amount of time.",
        "Implement a 'shake' or 'jiggle' heuristic that slightly perturbs the smallest circles and re-optimizes, helping the algorithm escape deep local minima.",
    ],
    NEAR_SOTA: [
        "Fine-tune the optimization tolerances (`ftol`, `gtol`, `eps`) and the bounds definitions to squeeze out the last fraction of radius without breaking constraints. Keep the parent's result as a guaranteed fallback.",
        "Implement a final, conservative 'cleanup' pass that only attempts to inflate circles infinitesimally without moving their centers, returning the refined result only if valid and otherwise returning the parent's result unchanged.",
    ],
}


_BAND_SET = {WEAK, GOOD, ELITE, NEAR_SOTA}


# ---------------------------------------------------------------- truncation
def _truncate_code(code: Optional[str], max_chars: int = 4000) -> str:
    """Middle-truncate on line boundaries.

    Tail truncation deletes the entrypoint / return that defines the program's
    contract, and cutting mid-token feeds the model syntactically broken code.
    Keeping head + tail preserves imports, signatures and the final return while
    dropping the (usually least load-bearing) middle.
    """
    code = code or ""
    if len(code) <= max_chars:
        return code

    head_budget = (max_chars * 2) // 3
    tail_budget = max_chars - head_budget

    head_cut = code.rfind("\n", 0, head_budget)
    if head_cut <= 0:
        head_cut = head_budget

    tail_anchor = len(code) - tail_budget
    tail_start = code.find("\n", tail_anchor)
    if tail_start == -1:
        tail_start = tail_anchor

    head = code[:head_cut].rstrip()
    tail = code[tail_start:].lstrip("\n")
    return f"{head}\n\n# ... [middle truncated to fit context] ...\n\n{tail}"


# ---------------------------------------------------------------- mutation
def evolution_block(parent_code, grandparent_code, best_code, worst_code,
                    band, arm_template, max_chars=4000,
                    include_strategy_scaffold=True, worst_diagnosis=None):
    """Build the exemplar + directive block appended to the task prompt.

    include_strategy_scaffold: if your generation backend runs Qwen3 with native
        thinking enabled, set this False so you do not get a redundant second
        reasoning pass. The <think> trace replaces the <strategy> block.
    worst_diagnosis: optional one-line, machine-generated reason the negative
        exemplar is bad (band, score delta, structural tag). Far higher signal
        per token than asking the model to reverse-engineer the flaw.
    """
    def trunc(c):
        return _truncate_code(c, max_chars)

    # At NEAR_SOTA the parent is almost always the SOTA itself, so showing the
    # SOTA block and telling the model "do not copy verbatim" directly fights
    # the conservatism directive. Suppress both there.
    show_sota = bool(best_code) and band != NEAR_SOTA

    parts = [
        "---",
        f"## MUTATION DIRECTIVE (Band: {band.upper()})",
        BAND_FREEDOM.get(band, ""),
        "",
        "### Specific Tactic to Apply:",
        f"> {arm_template}",
        "",
        "## TARGET PROGRAM TO IMPROVE (The Parent)",
        f"```python\n{trunc(parent_code)}\n```",
    ]

    if grandparent_code:
        parts += [
            "",
            "## HISTORICAL CONTEXT (The Grandparent)",
            "Review this to understand the trajectory of recent changes.",
            f"```python\n{trunc(grandparent_code)}\n```",
        ]

    if show_sota:
        parts += [
            "",
            "## GLOBAL SOTA (Reference Only)",
            "This is the current best known solution. Understand its logic, but DO NOT copy it verbatim. Use it as an architectural north star.",
            f"```python\n{trunc(best_code)}\n```",
        ]
    elif best_code and band == NEAR_SOTA:
        parts += [
            "",
            "## NOTE",
            "The target program above is at or near the global frontier. Preserve its structure exactly and refine additively.",
        ]

    if worst_code:
        neg = [
            "",
            "## NEGATIVE EXEMPLAR (What Not To Do)",
            "This program is technically valid but performs terribly. Avoid its structural flaws.",
        ]
        if worst_diagnosis:
            neg.append(f"Diagnosis: {worst_diagnosis}")
        neg.append(f"```python\n{trunc(worst_code)}\n```")
        parts += neg

    if include_strategy_scaffold:
        parts += [
            "---",
            "## OUTPUT CONSTRAINTS (CRITICAL)",
            "1. You MUST think step-by-step first. Enclose your reasoning strictly within `<strategy>` and `</strategy>` tags.",
            "2. Analyze the differences between the parent, the SOTA, and the worst exemplars.",
            "3. Detail exactly how you will implement the specific tactic requested.",
            "4. After your strategy, output ONE complete, valid Python program.",
            "5. The code must be enclosed in ```python and ```, and it must be the LAST fenced block in your response.",
            "6. Do not include any explanation or prose outside of the strategy tags or code blocks.",
            "7. Ensure all required entrypoints and validation rules are perfectly preserved.",
        ]
    else:
        parts += [
            "---",
            "## OUTPUT CONSTRAINTS (CRITICAL)",
            "1. Output ONE complete, valid Python program and nothing else.",
            "2. The code must be enclosed in ```python and ```, and it must be the LAST fenced block in your response.",
            "3. Do not echo, restate, or lightly edit the parent before your final answer. Emit only the final program.",
            "4. Ensure all required entrypoints and validation rules are perfectly preserved.",
        ]

    return "\n".join(parts)


# ---------------------------------------------------------------- reflection
REFLECTION_SYSTEM = (
    "You are the Meta-Search Controller for an evolutionary program-search algorithm. "
    "Your objective is to analyze a batch of recent rollouts (program mutations) and synthesize "
    "a new, highly effective mutation prompt to feed back into the system's Thompson sampling bandit.\n\n"
    "You will receive a rollout log detailing the parent's score band, the mutation applied, and the outcome "
    "(Invalid, Sterile/Redundant, or Valid with its measured improvement). You will also receive the mutation "
    "prompts that already exist in the target band's pool.\n\n"
    "Your task: Identify the single most dominant bottleneck or failure mode in this iteration, then write ONE "
    "concrete, actionable mutation prompt that overcomes it AND is semantically distinct from every prompt already "
    "in the pool. Do not paraphrase an existing arm."
)

# Kept for backwards compatibility / readability. Prefer build_reflection_user().
REFLECTION_USER_TEMPLATE = """Goal: {goal}

## Rollout Log for Current Iteration:
{rollout_log}

## Valid Score Bands:
["weak", "good", "elite", "near_sota"]

{band_instruction}

## Existing mutation prompts already in the target band's pool (DO NOT duplicate or paraphrase these):
{existing_arms}

## Task:
1. Identify the ONE dominant failure mode.
2. {band_task}
3. Write a new mutation instruction (prompt) that directly addresses it. Make it actionable, specific, and clearly distinct from every existing prompt listed above and from generic advice.

## Output Format:
Respond strictly with a single JSON object. No markdown, no prose, no code blocks.

Example:
{{
  "failure_mode": "Most valid mutations were sterile: they re-derived the parent's layout with cosmetic changes and produced no measurable improvement.",
  "band": "good",
  "prompt": "Force structural divergence: identify the single most constrained element in the parent's layout and rebuild the construction order around relieving that constraint first, rather than optimizing the existing ordering."
}}"""


def _render_existing_arms(arms) -> str:
    if not arms:
        return "(none yet)"
    return "\n".join(f"- {a.strip()}" for a in arms if a and a.strip()) or "(none yet)"


def build_reflection_user(goal, rollout_log, existing_arms_by_band=None,
                          target_band=None) -> str:
    """Assemble the reflection user message.

    existing_arms_by_band: optional {band: [arm_text, ...]} so the controller can
        avoid emitting near-duplicate arms (which fragment the bandit posterior).
    target_band: if set, the controller must write for this band (lets the caller
        rotate bands and stop the most-visible band, usually WEAK, from starving
        the others). If None, the controller self-selects the band.
    """
    existing_arms_by_band = existing_arms_by_band or {}

    if target_band:
        tb = str(target_band).strip().lower()
        band_instruction = (
            f"## Target band (FIXED): {tb}\n"
            f"You MUST set \"band\" to \"{tb}\". Write the prompt for programs in this band only."
        )
        band_task = f'Set "band" to "{tb}" (this is fixed; do not choose another band).'
        existing = _render_existing_arms(existing_arms_by_band.get(tb))
    else:
        band_instruction = (
            "## Band selection\n"
            "Select the single band this failure mode most afflicts and set \"band\" to it."
        )
        band_task = "Select the most appropriate band for this correction."
        # Show all arms grouped, since the model picks the band itself.
        if existing_arms_by_band:
            blocks = []
            for b in (WEAK, GOOD, ELITE, NEAR_SOTA):
                blocks.append(f"[{b}]\n{_render_existing_arms(existing_arms_by_band.get(b))}")
            existing = "\n\n".join(blocks)
        else:
            existing = "(none yet)"

    return REFLECTION_USER_TEMPLATE.format(
        goal=goal,
        rollout_log=rollout_log,
        band_instruction=band_instruction,
        band_task=band_task,
        existing_arms=existing,
    )


def parse_reflection(text: str) -> Optional[dict]:
    """Parse the first valid JSON object from the controller's output.

    Uses raw_decode from the first '{' instead of a greedy {.*} regex. The greedy
    version spanned from the first brace to the last brace in the whole response,
    so any extra brace-bearing text (a stray example, fenced output, trailing
    prose) made json.loads fail and silently dropped the arm. raw_decode stops at
    the end of the first complete object and ignores whatever follows.
    """
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    band = str(obj.get("band", "")).strip().lower()
    prompt = str(obj.get("prompt", "")).strip()
    if band not in _BAND_SET or not prompt:
        return None

    return {
        "failure_mode": str(obj.get("failure_mode", "")).strip(),
        "band": band,
        "prompt": prompt,
    }