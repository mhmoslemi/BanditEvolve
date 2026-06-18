"""
End-of-iteration reflection (step 9).

The LLM is shown a compact log of every rollout in the iteration and asked to
name the single dominant failure mode, the band it belongs to, and a new
mutation prompt to counter it. The returned prompt is appended as a new arm to
that band's pool, where the Thompson-sampling bandit will explore it on later
iterations.
"""

from typing import List, Optional

from prompts import REFLECTION_SYSTEM, REFLECTION_USER_TEMPLATE, parse_reflection


def _format_log(rollouts) -> str:
    lines = []
    for i, r in enumerate(rollouts):
        if r.kind == "explore":
            lines.append(f"[{i}] explore: outcome={r.outcome}"
                         + (f" value={r.value:.4f}" if r.value is not None else ""))
        else:
            extra = ""
            if r.dmu is not None:
                extra = f" dmu={r.dmu:+.4f} dsigma={r.dsigma:.4f}"
            lines.append(
                f"[{i}] mutate band={r.band} arm={r.arm_idx}({r.arm_source}) "
                f"outcome={r.outcome}{extra}"
            )
    return "\n".join(lines) if lines else "(no rollouts)"


def reflect(llm, goal: str, rollouts) -> Optional[dict]:
    """Returns {failure_mode, band, prompt} or None if the LLM response is
    unparseable. Never raises: a failed reflection just skips growing the pools."""
    user = REFLECTION_USER_TEMPLATE.format(goal=goal, rollout_log=_format_log(rollouts))
    messages = [{"role": "system", "content": REFLECTION_SYSTEM},
                {"role": "user", "content": user}]
    try:
        text = llm.complete(messages)
    except Exception:
        return None
    return parse_reflection(text)
