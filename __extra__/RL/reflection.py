"""
End-of-iteration reflection (step 9).

The LLM is shown a compact log of every rollout in the iteration and asked to
name the single dominant failure mode, the band it belongs to, and a new
mutation prompt to counter it. The returned prompt is appended as a new arm to
that band's pool, where the Thompson-sampling bandit will explore it on later
iterations.

This now builds the user message with build_reflection_user(...). The rewritten
prompts.py template carries extra fields (band instruction/task, the existing
arms in the pool); calling REFLECTION_USER_TEMPLATE.format(goal=, rollout_log=)
against it raises KeyError. The optional existing_arms_by_band / target_band are
defaulted so the frozen engine's reflect(self.llm, self.goal, rollouts) call
keeps working unchanged, while the RL path can pass the current pools so the
controller avoids emitting near-duplicate arms.
"""

from typing import Optional

from BanditEvolve.__extra__.RL.prompts import REFLECTION_SYSTEM, build_reflection_user, parse_reflection


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


def reflect(llm, goal: str, rollouts, existing_arms_by_band=None,
            target_band=None) -> Optional[dict]:
    """Returns {failure_mode, band, prompt} or None if the LLM response is
    unparseable. Never raises: a failed reflection just skips growing the pools.

    existing_arms_by_band: optional {band: [arm_text, ...]} passed to the
        controller so it does not re-propose an arm already in the pool.
    target_band: optional band to force (enables caller-side band rotation).
    """
    user = build_reflection_user(
        goal, _format_log(rollouts),
        existing_arms_by_band=existing_arms_by_band,
        target_band=target_band,
    )
    messages = [{"role": "system", "content": REFLECTION_SYSTEM},
                {"role": "user", "content": user}]
    try:
        text = llm.complete(messages)
    except Exception:
        return None
    return parse_reflection(text)
