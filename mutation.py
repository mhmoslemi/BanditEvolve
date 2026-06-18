"""
Build the chat messages for one mutation (step 6).

We reuse the problem's own task prompt (the same persona, rules, and validator
text the problem already defines for the parent) and append the evolution
context block: the band-specific freedom clause, the sampled arm's tactic, and
the parent / grandparent / archive-best / worst-but-valid exemplars.
"""

from prompts import evolution_block


def build_mutation_messages(problem, parent_ctx, grandparent_code, best_code,
                            worst_code, band, arm_template, max_chars=4000):
    base = problem.build_prompt(parent_ctx)        # list[{role, content}]
    block = evolution_block(
        parent_code=parent_ctx.code,
        grandparent_code=grandparent_code,
        best_code=best_code,
        worst_code=worst_code,
        band=band,
        arm_template=arm_template,
        max_chars=max_chars,
    )
    messages = [dict(m) for m in base]
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] = messages[-1]["content"] + "\n\n" + block
    else:
        messages.append({"role": "user", "content": block})
    return messages
