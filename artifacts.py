"""
Per-rollout run artifacts.

Independent of the live stdout log: this captures, on disk, exactly what went
into the policy and what came back for every rollout, so a finished run can be
inspected offline. The engine writes one directory per iteration, and inside it
one directory per rollout, holding the prompt, the raw response, the extracted
program, and the sandbox evaluation.

    <artifacts_dir>/<problem>_<ts>/
        iter_000/
            rollout_p0_k0/
                prompt.txt     # messages rendered role-by-role (human readable)
                prompt.json    # the raw messages list
                response.txt   # the raw LLM completion
                code.py        # extracted program (only if one was parsed)
                eval.json      # rollout record + sandbox EvalResult
            rollout_p0_k1/ ...
        seeds/                 # bootstrap seed generations (same layout)
            round_001/gen_0/ ...
"""

import json
import os
from dataclasses import asdict, is_dataclass


def _render_messages(messages) -> str:
    """Flatten a chat-messages list into a readable role/content transcript."""
    if not isinstance(messages, (list, tuple)):
        return str(messages)
    parts = []
    for m in messages:
        if isinstance(m, dict):
            parts.append(f"===== {m.get('role', '?')} =====\n{m.get('content', '')}")
        else:
            parts.append(str(m))
    return "\n\n".join(parts)


def _jsonable(obj):
    """Best-effort conversion to something json.dump can handle."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def save_rollout(rollout_dir, messages, response, rec=None, ev=None, code=None):
    """Write one rollout's artifacts into rollout_dir (created if needed)."""
    os.makedirs(rollout_dir, exist_ok=True)

    with open(os.path.join(rollout_dir, "prompt.txt"), "w") as f:
        f.write(_render_messages(messages))
    with open(os.path.join(rollout_dir, "prompt.json"), "w") as f:
        json.dump(_jsonable(messages), f, indent=2)

    with open(os.path.join(rollout_dir, "response.txt"), "w") as f:
        f.write(response if isinstance(response, str) else str(response or ""))

    if code:
        with open(os.path.join(rollout_dir, "code.py"), "w") as f:
            f.write(code)

    with open(os.path.join(rollout_dir, "eval.json"), "w") as f:
        json.dump({"record": _jsonable(rec), "eval": _jsonable(ev)}, f, indent=2)
