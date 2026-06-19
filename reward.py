"""
Extract Python code from a model response. Reused from the TTT-local codebase,
loosened so a missing or oddly-tagged fence does not throw away an otherwise
usable program (the common "no_code" waste during bootstrap).

Strategies, in order:
  1. A well-formed ```python (or ```py / ```python3) block (take the last one)
  2. An unterminated ```python ... <EOS> (model hit the token limit mid-block)
  3. A generic ``` ... ``` block whose body looks like Python
  4. The whole response if it already looks like a Python program (def/import)
  5. A salvage: from the first top-level `import`/`def` line to the end
Returns code with fences removed, or None.
"""

import re
from typing import Optional


def _looks_like_python(text: str) -> bool:
    t = (text or "").lstrip()
    return t.startswith(("import ", "from ", "def ", "class ", "#", "@",
                         "import numpy", "try:"))


def extract_python_code(response: str) -> Optional[str]:
    if response is None:
        return None

    # Drop any <think>...</think> reasoning (Qwen3, R1, gpt-oss, etc.).
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # If thinking opened but never closed, the model never reached the code.
    if "<think>" in response and "</think>" not in response:
        return None

    # 1) ```python / ```py / ```python3 fenced block, take the LAST complete one
    matches = re.findall(r"```(?:python3?|py)\s*\n?(.*?)```", response,
                         re.DOTALL | re.IGNORECASE)
    if matches and matches[-1].strip():
        return matches[-1].strip()

    # 2) unterminated ```python ... <EOS>
    m = re.search(r"```(?:python3?|py)\s*\n?(.*)$", response,
                  re.DOTALL | re.IGNORECASE)
    if m:
        code = re.sub(r"\n?```\s*$", "", m.group(1)).strip()
        if code:
            return code

    # 3) any ``` ... ``` block, last one, only if it smells like Python
    matches = re.findall(r"```\s*\n?(.*?)```", response, re.DOTALL)
    for body in reversed(matches):
        if body.strip() and _looks_like_python(body):
            return body.strip()

    # 4) the whole response is already a program
    stripped = (response or "").strip()
    if _looks_like_python(stripped):
        return stripped

    # 5) salvage: from the first plausible Python start line to the end
    lines = (response or "").splitlines()
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith(("import ", "from ", "def ", "class ")):
            candidate = "\n".join(lines[i:]).strip()
            candidate = re.sub(r"\n?```\s*$", "", candidate).strip()
            if candidate:
                return candidate
    return None