"""
Extract Python code from a model response. Reused from the TTT-local codebase.

Strategies, in order:
  1. A well-formed ```python ... ``` block (take the last one)
  2. An unterminated ```python ... <EOS> (model hit the token limit mid-block)
  3. A generic ``` ... ``` block
  4. The raw response if it smells like Python
Returns code with fences removed, or None.
"""

import re
from typing import Optional


def extract_python_code(response: str) -> Optional[str]:
    if response is None:
        return None

    # Drop any <think>...</think> reasoning (Qwen3, R1, gpt-oss, etc.).
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    if "<think>" in response and "</think>" not in response:
        return None  # never produced code, still inside thinking when truncated

    matches = re.findall(r"```python\s*\n?(.*?)```", response, re.DOTALL)
    if matches and matches[-1].strip():
        return matches[-1].strip()

    m = re.search(r"```python\s*\n?(.*)$", response, re.DOTALL)
    if m:
        code = re.sub(r"\n?```\s*$", "", m.group(1)).strip()
        if code:
            return code

    matches = re.findall(r"```\s*\n?(.*?)```", response, re.DOTALL)
    if matches and matches[-1].strip():
        return matches[-1].strip()

    stripped = (response or "").strip()
    if stripped.startswith(("import ", "from ", "def ", "class ", "#")):
        return stripped

    return None
