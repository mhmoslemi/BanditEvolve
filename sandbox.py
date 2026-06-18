"""
Run model-generated code in a subprocess with a hard timeout.

Write the code to a temp file, spawn a Python subprocess that imports it, calls
the named entrypoint, and pickles the return value. The parent reads the pickle
on clean exit, or kills the child (and its process group) on timeout.

Reused almost verbatim from the TTT-local codebase: it is backend-agnostic and
has nothing to do with weight training, so it transfers directly.
"""

import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile


RUNNER_TEMPLATE = r'''
import os
import sys
import pickle
import traceback
import importlib.util

try:
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
except Exception:
    pass

PROGRAM_PATH = "__PROGRAM_PATH__"
FUNCTION_NAME = "__FUNCTION_NAME__"
RESULTS_PATH = "__RESULTS_PATH__"

sys.path.insert(0, os.path.dirname(PROGRAM_PATH))

try:
    spec = importlib.util.spec_from_file_location("program", PROGRAM_PATH)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    fn = getattr(program, FUNCTION_NAME)
    result = fn()
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({"ok": True, "value": result}, f)
except Exception as e:
    tb = traceback.format_exc()
    try:
        with open(RESULTS_PATH, "wb") as f:
            pickle.dump({"ok": False, "error": str(e), "traceback": tb}, f)
    except Exception:
        pass
    sys.stderr.write(tb)
'''


def _kill_tree(proc, pgid, hard=False):
    sig = signal.SIGKILL if hard else signal.SIGTERM
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
        except Exception:
            pass
    if shutil.which("pkill"):
        try:
            subprocess.run(
                ["pkill", "-KILL" if hard else "-TERM", "-P", str(proc.pid)],
                check=False,
            )
        except Exception:
            pass


def run_code(code: str, entrypoint: str, timeout_s: float, max_cpus: int = 2):
    """Execute `code` in a subprocess, call `entrypoint()`, return its value.

    Returns: {"ok": True, "value": ..., "stdout": ...}
          or {"ok": False, "error": ..., "stdout": ...}
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        program_path = f.name
        f.write(code)

    runner_src = (
        RUNNER_TEMPLATE
        .replace("__PROGRAM_PATH__", program_path)
        .replace("__FUNCTION_NAME__", entrypoint)
        .replace("__RESULTS_PATH__", program_path + ".pkl")
    )
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        runner_path = f.name
        f.write(runner_src)

    results_path = program_path + ".pkl"

    env = os.environ.copy()
    t = str(max(1, int(max_cpus)))
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
        env.setdefault(key, t)

    proc = subprocess.Popen(
        [sys.executable, runner_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, start_new_session=True,
    )
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    stdout_bytes = b""
    timed_out = False
    try:
        stdout_bytes, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc, pgid, hard=False)
        try:
            stdout_bytes, _ = proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            _kill_tree(proc, pgid, hard=True)
            try:
                stdout_bytes, _ = proc.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

    _kill_tree(proc, pgid, hard=True)
    stdout_text = stdout_bytes.decode(errors="ignore") if stdout_bytes else ""

    if timed_out:
        result = {"ok": False, "error": f"Timeout after {timeout_s}s", "stdout": stdout_text}
    elif not os.path.exists(results_path):
        result = {"ok": False, "error": f"No results (rc={proc.returncode})", "stdout": stdout_text}
    else:
        try:
            with open(results_path, "rb") as f:
                payload = pickle.load(f)
            payload["stdout"] = stdout_text
            result = payload
        except Exception as e:
            result = {"ok": False, "error": f"Failed to read results: {e}", "stdout": stdout_text}

    for p in [program_path, runner_path, results_path]:
        try:
            os.unlink(p)
        except (FileNotFoundError, OSError):
            pass
    return result
