"""
Entry point.

    python main.py --problem circle_packing
    python main.py --problem circle_packing --llm-backend dummy   # offline wiring test

Logging: the whole run is teed to a .log file under logs/ (override the path with
the RUN_LOG env var). Everything that reaches stdout/stderr -- every print(), the
logging module, and uncaught tracebacks -- is mirrored to that file and flushed on
every write, so `tail -f` shows output with no buffering delay.
"""

import datetime
import os
import sys
import warnings

# transformers emits FutureWarnings (attention-mask API) on every generate; mute.
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import load_config
from llm import make_llm
from engine import Engine
from problems.registry import get_problem


class _Tee:
    """Mirror a text stream to a log file, flushing both on every write.

    Flushing per-write is what gives "no delay": even print() calls without
    flush=True land in the file (and on screen) immediately, so a follower like
    `tail -f run.log` stays current instead of waiting on a 4-8KB stdio buffer.
    """

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        n = self._stream.write(data)
        self._stream.flush()
        self._fh.write(data)
        self._fh.flush()
        return n

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    # keep the object stream-like for callers that probe the terminal (tqdm,
    # color detection, libraries that call fileno() to dup the fd, etc.)
    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _setup_logging(problem_name):
    """Tee stdout+stderr to logs/<problem>_<ts>.log and route logging there too.

    Returns (path, file_handle); the handle is closed by the caller on exit.
    """
    path = os.environ.get("RUN_LOG")
    if not path:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("logs", f"{problem_name}_{ts}.log")

    # buffering=1 = line-buffered file; the Tee also flushes explicitly per write.
    fh = open(path, "a", buffering=1, encoding="utf-8")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    # NOTE: deliberately do NOT touch logging config here. Any library that
    # already logs (huggingface_hub, urllib3, ...) keeps its own level and its
    # output is teed via stderr as-is. Raising the root logger to INFO would
    # unleash a flood of HTTP "GET https://huggingface.co/..." lines into the log.

    print(f"[log] writing run to {os.path.abspath(path)}", flush=True)
    return path, fh, orig_out, orig_err


def main():
    cfg, merged = load_config()

    # Set up the run log as early as possible so every line below is captured.
    log_path, log_fh, orig_out, orig_err = _setup_logging(cfg.problem)
    try:
        _run(cfg, merged)
    except BaseException:
        # write the traceback through the teed stderr BEFORE closing the file,
        # otherwise the interpreter prints it after teardown and it's lost.
        import traceback
        traceback.print_exc()
        raise
    finally:
        # restore the real streams before closing the file, so the interpreter's
        # shutdown flush doesn't write to a closed handle (that errors as exit 120).
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = orig_out, orig_err
        log_fh.close()


def _run(cfg, merged):
    problem = get_problem(cfg.problem, merged)

    # let the dummy LLM know the entrypoint so it returns a parseable stub
    setattr(cfg, "_entrypoint", getattr(problem, "entrypoint", "run"))
    llm = make_llm(cfg)

    print("=" * 64)
    print("Band-bandit evolutionary search")
    print(f"  problem        : {cfg.problem}"
          + (f" ({cfg.problem_type})" if cfg.problem_type else ""))
    print(f"  metric         : {problem.metric_name} "
          f"({'maximize' if problem.maximize else 'minimize'})")
    print(f"  llm            : {cfg.llm_backend}:{cfg.llm_model}")
    print(f"  iters          : {cfg.num_iters}")
    print(f"  parents x roll : {cfg.num_parents} x {cfg.rollouts_per_parent} "
          f"= {cfg.num_parents * cfg.rollouts_per_parent}/iter")
    print(f"  eval seeds     : {cfg.num_eval_seeds}   explore eps: {cfg.explore_eps}")
    print("=" * 64)

    engine = Engine(cfg, problem, llm)
    best = engine.run()

    print("\n" + "=" * 64)
    if best is not None:
        raw = (f"  (raw {problem.metric_name} = {best.raw_score:.6f})"
               if best.raw_score is not None else "")
        print(f"best reward: {best.value:.6f}{raw}   found at iter {best.timestep}")
        print("band arms:", engine.bandit.summary())
        print("band delta stats:", engine.band_stats.summary())
        print("\n--- best code ---\n" + best.code + "\n--- end ---")
    else:
        print("no valid solution produced")


if __name__ == "__main__":
    main()