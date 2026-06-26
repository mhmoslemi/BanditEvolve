"""
Plot REPRODUCIBLE search progress from a run's saved programs.

The logged `value=` scores are single-seed (eval seed 0). The generated programs
are stochastic, so that one score is optimistic and not what you get when you
re-run the program. This script RE-EVALUATES every saved program over K seeds
(using the project's own evaluate_candidate, so seeding/validation/scoring match
the engine exactly) and plots the MEAN (reproducible) score with a +/-1 std band:

  * best ever seen     = cumulative max of per-program MEAN score
  * avg of top-3 ever   = cumulative mean of the top-3 per-program MEAN scores

For contrast it also draws the optimistic single-seed best (the dashed curve the
log shows). The gap between the two is exactly the "I can't reproduce it" effect.

Per-program means are cached in the run dir (_repro_cache.json), so re-plots and
bumping --seeds are cheap.

Usage:
    python plot_progress_repro.py [RUN_DIR] [out.png] \
        [--seeds K] [--workers W] [--min-value V] [--config PATH]
"""

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problems.registry import get_problem
from evaluation import evaluate_candidate

SOTA = 2.635983


def latest_run_dir():
    cands = [d for d in glob.glob("runs/circle_packing_*") if os.path.isdir(d)]
    return max(cands, key=os.path.getmtime) if cands else None


def collect(run_dir, min_value):
    """code -> (first_iter, logged_single_seed_value) for every VALID saved program."""
    progs = {}
    for cf in glob.glob(os.path.join(run_dir, "iter_*", "*", "code.py")):
        m = re.search(r"iter_(\d+)", cf)
        it = int(m.group(1)) if m else 0
        evp = os.path.join(os.path.dirname(cf), "eval.json")
        try:
            evd = (json.load(open(evp)) or {}).get("eval") or {}
        except (OSError, ValueError):
            evd = {}
        if not evd.get("valid") or evd.get("value") is None:
            continue
        logged = float(evd["value"])
        if logged < min_value:
            continue
        try:
            code = open(cf).read()
        except OSError:
            continue
        if not code.strip():
            continue
        if code not in progs or it < progs[code][0]:
            progs[code] = (min(it, progs.get(code, (it, 0))[0]), logged)
    return progs


def reevaluate(progs, problem, seeds, timeout, workers, cache_path):
    """code -> (mean, std) over `seeds`, cached by (seeds, code) hash."""
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except (OSError, ValueError):
            cache = {}

    out, todo = {}, []
    for code in progs:
        h = hashlib.sha1((repr(seeds) + code).encode()).hexdigest()
        if h in cache:
            out[code] = (cache[h]["mean"], cache[h]["std"])
        else:
            todo.append((code, h))
    print(f"[repro] {len(out)} cached, {len(todo)} to evaluate "
          f"over {len(seeds)} seed(s) with {workers} worker(s) ...", flush=True)

    def work(code):
        ev = evaluate_candidate(problem, code, list(seeds), timeout)
        vals = [ev.per_seed[s] for s in seeds if s in ev.per_seed]
        mean = sum(vals) / len(vals) if vals else 0.0
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    total = len(todo)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, c): (c, h) for (c, h) in todo}
        done, best, t0, last = 0, float("-inf"), time.time(), 0.0
        for fut in as_completed(futs):
            c, h = futs[fut]
            try:
                mean, std = fut.result()
            except Exception:
                mean, std = 0.0, 0.0
            out[c] = (mean, std)
            cache[h] = {"mean": mean, "std": std}
            done += 1
            best = max(best, mean)
            # checkpoint the cache periodically so a crash/Ctrl-C keeps progress
            if done % 200 == 0:
                try:
                    json.dump(cache, open(cache_path, "w"))
                except OSError:
                    pass
            now = time.time()
            if now - last >= 1.0 or done == total:      # throttle to ~1 line/sec
                last = now
                el = now - t0
                rate = done / el if el > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                print(f"[repro]   {done}/{total} ({100 * done / total:4.1f}%)  "
                      f"elapsed {el:5.0f}s  eta {eta:5.0f}s  "
                      f"{rate:4.1f} prog/s  best={best:.4f}", flush=True)
    try:
        json.dump(cache, open(cache_path, "w"))
    except OSError:
        pass
    return out


def curves(progs, means, n_iters):
    """Per-iter cumulative best & top-3 of the MEAN scores (with spread), plus the
    optimistic single-seed best from the logged values."""
    by_iter = {}
    for code, (it, logged) in progs.items():
        mean, std = means[code]
        by_iter.setdefault(it, []).append((mean, std, logged))

    xs, best, best_sd, top3, top3_sd, opt = [], [], [], [], [], []
    pool = []
    for it in range(n_iters + 1):
        pool.extend(by_iter.get(it, []))
        if not pool:
            continue
        by_mean = sorted(pool, key=lambda t: t[0], reverse=True)
        xs.append(it)
        best.append(by_mean[0][0])
        best_sd.append(by_mean[0][1])
        top = by_mean[:3]
        top3.append(sum(t[0] for t in top) / len(top))
        top3_sd.append(sum(t[1] for t in top) / len(top))
        opt.append(max(t[2] for t in pool))           # optimistic single-seed best
    return xs, best, best_sd, top3, top3_sd, opt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="runs/<problem>_<ts> dir (default: latest)")
    ap.add_argument("out", nargs="?", default="progress_repro.png")
    ap.add_argument("--seeds", type=int, default=8, help="re-eval each program over seeds 0..K-1")
    ap.add_argument("--workers", type=int, default=16, help="parallel sandbox evaluations")
    ap.add_argument("--min-value", type=float, default=0.0,
                    help="skip programs whose logged value is below this (speeds it up; "
                         "low scorers never affect best/top-3 anyway)")
    ap.add_argument("--config", default="configs/circle_packing.yaml")
    args = ap.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    if not run_dir or not os.path.isdir(run_dir):
        print("no run dir found"); return
    merged = yaml.safe_load(open(args.config)) or {}
    problem = get_problem(merged.get("problem", "circle_packing"), merged)
    timeout = float(merged.get("timeout", 100.0))
    seeds = list(range(args.seeds))

    progs = collect(run_dir, args.min_value)
    if not progs:
        print(f"no valid programs in {run_dir}"); return
    n_iters = max(it for it, _ in progs.values())
    print(f"[repro] run_dir={run_dir}  programs={len(progs)}  iters=0..{n_iters}", flush=True)

    cache_path = os.path.join(run_dir, "_repro_cache.json")
    means = reevaluate(progs, problem, seeds, timeout, args.workers, cache_path)
    xs, best, best_sd, top3, top3_sd, opt = curves(progs, means, n_iters)
    if not xs:
        print("nothing to plot"); return

    print(f"[repro] final: best(mean over {args.seeds} seeds)={best[-1]:.4f} "
          f"+/-{best_sd[-1]:.4f}   optimistic single-seed best={opt[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b = [v for v in best]; bsd = [s for s in best_sd]
    ax.fill_between(xs, [m - s for m, s in zip(b, bsd)], [m + s for m, s in zip(b, bsd)],
                    color="C0", alpha=0.15)
    ax.plot(xs, best, "-o", ms=3, lw=1.7, color="C0", label=f"best ever (mean of {args.seeds} seeds)")
    t = [v for v in top3]; tsd = [s for s in top3_sd]
    ax.fill_between(xs, [m - s for m, s in zip(t, tsd)], [m + s for m, s in zip(t, tsd)],
                    color="C1", alpha=0.15)
    ax.plot(xs, top3, "-s", ms=3, lw=1.7, color="C1", label="avg top-3 ever (mean)")
    ax.plot(xs, opt, ":", lw=1.3, color="gray", label="best ever (optimistic, 1 seed)")
    ax.axhline(SOTA, ls="--", lw=1.4, color="lightcoral", label=f"SOTA = {SOTA}")

    ax.set_xlabel("iteration")
    ax.set_ylabel("sum of radii")
    ax.set_title(f"Reproducible search progress  ({os.path.basename(run_dir)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[repro] saved {args.out}")


if __name__ == "__main__":
    main()
