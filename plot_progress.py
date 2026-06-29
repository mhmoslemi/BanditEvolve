"""
Plot search progress for several runs on one figure, per iteration:
  * best score ever seen    (cumulative max of all valid rollout scores)
  * top-5 mean              (cumulative mean of the 5 best scores ever seen)

Both curves come from the SAME pool of valid scores (the `value=` lines: the
bootstrap seeds plus every valid rollout), so the top-5 mean is always <= best.
The final iteration of every run is annotated with its best score at full
available precision (the `[iter N] best=` summary prints 6 decimals), with
offsets chosen so the labels don't overlap.

Usage:
    python plot_progress.py [out.png]
"""

import re
import sys

import matplotlib
matplotlib.use("Agg")            # headless: write a file, no display needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LOGDIR = "/work/mohammad/BanditEvolve/logs"
SOTA = 2.635983                  # known best sum of radii for n=26 (config target)

ITER_START = re.compile(r"^\[iter (\d+)\].*parents picked")   # start of iter N's rollouts
ITER_BEST = re.compile(r"^\[iter (\d+)\] best=([0-9]+\.[0-9]+)")  # iter N summary (6-dec best)
VALUE = re.compile(r"value=([0-9]+\.[0-9]+)")                # a valid rollout / seed score

# One entry per run.  `ann` = (dx, dy) text offset in points for the final best
# label, picked so the four labels never overlap.
RUNS = [
    dict(file="circle_packing_20260625_202129-yes0.log",
         label="no RL  (untuned band)",
         color="#1f77b4", ann=(-12, 28)),
    dict(file="circle_packing_20260626_201704-yes1.log",
         label="all-band RL GRPO  (untuned band)",
         color="#ff7f0e", ann=(10, -28)),
    dict(file="circle_packing_20260628_065511-yes2.log",
         label="all-band RL GRPO  (tuned band)",
         color="#2ca02c", ann=(-14, -34)),
    dict(file="circle_packing_20260629_053249.log",
         label="all-band RL A2C  (tuned band)",
         color="#d62728", ann=(12, 24)),
]


def parse(path):
    """Return (per_iter, best_raw):
      per_iter[i] = list of valid scores produced during iter i (seeds -> -1)
      best_raw[i] = the raw `best=` string printed in iter i's summary line
    """
    per_iter = {}
    best_raw = {}
    cur = -1                     # seeds / bootstrap land before iter 0
    with open(path) as f:
        for line in f:
            m = ITER_START.match(line)
            if m:
                cur = int(m.group(1))
                continue
            b = ITER_BEST.match(line)
            if b:
                best_raw[int(b.group(1))] = b.group(2)
                continue          # summary line carries best=/raw=, not a score
            for v in VALUE.findall(line):
                per_iter.setdefault(cur, []).append(float(v))
    return per_iter, best_raw


def progress(per_iter, best_raw):
    """Cumulative best and top-5 mean over the iterations that printed a summary."""
    pool = list(per_iter.get(-1, []))        # seed scores = the pre-iter-0 baseline
    xs, best, top5 = [], [], []
    for it in sorted(best_raw):
        pool.extend(per_iter.get(it, []))
        if not pool:
            continue
        top = sorted(pool, reverse=True)
        xs.append(it)
        best.append(top[0])
        top5.append(sum(top[:5]) / min(5, len(top)))
    return xs, best, top5


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "progress.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    for run in RUNS:
        path = f"{LOGDIR}/{run['file']}"
        per_iter, best_raw = parse(path)
        xs, best, top5 = progress(per_iter, best_raw)
        if not xs:
            print(f"no iterations parsed from {path}")
            continue

        c = run["color"]
        ax.plot(xs, best, "-o", ms=3.5, lw=1.8, color=c)            # best ever seen
        ax.plot(xs, top5, "--s", ms=3, lw=1.4, color=c, alpha=0.8)  # top-5 mean

        # annotate the final iteration's best at full (6-dec) precision
        fx, fy = xs[-1], best[-1]
        final_raw = best_raw[max(best_raw)]
        dx, dy = run["ann"]
        ax.annotate(
            final_raw,
            xy=(fx, fy), xycoords="data",
            xytext=(dx, dy), textcoords="offset points",
            fontsize=9, fontweight="bold", color=c,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c, lw=1.0, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.0),
        )
        print(f"{run['file']}: {len(xs)} iters, "
              f"final best = {final_raw}, final top-5 mean = {top5[-1]:.4f}")

    ax.axhline(SOTA, ls="--", lw=1.4, color="lightcoral")
    ax.set_xlabel("iteration")
    ax.set_ylabel("sum of radii")
    ax.set_title("Search progress — best ever seen & top-5 mean")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.08)           # room on the sides for the final-score labels

    # two legends: one mapping color -> run, one explaining the line styles
    run_handles = [Line2D([0], [0], color=r["color"], lw=2.2, label=r["label"])
                   for r in RUNS]
    style_handles = [
        Line2D([0], [0], color="black", lw=1.8, ls="-", marker="o", ms=3.5,
               label="best ever seen"),
        Line2D([0], [0], color="black", lw=1.4, ls="--", marker="s", ms=3,
               label="top-5 mean"),
        Line2D([0], [0], color="lightcoral", lw=1.4, ls="--",
               label=f"SOTA = {SOTA}"),
    ]
    leg1 = ax.legend(handles=run_handles, loc="lower right", fontsize=9, title="run")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
