"""
Progress bar helper, same pattern as the TTT gen_workers.make_progress_bar:
use tqdm when available, fall back to a coarse print bar otherwise. The wrapper
exposes update / write / close so callers can interleave detail lines with the
bar without clobbering it (tqdm.write keeps the bar pinned to the bottom).
"""

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False


class _PrintBar:
    def __init__(self, total, desc):
        self.total = max(int(total), 1)
        self.desc = desc
        self.n = 0
        self._last = -1

    def update(self, k=1):
        self.n += k
        d = int(20 * self.n / self.total)
        if d != self._last:
            self._last = d
            pct = int(100 * self.n / self.total)
            print(f"[{self.desc}] {self.n}/{self.total} ({pct}%)", flush=True)

    def write(self, msg):
        print(msg, flush=True)

    def close(self):
        if self.n < self.total:
            print(f"[{self.desc}] {self.n}/{self.total}", flush=True)


class _TqdmBar:
    def __init__(self, total, desc):
        self.bar = tqdm(total=int(max(total, 1)), desc=desc, unit="roll",
                        leave=False, dynamic_ncols=True)

    def update(self, k=1):
        self.bar.update(k)

    def write(self, msg):
        tqdm.write(msg)

    def close(self):
        self.bar.close()


def make_bar(total, desc="progress"):
    return _TqdmBar(total, desc) if _HAS_TQDM else _PrintBar(total, desc)