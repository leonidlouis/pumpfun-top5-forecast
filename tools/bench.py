#!/usr/bin/env python3
"""Synthetic benchmark: for each seed, generate a market (tools/synth.py), run main.py and the baseline port on its
history, score both on the continued process's answer reads (tools/score.py). Synthetic numbers only.

    python3 tools/bench.py [--seeds 1 2 3 ...] [--work /tmp/pf-bench]
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import score as scorer  # noqa: E402
import synth  # noqa: E402


def run(script, ind, outd):
    env = dict(os.environ, MAKEFUN_INPUT_DIR=ind, MAKEFUN_OUTPUT_DIR=outd)
    t = time.perf_counter()
    subprocess.run([sys.executable, script], env=env, check=True)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--work", default="/tmp/pf-bench")
    a = ap.parse_args()
    tot = {"main": [], "baseline": []}
    for seed in a.seeds:
        d = os.path.join(a.work, f"seed{seed}")
        synth.generate(d, seed=seed)
        ind = os.path.join(d, "input")
        with open(os.path.join(d, "answers.json")) as f:
            ans = json.load(f)
        line = [f"seed {seed}:"]
        for name, script in (("main", os.path.join(ROOT, "main.py")), ("baseline", os.path.join(HERE, "baseline.py"))):
            outd = os.path.join(d, "out-" + name)
            os.makedirs(outd, exist_ok=True)
            secs = run(script, ind, outd)
            with open(os.path.join(outd, "forecast.json")) as f:
                fc = json.load(f)
            s = scorer.score(fc, ans["reads"], ans["history_mints"])
            tot[name].append(s)
            line.append(f"{name} {s['log_score']:.3f} (order {s['order']:.3f}, size {s['size']:.3f}, agree {s['agreement']}, {secs:.1f}s)")
        print("  ".join(line))
    for name, ss in tot.items():
        n = len(ss)
        print(f"mean {name}: log {sum(s['log_score'] for s in ss) / n:.3f}  order {sum(s['order'] for s in ss) / n:.3f}"
              f"  size {sum(s['size'] for s in ss) / n:.3f}  agreement {sum(s['agreement'] for s in ss) / n:.1f}")


if __name__ == "__main__":
    main()
