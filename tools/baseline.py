#!/usr/bin/env python3
"""A Python port of the house persistence baseline (kit starter/main.ts), for local comparison only.

Place k: 0.30 on the last mark's #k, 0.10 on #k-1 and #k+1; sizes a normal around log10 of #k's cap, sd 0.35,
each bin floored to 6 decimals; top5 the last mark's top 5.

    MAKEFUN_INPUT_DIR=... MAKEFUN_OUTPUT_DIR=... python3 tools/baseline.py
"""

import json
import math
import os


def baseline(task, last_rows):
    chart = sorted(last_rows, key=lambda r: r["rank"])
    places = task["places"]
    edges = task["size"]["edges"]
    probs, sizes = [], []

    def above(x, mean, sd):
        if x == -math.inf:
            return 1.0
        if x == math.inf:
            return 0.0
        return 0.5 * math.erfc((x - mean) / (sd * math.sqrt(2)))

    for k in range(1, places + 1):
        probs.append({"mint": chart[k - 1]["mint"], "place": k, "p": 0.3})
        if k > 1:
            probs.append({"mint": chart[k - 2]["mint"], "place": k, "p": 0.1})
        if k < len(chart):
            probs.append({"mint": chart[k]["mint"], "place": k, "p": 0.1})
        mean = math.log10(int(chart[k - 1]["market_cap_lamports"])) - 9
        for b in range(len(edges) + 1):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == len(edges) else edges[b]
            u = math.floor((above(lo, mean, 0.35) - above(hi, mean, 0.35)) * 1e6)
            if u > 0:
                sizes.append({"place": k, "bin": b, "p": u / 1e6})
    return {"probabilities": probs, "top5": [r["mint"] for r in chart[:places]], "sizes": sizes}


def main():
    ind = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    outd = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(ind, "task.json")) as f:
        task = json.load(f)
    last = None
    with open(os.path.join(ind, "history.ndjson")) as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    with open(os.path.join(outd, task["output"]), "w") as f:
        f.write(json.dumps(baseline(task, last["rows"])) + "\n")


if __name__ == "__main__":
    main()
