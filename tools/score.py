#!/usr/bin/env python3
"""The RULES.txt log score, re-implemented (stdlib only).

    python3 tools/score.py FORECAST.json ANSWERS.json [--edges-from TASK.json]

ANSWERS.json: {"reads": [{"chart": [{"mint", "market_cap_lamports"}, ...ranked, rank 1 first]}, ...],
               "history_mints": [...]}   (tools/synth.py writes this shape)

Per read: order score = sum over places k of ln(max(1e-4, p_k)) where p_k is the row's p for the coin actually at
place k, else (1 - sum of place k's surviving rows) / (chart coins not named at place k); rows naming a mint the
history never names are dropped first. Size score likewise over the 82 bins of log10(cap in SOL) (bin 0 below
edges[0], bin i in [edges[i-1], edges[i]), bin 81 from edges[-1]). Log score = mean over reads of order + size.
A chart with fewer than 5 coins scores the places it has.
"""

import argparse
import bisect
import json
import math

FLOOR = 1e-4
DEFAULT_EDGES = [round(2 + 0.05 * i, 2) for i in range(81)]


def size_bin(lamports, edges):
    x = math.log10(int(lamports)) - 9.0
    return bisect.bisect_right(edges, x)


def score(forecast, reads, history_mints=None, edges=None, places=5):
    edges = edges or DEFAULT_EDGES
    nb = len(edges) + 1
    hm = set(history_mints) if history_mints is not None else None
    prob = {k: {} for k in range(1, places + 1)}
    for r in forecast.get("probabilities", []):
        if hm is not None and r["mint"] not in hm:
            continue
        prob[r["place"]][r["mint"]] = float(r["p"])
    size = {k: {} for k in range(1, places + 1)}
    for r in forecast.get("sizes", []):
        size[r["place"]][r["bin"]] = float(r["p"])
    top5 = [m for m in forecast.get("top5", []) if hm is None or m in hm]
    per = []
    for rd in reads:
        chart = rd["chart"]
        mints = [c["mint"] for c in chart]
        o = s = 0.0
        agree = 0
        for k in range(1, min(places, len(chart)) + 1):
            m = mints[k - 1]
            named = prob[k]
            if m in named:
                p = named[m]
            else:
                rest = sum(1 for x in mints if x not in named)
                p = (1.0 - sum(named.values())) / rest if rest else 0.0
            o += math.log(max(FLOOR, p))
            b = size_bin(chart[k - 1]["market_cap_lamports"], edges)
            if b in size[k]:
                q = size[k][b]
            else:
                q = (1.0 - sum(size[k].values())) / (nb - len(size[k])) if nb > len(size[k]) else 0.0
            s += math.log(max(FLOOR, q))
        top_actual = mints[:places]
        for i, m in enumerate(top5):
            if m in top_actual:
                agree += 5 - abs(i - top_actual.index(m))
        per.append((o, s, agree))
    n = len(per) or 1
    return {"log_score": round(sum(o + s for o, s, _ in per) / n, 6),
            "order": round(sum(o for o, _, _ in per) / n, 6),
            "size": round(sum(s for _, s, _ in per) / n, 6),
            "agreement": sum(a for _, _, a in per),
            "per_read": [round(o + s, 6) for o, s, _ in per]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("forecast")
    ap.add_argument("answers")
    ap.add_argument("--edges-from")
    a = ap.parse_args()
    with open(a.forecast) as f:
        fc = json.load(f)
    with open(a.answers) as f:
        ans = json.load(f)
    edges = None
    if a.edges_from:
        with open(a.edges_from) as f:
            edges = json.load(f)["size"]["edges"]
    print(json.dumps(score(fc, ans["reads"], ans.get("history_mints"), edges)))


if __name__ == "__main__":
    main()
