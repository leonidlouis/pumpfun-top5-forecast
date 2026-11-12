#!/usr/bin/env python3
"""Synthetic pump.fun-like market in the kit's format (stdlib only), for exercising main.py on the final's shape.

    python3 tools/synth.py OUT_DIR [--seed N] [--history-hours 11] [--rate 300]

writes OUT_DIR/input/task.json, OUT_DIR/input/history.ndjson (one mark per 10 minutes from t0 to the cut, top 200)
and OUT_DIR/answers.json (the seven answer reads' charts, top 100, plus history_mints), with answer reads 7..13 h
after the cut, as in the final.

The process is deliberately not main.py's own model: each coin launches at ~28 SOL on its curve, most fizzle within
minutes, a few "runners" climb for a lognormal time to a heavy-tailed peak, then decay with noisy negative drift and
an occasional rug. A curve coin cannot pass the graduation cap (~400 SOL) without graduating to its pool; mayhem
coins are ranked only once graduated. Numbers are synthetic, never real data.
"""

import argparse
import json
import math
import os
import random
from datetime import datetime, timedelta, timezone

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
CURVE_MIN = math.log10(27.958993)    # a fresh curve's cap, log10 SOL
GRAD = math.log10(410.0)             # a curve completes about here
STEP_H = 1.0 / 6.0


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def mint(rng):
    return "".join(rng.choice(B58) for _ in range(40)) + "pump"


class Coin:
    def __init__(self, rng, born_h):
        self.mint = mint(rng)
        self.born = born_h
        self.mayhem = rng.random() < 0.05
        runner = rng.random() < 0.08
        if runner:
            self.gain = rng.expovariate(1 / 0.6)
            self.t_peak = math.exp(rng.gauss(math.log(1.2), 0.9))
        else:
            self.gain = rng.expovariate(1 / 0.18)
            self.t_peak = math.exp(rng.gauss(math.log(0.15), 0.6))
        self.decay = rng.uniform(0.02, 0.18)        # log10 per hour after the peak
        self.vol = rng.uniform(0.10, 0.35)           # log10 per sqrt(hour)
        self.x = CURVE_MIN
        self.noise = 0.0
        self.graduated = False
        self.rugged = False

    def step(self, rng, now_h):
        age = now_h - self.born
        if age < 0:
            return
        if age <= self.t_peak:
            trend = CURVE_MIN + self.gain * (age / self.t_peak)
        else:
            trend = CURVE_MIN + self.gain - self.decay * (age - self.t_peak)
        # AR(1) noise in log space, fat-tailed shocks
        shock = rng.gauss(0, 1) if rng.random() < 0.9 else rng.gauss(0, 3)
        self.noise = 0.97 * self.noise + self.vol * math.sqrt(STEP_H) * shock * (1.5 if age < 1 else 1.0)
        if not self.rugged and rng.random() < 0.02 * STEP_H:
            self.rugged = True
        x = trend + self.noise - (1.6 if self.rugged else 0.0)
        if not self.graduated:
            if x >= GRAD:
                self.graduated = True
            else:
                x = max(CURVE_MIN, x)
        self.x = x

    def ranked(self):
        return self.graduated or not self.mayhem

    def row(self, rank, t0, slot):
        lam = int(10 ** (self.x + 9))   # the cap already stands on the coin's own supply S
        created = t0 + timedelta(hours=self.born)
        return {"rank": rank, "mint": self.mint, "market_cap_lamports": str(lam),
                "venue": "pool" if self.graduated else "curve", "slot": slot, "name": "SYN", "symbol": "SYN",
                "created_at": iso(created), "complete": self.graduated, "mayhem": self.mayhem,
                "quote_reserves_lamports": str(lam // 8)}


def generate(out_dir, seed=1, history_hours=11.0, rate=300.0):
    rng = random.Random(seed)
    t0 = datetime(2030, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
    cut_h = history_hours
    reads_h = [cut_h + 7 + i for i in range(7)]
    end_h = reads_h[-1]
    coins = []
    marks = []
    answers = []
    n_steps = int(round(end_h / STEP_H))
    slot = 400_000_000
    for s in range(n_steps + 1):
        now = s * STEP_H
        # launches during the previous step (Poisson by thinning, mild daily cycle)
        if s > 0:
            lam = rate * STEP_H * (1 + 0.3 * math.sin(2 * math.pi * now / 24))
            n = 0
            L = math.exp(-lam)
            p = 1.0
            while True:
                p *= rng.random()
                if p < L:
                    break
                n += 1
            for _ in range(n):
                coins.append(Coin(rng, now - rng.random() * STEP_H))
        for c in coins:
            c.step(rng, now)
        slot += 1500
        ranked = sorted((c for c in coins if c.ranked()), key=lambda c: (-c.x, c.mint))
        at = t0 + timedelta(hours=now, seconds=5)
        if now < cut_h - 1e-9:
            rows = [c.row(i + 1, t0, slot) for i, c in enumerate(ranked[:200])]
            marks.append({"schema": "makefun/pumpfun-top5-forecast/mark/v1", "at": iso(at), "ok": True,
                          "error": None, "kind": "mark", "t0": iso(t0),
                          "counts": {"eligible": len(coins), "graduated": sum(c.graduated for c in coins),
                                     "excluded_non_sol": 0, "priced": len(coins), "refused": 0, "calls": 1,
                                     "http_429": 0, "stream_coverage": None},
                          "rows": rows})
        if any(abs(now - r) < 1e-6 for r in reads_h):
            answers.append({"at": iso(at), "chart": [{"mint": c.mint, "market_cap_lamports": str(int(10 ** (c.x + 9)))}
                                                     for c in ranked[:100]]})
    hist_mints = sorted({r["mint"] for m in marks for r in m["rows"]})
    task = {"schema": "makefun/pumpfun-top5-forecast/task/v1", "competition": "synthetic", "t0": iso(t0),
            "history_until": iso(t0 + timedelta(hours=cut_h)),
            "window": {"deadline": iso(t0 + timedelta(hours=cut_h + 1)), "answer_at": iso(t0 + timedelta(hours=end_h))},
            "reads": [iso(t0 + timedelta(hours=r)) for r in reads_h], "places": 5, "chart_size": 100,
            "size": {"edges": [round(2 + 0.05 * i, 2) for i in range(81)]}, "cadence_minutes": 10,
            "output": "forecast.json", "marks": len(marks)}
    os.makedirs(os.path.join(out_dir, "input"), exist_ok=True)
    with open(os.path.join(out_dir, "input", "task.json"), "w") as f:
        json.dump(task, f, indent=1)
    with open(os.path.join(out_dir, "input", "history.ndjson"), "w") as f:
        for m in marks:
            f.write(json.dumps(m, separators=(",", ":")) + "\n")
    with open(os.path.join(out_dir, "answers.json"), "w") as f:
        json.dump({"reads": answers, "history_mints": hist_mints}, f)
    return task, marks, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--history-hours", type=float, default=11.0)
    ap.add_argument("--rate", type=float, default=300.0)
    a = ap.parse_args()
    task, marks, answers = generate(a.out_dir, a.seed, a.history_hours, a.rate)
    print(f"{len(marks)} marks, {len(answers)} answer reads -> {a.out_dir}")


if __name__ == "__main__":
    main()
