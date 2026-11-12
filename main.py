#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: a self-calibrating Monte Carlo forecast (Python 3.13, stdlib only).

Reads $MAKEFUN_INPUT_DIR (default /input): task.json and history.ndjson.
Writes $MAKEFUN_OUTPUT_DIR (default /output)/<task.output>: {"probabilities", "top5", "sizes"}.

Model, in short:
  * existing coins: a fat-tailed (Student-t, nu=3) random walk in log10(cap in SOL) from the coin's last
    observation, drift mu per hour and scale sigma per sqrt(hour) fitted on the history's own (coin, lag) pairs
    and shrunk toward priors, plus a death hazard (dropping off the top-200 chart) fitted the same way;
  * coins born after the history ends: a cohort bootstrap, one history launch cohort sampled per future launch
    slot, its coins' caps taken at the age the newcomers will have at the read (extrapolated by the random walk
    beyond the oldest observed age);
  * each simulation ranks existing + newcomer caps at one answer read and records who holds places 1..5 and
    their caps; the order probabilities are the hit shares (shrunk, newcomers left unnamed) and the sizes are
    kernel-smoothed histograms.
No network, no model calls, deterministic (fixed seed, stable tie-breaks).
"""

import bisect
import heapq
import json
import math
import os
import random
from datetime import datetime

SEED = 20261112
NU = 3.0                     # Student-t degrees of freedom of the log-cap increments
T_MAD = 0.7649               # MAD of a standard t(3) variate (normal: 0.6745)
T_CLIP = 4.0                 # increments are clipped at 4 scale units: the max over ~200 coins must stay sane
MU_PRIOR = -0.04             # log10 per hour
SIGMA_PRIOR = 0.22           # log10 per sqrt(hour)
HAZARD_PRIOR = 0.03          # deaths per hour among top-50 coins
PRIOR_WEIGHT = 25.0          # prior strength, in distinct coins
MU_RANGE = (-0.30, 0.10)
SIGMA_RANGE = (0.08, 0.60)
HAZARD_RANGE = (0.0, 0.25)
MIN_PAIR_AGE_H = 1.0         # pairs whose earlier observation is younger than this are not used for mu/sigma
TOP_FOR_PAIRS = 50
MAX_EXISTING = 250
WORK_BUDGET = 2_400_000      # simulations x simulated coins
MIN_SIMS, MAX_SIMS = 1500, 6000
EPS_ORDER = 0.12             # share of each place moved off the MC to a broad prior
MIN_UNNAMED = 0.08           # every place leaves at least this unnamed
SIZE_KERNEL_SD = 0.12
SIZE_UNIFORM = 0.03
FINE = 0.01                  # fine histogram cell, log10 units
FINE_LO, FINE_HI = 0.0, 8.0
MICRO = 1_000_000
PLACEHOLDERS = ["1" * 31 + c for c in "23456789"]


# ----------------------------------------------------------------------------------------------------- input

def parse_time(s):
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def log_cap(lamports):
    try:
        v = int(str(lamports))
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return math.log10(v) - 9.0


def load(input_dir):
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    marks = []
    path = os.path.join(input_dir, "history.ndjson")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(m, dict) or m.get("ok") is False:
                    continue
                at = parse_time(m.get("at"))
                if at is None:
                    continue
                rows = m.get("rows") if isinstance(m.get("rows"), list) else []
                marks.append((at, rows))
    marks.sort(key=lambda x: x[0])
    return task, marks


class Coin:
    __slots__ = ("mint", "created", "obs", "venue", "peak", "first_rank")

    def __init__(self, mint, created):
        self.mint = mint
        self.created = created
        self.obs = []          # (mark index, log10 cap)
        self.venue = "curve"
        self.peak = -1e9
        self.first_rank = 10 ** 6


def build(task, marks):
    t0 = parse_time(task.get("t0"))
    if t0 is None:
        t0 = marks[0][0] if marks else 0.0
    H = 3600.0
    times = []      # hours since t0
    charts = []     # per mark: list of (mint, logcap) by rank
    floors = []     # per mark: log cap of the 200th coin, or None when the chart is not full
    coins = {}
    for mi, (at, rows) in enumerate(marks):
        good = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            mint = r.get("mint")
            lc = log_cap(r.get("market_cap_lamports"))
            if not isinstance(mint, str) or not mint or lc is None:
                continue
            rank = r.get("rank") if isinstance(r.get("rank"), int) else 10 ** 6
            good.append((rank, -lc, mint, lc, r))
        good.sort()
        chart = []
        seen = set()
        for rank, _, mint, lc, r in good:
            if mint in seen:
                continue
            seen.add(mint)
            c = coins.get(mint)
            if c is None:
                created = parse_time(r.get("created_at"))
                created = (created - t0) / H if created is not None else (at - t0) / H
                c = Coin(mint, created)
                coins[mint] = c
            c.obs.append((mi, lc))
            c.venue = r.get("venue") or c.venue
            c.peak = max(c.peak, lc)
            c.first_rank = min(c.first_rank, len(chart) + 1)
            chart.append((mint, lc))
        times.append((at - t0) / H)
        charts.append(chart)
        floors.append(chart[-1][1] if len(chart) >= 200 else None)
    return t0, times, charts, floors, coins


# ------------------------------------------------------------------------------------------------ statistics

def wmedian(pairs):
    """Weighted median of (value, weight) pairs."""
    if not pairs:
        return None
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def age_factor(age_h):
    """Younger coins are more volatile: a mild power law in age, 1 at 3 h."""
    return clamp((3.0 / max(age_h, 0.1)) ** 0.25, 0.8, 1.8)


def cap_factor(lc):
    """Bigger caps are a little steadier."""
    return clamp(1.0 - 0.08 * (lc - 3.5), 0.85, 1.2)


def estimate_dynamics(times, charts, coins):
    """Fit mu (log10/h), sigma (log10/sqrt h) and the death hazard (1/h) on the history's own pairs."""
    n = len(times)
    info = {"pairs": 0, "coins": 0, "dropouts": 0}
    if n < 2:
        return MU_PRIOR, SIGMA_PRIOR, HAZARD_PRIOR, info
    span = times[-1] - times[0]
    cad = max(1e-6, span / (n - 1))
    steps = set()
    for lag_h in (1, 2, 3, 4, 5, 6):
        s = int(round(lag_h / cad))
        if 1 <= s <= n - 1:
            steps.add(s)
    if span < 1.0:
        steps.update(range(1, n))
    tau = max(2.0, span / 2.0)
    pos = [dict(c) for c in charts]
    obs = []            # (lag_h, d, weight, age)
    expo = 0.0
    dropped = 0.0
    used = {}
    for s in sorted(steps):
        for i in range(n - s):
            j = i + s
            lag = times[j] - times[i]
            if lag <= 0:
                continue
            w = math.exp(-(times[-1] - times[j]) / tau)
            full_j = len(charts[j]) >= 200
            for mint, lc in charts[i][:TOP_FOR_PAIRS]:
                age = times[i] - coins[mint].created
                if age < MIN_PAIR_AGE_H:
                    continue
                later = pos[j].get(mint)
                expo += w * lag
                used[mint] = used.get(mint, 0) + 1
                if later is None:
                    if full_j:
                        dropped += w
                    continue
                obs.append((lag, later - lc, w, age))
    info["pairs"] = len(obs)
    info["coins"] = len(used)
    info["dropouts"] = round(dropped, 3)
    k = len(used)
    if not obs or k == 0:
        return MU_PRIOR, SIGMA_PRIOR, HAZARD_PRIOR, info
    sw_d = sum(w * d for lag, d, w, _ in obs)
    sw_l = sum(w * lag for lag, d, w, _ in obs)
    mu_e = sw_d / sw_l
    # robust scale: weighted median of |residual| / sqrt(lag), in t(3) units, corrected to the prior's age
    res = [(abs(d - mu_e * lag) / math.sqrt(lag) / age_factor(age), w) for lag, d, w, age in obs]
    sig_e = wmedian(res) / T_MAD
    haz_e = dropped / expo if expo > 0 else HAZARD_PRIOR
    a = k / (k + PRIOR_WEIGHT)
    mu = clamp(a * mu_e + (1 - a) * MU_PRIOR, *MU_RANGE)
    sigma = clamp(math.exp(a * math.log(max(sig_e, 1e-3)) + (1 - a) * math.log(SIGMA_PRIOR)), *SIGMA_RANGE)
    hazard = clamp(a * haz_e + (1 - a) * HAZARD_PRIOR, *HAZARD_RANGE)
    info.update(mu_emp=round(mu_e, 4), sigma_emp=round(sig_e, 4), hazard_emp=round(haz_e, 4))
    return mu, sigma, hazard, info


# ---------------------------------------------------------------------------------------------- simulation

def t_pool(rng, size=1 << 16):
    """A fixed pool of standardized Student-t(3) draws (unit MAD-matched to a normal), indexed at random."""
    out = []
    for _ in range(size):
        z = rng.gauss(0.0, 1.0)
        chi = rng.gauss(0.0, 1.0) ** 2 + rng.gauss(0.0, 1.0) ** 2 + rng.gauss(0.0, 1.0) ** 2
        t = z / math.sqrt(max(chi, 1e-12) / NU) * 0.6745 / T_MAD
        if -T_CLIP <= t <= T_CLIP:
            out.append(t)
        else:
            out.append(math.copysign(T_CLIP, t))
    return out


def cohort_table(times, charts, coins, width_h):
    """cohorts[c] = list of (age_h, caps sorted desc) per mark, for coins created in [c*w, (c+1)*w)."""
    cohorts = {}
    for mi, chart in enumerate(charts):
        t = times[mi]
        by = {}
        for mint, lc in chart:
            cr = coins[mint].created
            c = int(math.floor(cr / width_h)) if cr >= 0 else -1
            if c < 0:
                continue
            by.setdefault(c, []).append(lc)
        ncoh = int(math.floor(t / width_h))   # cohorts fully born before this mark
        for c in range(ncoh):
            age = t - (c + 0.5) * width_h
            caps = sorted(by.get(c, []), reverse=True)[:8]
            cohorts.setdefault(c, []).append((age, caps))
    return cohorts


def newcomer_plan(cohorts, t_last, t_read, width_h):
    """For each future launch slot between the last mark and the read: (inclusion prob, candidates)."""
    plan = []
    if t_read <= t_last or not cohorts:
        return plan
    start = t_last
    while start < t_read - 1e-9:
        end = min(start + width_h, t_read)
        frac = (end - start) / width_h
        age = t_read - (start + end) / 2.0
        slack = max(width_h, 0.3 * age)
        cands = []
        for c in sorted(cohorts):
            obs = cohorts[c]
            best = None
            for a, caps in obs:
                if a <= age + width_h / 2 and (best is None or abs(a - age) < abs(best[0] - age)):
                    best = (a, caps)
            if best is not None and age - best[0] <= slack:
                cands.append((max(0.0, age - best[0]), best[1], best[0]))
        if not cands:   # nothing observed that old: every cohort at its oldest, extrapolated
            for c in sorted(cohorts):
                a, caps = max(cohorts[c], key=lambda x: x[0])
                cands.append((max(0.0, age - a), caps, a))
        plan.append((frac, cands))
        start = end
    return plan


def simulate(task, times, charts, floors, coins, rng):
    reads = [parse_time(r) for r in task.get("reads") or []]
    t0 = parse_time(task.get("t0"))
    reads_h = [(r - t0) / 3600.0 for r in reads if r is not None and t0 is not None]
    if not reads_h:
        hu = parse_time(task.get("history_until"))
        reads_h = [((hu - t0) / 3600.0) if hu is not None and t0 is not None else (times[-1] if times else 0.0)]
    places = int(task.get("places") or 5)
    mu, sigma, hazard, info = estimate_dynamics(times, charts, coins)

    t_last = times[-1] if times else 0.0
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    width = 1.0 if span >= 3.0 else max(1.0 / 6.0, (span / max(1, len(times) - 1)) if len(times) > 1 else 1.0 / 6.0)

    # existing coins: start state at the last mark
    last_chart = charts[-1] if charts else []
    in_last = {m: lc for m, lc in last_chart}
    floor_last = floors[-1] if floors else None
    low_last = last_chart[-1][1] if last_chart else 1.0
    below = (floor_last if floor_last is not None else low_last) - 0.3
    starts = []
    for mint in coins:
        c = coins[mint]
        if mint in in_last:
            starts.append((in_last[mint], mint))
        else:
            # off the chart at the last mark: at most its floor; only keep coins that once ranked high
            if c.first_rank <= 20:
                starts.append((min(c.obs[-1][1], below), mint))
    starts.sort(key=lambda x: (-x[0], x[1]))
    starts = starts[:MAX_EXISTING]
    mints = [m for _, m in starts]

    # per coin, per read: loc, scale, death probability
    params = []
    for x0, mint in starts:
        c = coins[mint]
        age_last = max(0.0, t_last - c.created)
        vf = (0.9 if c.venue == "pool" else 1.15) * cap_factor(x0)
        per = []
        for tr in reads_h:
            h = max(0.0, tr - t_last)
            mid_age = age_last + h / 2.0
            sc = sigma * age_factor(mid_age) * vf * math.sqrt(h)
            pd = 1.0 - math.exp(-hazard * h)
            per.append((x0 + mu * h, sc, pd))
        params.append(per)

    plans = [newcomer_plan(cohort_table(times, charts, coins, width), t_last, tr, width) for tr in reads_h]
    n_new = max((sum(len(c[0][1]) if c else 0 for _, c in p) for p in plans), default=0)
    per_sim = len(starts) + max(1, n_new)
    n_sims = int(clamp(WORK_BUDGET // per_sim, MIN_SIMS, MAX_SIMS))
    n_sims -= n_sims % len(reads_h)

    tp = t_pool(rng)
    tn = len(tp)
    rnd = rng.random
    hits = [dict() for _ in range(places)]
    new_hits = [0] * places
    nfine = int(round((FINE_HI - FINE_LO) / FINE))
    fine = [[0] * nfine for _ in range(places)]
    filled = [0] * places
    nlargest = heapq.nlargest
    for s in range(n_sims):
        r = s % len(reads_h)
        vals = []
        app = vals.append
        for i, per in enumerate(params):
            loc, sc, pd = per[r]
            v = loc + sc * tp[int(rnd() * tn)]
            if rnd() < pd:
                v -= 1.0 + rnd()
            app((v, i))
        for frac, cands in plans[r]:
            if frac < 1.0 and rnd() >= frac:
                continue
            dt, caps, a_obs = cands[int(rnd() * len(cands))]
            if dt > 0.02:
                sc = sigma * age_factor(a_obs + dt / 2.0) * math.sqrt(dt)
                pd = 1.0 - math.exp(-hazard * dt)
                for lc in caps:
                    v = lc + mu * dt + sc * tp[int(rnd() * tn)]
                    if rnd() < pd:
                        v -= 1.0 + rnd()
                    app((v, -1))
            else:
                for lc in caps:
                    app((lc + 0.02 * tp[int(rnd() * tn)], -1))
        top = nlargest(places, vals)
        for k, (v, i) in enumerate(top):
            if i < 0:
                new_hits[k] += 1
            else:
                hk = hits[k]
                hk[i] = hk.get(i, 0) + 1
            cell = int((v - FINE_LO) / FINE)
            fine[k][0 if cell < 0 else nfine - 1 if cell >= nfine else cell] += 1
            filled[k] += 1
    info.update(mu=round(mu, 4), sigma=round(sigma, 4), hazard=round(hazard, 4), sims=n_sims,
                existing=len(starts), width_h=round(width, 4))
    return dict(mints=mints, starts=starts, hits=hits, new_hits=new_hits, fine=fine, filled=filled,
                n_sims=n_sims, places=places, info=info)


# -------------------------------------------------------------------------------------------------- output

def floor6(p):
    return math.floor(p * MICRO) / MICRO


def order_rows(sim, chart_size):
    places = sim["places"]
    mints = sim["mints"]
    n = max(1, sim["n_sims"])
    # broad prior over existing coins: by their rank at the cut
    prior = [1.0 / (i + 3.0) for i in range(len(mints))]
    zp = sum(prior) or 1.0
    prior = [p / zp for p in prior]
    rows = []
    for k in range(places):
        hk = sim["hits"][k]
        mc_new = sim["new_hits"][k] / n
        q = {}
        for i in range(len(mints)):
            q[i] = (1.0 - EPS_ORDER) * hk.get(i, 0) / n + EPS_ORDER * (1.0 - mc_new) * prior[i]
        cand = sorted(q, key=lambda i: (-q[i], mints[i]))[:100]
        total = sum(q[i] for i in cand)
        cap = 1.0 - MIN_UNNAMED
        scale = min(1.0, cap / total) if total > 0 else 1.0
        named = {i: q[i] * scale for i in cand}
        # a coin is worth naming only above the share it would get unnamed (one pass; about half of the
        # named coins are assumed to be off the answer's chart, so the share is not overstated)
        unnamed = 1.0 - sum(named.values())
        u = unnamed / max(chart_size / 2.0, chart_size - len(named) / 2.0)
        named = {i: p for i, p in named.items() if p >= u}
        for i in sorted(named, key=lambda i: (-named[i], mints[i])):
            p = floor6(named[i])
            if p > 0:
                rows.append({"mint": mints[i], "place": k + 1, "p": p})
    return rows


def pick_top5(rows, sim, coins, places):
    by_place = {}
    for r in rows:
        by_place.setdefault(r["place"], []).append(r)
    out = []
    for k in range(1, places + 1):
        for r in sorted(by_place.get(k, []), key=lambda r: (-r["p"], r["mint"])):
            if r["mint"] not in out:
                out.append(r["mint"])
                break
    for _, m in sim["starts"]:
        if len(out) >= 5:
            break
        if m not in out:
            out.append(m)
    for m in sorted(coins, key=lambda m: (-coins[m].peak, m)):
        if len(out) >= 5:
            break
        if m not in out:
            out.append(m)
    for m in PLACEHOLDERS:
        if len(out) >= 5:
            break
        if m not in out:
            out.append(m)
    return out[:5]


def normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def size_rows(sim, edges):
    places = sim["places"]
    nb = len(edges) + 1
    nfine = len(sim["fine"][0]) if sim["fine"] else 0
    # kernel weights: fine cell centre -> mass per bin
    bounds = [-math.inf] + list(edges) + [math.inf]
    kern = []
    for cell in range(nfine):
        x = FINE_LO + (cell + 0.5) * FINE
        row = []
        prev = 0.0
        for b in range(nb):
            hi = bounds[b + 1]
            c = 1.0 if hi == math.inf else normal_cdf((hi - x) / SIZE_KERNEL_SD)
            row.append(c - prev)
            prev = c
        kern.append(row)
    rows = []
    for k in range(places):
        dens = [0.0] * nb
        tot = sim["filled"][k]
        if tot > 0:
            for cell, cnt in enumerate(sim["fine"][k]):
                if cnt:
                    kr = kern[cell]
                    for b in range(nb):
                        dens[b] += cnt * kr[b]
            dens = [d / tot for d in dens]
            dens = [(1.0 - SIZE_UNIFORM) * d + SIZE_UNIFORM / nb for d in dens]
        else:
            dens = [1.0 / nb] * nb
        for b in range(nb):
            p = floor6(dens[b])
            if p > 0:
                rows.append({"place": k + 1, "bin": b, "p": p})
    return rows


def forecast(task, marks, seed=SEED):
    rng = random.Random(seed)
    t0, times, charts, floors, coins = build(task, marks)
    edges = list((task.get("size") or {}).get("edges") or [2.0 + 0.05 * i for i in range(81)])
    chart_size = int(task.get("chart_size") or 100)
    sim = simulate(task, times, charts, floors, coins, rng)
    rows = order_rows(sim, chart_size)
    top5 = pick_top5(rows, sim, coins, sim["places"])
    sizes = size_rows(sim, edges)
    return {"probabilities": rows, "top5": top5, "sizes": sizes}, sim["info"]


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    task, marks = load(input_dir)
    out, info = forecast(task, marks)
    name = task.get("output") or "forecast.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
        f.write(json.dumps(out, separators=(",", ":")) + "\n")
    if os.environ.get("PF_DEBUG"):
        print(json.dumps(info, sort_keys=True))


if __name__ == "__main__":
    main()
