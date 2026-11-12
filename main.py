#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: a self-calibrating Monte Carlo forecast (Python 3.13, stdlib only).

Reads $MAKEFUN_INPUT_DIR (default /input): task.json and history.ndjson.
Writes $MAKEFUN_OUTPUT_DIR (default /output)/<task.output>: {"probabilities", "top5", "sizes"}.

Model, in short:
  * existing coins: a fat-tailed (Student-t, nu=3, clipped) walk in log10(cap in SOL) from the coin's last
    observation. Its drift mu (+ beta per log10 of cap: big coins may decay differently), its scale sigma at a 1 h
    lag and the exponent H of the scale's growth with the lag (sigma * h**H; 0.5 is a random walk) are fitted on
    the history's own (coin, lag) pairs of top-50 coins and shrunk toward priors; the hazard of dying (falling off
    the top-200 chart) is fitted on the same pairs;
  * coins born after the history ends: a cohort bootstrap, one history launch cohort sampled per future launch
    slot, its coins' caps taken at the age the newcomers will have at the read (extrapolated by the walk beyond
    the oldest age observed);
  * each simulation ranks existing + newcomer caps at one answer read (cycling over the reads) and records who
    holds places 1..5, their caps and which existing coins clear the chart's floor;
  * order: per place, the hit shares shrunk toward a broad prior; the named set (<= 100 rows) is the prefix that
    maximises the forecast's own expected log score, given each coin's chance of being on the answer's chart
    (naming a likely chart coin at a small p takes it out of the unnamed pool newcomers share);
  * sizes: per place, the simulated log caps, Gaussian-kernel smoothed, mixed with a little uniform.
No network, no model calls, deterministic (fixed seed, stable tie-breaks, no wall clock).
"""

import heapq
import json
import math
import os
import random
from datetime import datetime

SEED = 20261112
NU = 3.0                     # Student-t degrees of freedom of the log-cap increments
T_MAD = 0.7649               # MAD of a standard t(3) variate (normal: 0.6745)
T_CLIP = 4.0                 # increments clipped at 4 scale units: the max over ~200 coins must stay sane
MU_PRIOR = -0.04             # log10 per hour
SIGMA_PRIOR = 0.22           # log10 at a 1 h lag
H_PRIOR = 0.5                # scale grows as lag**H (random walk)
HAZARD_PRIOR = 0.03          # deaths per hour among top-50 coins
PRIOR_WEIGHT = 25.0          # prior strength, in distinct coins
MU_RANGE = (-0.30, 0.10)
BETA_RANGE = (-0.06, 0.03)   # drift per hour per log10 of cap above CAP_REF
GAMMA_RANGE = (-0.15, 0.15)  # drift per hour per unit of ln(age / AGE_REF)
JUMP_RANGE = (-0.50, 0.20)   # one-off change within the first hour (regression to the mean of a top coin)
DEATH_DROP = 1.0             # a dropout counts as a death only when the later floor is this far below the coin
CAP_REF = 3.5
AGE_REF = 3.0
SIGMA_RANGE = (0.08, 0.60)
H_RANGE = (0.30, 0.60)
HAZARD_RANGE = (0.0, 0.25)
MIN_PAIR_AGE_H = 1.0         # pairs whose earlier observation is younger than this are not used
TOP_FOR_PAIRS = 50
MAX_EXISTING = 250
WORK_BUDGET = 6_000_000      # simulations x simulated coins
MIN_SIMS, MAX_SIMS = 2100, 21000
EPS_ORDER = 0.12             # share of each place moved off the MC to a broad prior
MIN_UNNAMED = 0.08           # every place leaves at least this unnamed
P_MIN = 3e-4                 # the smallest p a named row carries
CHART_TRUST = 0.35           # discount on the simulated chance of being on the answer's chart (it overstates it
                             # on synthetic data, and a wrong one costs the floor)
MAX_ROWS = 100
SIZE_KERNEL_SD = 0.15          # log10; a little wider than 0.12: the MC is somewhat overconfident on sizes
SIZE_UNIFORM = 0.03
FINE = 0.01                  # fine histogram cell, log10 units
FINE_LO, FINE_HI = 0.0, 8.0
MICRO = 1_000_000
PLACEHOLDERS = ["1" * 31 + c for c in "23456789"]
SCORE_FLOOR = 1e-4


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
    """Marks -> times (h since t0), charts [(mint, logcap)] by rank, 200th-coin floors, coins by mint."""
    t0 = parse_time(task.get("t0"))
    if t0 is None:
        t0 = marks[0][0] if marks else 0.0
    times, charts, floors, coins = [], [], [], {}
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
        good.sort(key=lambda g: (g[0], g[1], g[2]))
        chart = []
        seen = set()
        for rank, _, mint, lc, r in good:
            if mint in seen:
                continue
            seen.add(mint)
            c = coins.get(mint)
            if c is None:
                created = parse_time(r.get("created_at"))
                created = (created - t0) / 3600.0 if created is not None else (at - t0) / 3600.0
                c = Coin(mint, created)
                coins[mint] = c
            c.obs.append((mi, lc))
            if isinstance(r.get("venue"), str):
                c.venue = r["venue"]
            c.peak = max(c.peak, lc)
            c.first_rank = min(c.first_rank, len(chart) + 1)
            chart.append((mint, lc))
        times.append((at - t0) / 3600.0)
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
    return clamp(1.0 - 0.08 * (lc - CAP_REF), 0.85, 1.2)


class Dyn:
    """The fitted dynamics. Change over h hours: jump min(1, h) + the integral of the drift per hour
    mu + beta (x - CAP_REF) + gamma ln(age / AGE_REF) as the coin ages; scale(h) = sigma h**H; a death hazard
    per hour (a death costs 1 to 2 log10)."""

    def __init__(self, mu=MU_PRIOR, beta=0.0, gamma=0.0, sigma=SIGMA_PRIOR, H=H_PRIOR, hazard=HAZARD_PRIOR,
                 jump=0.0):
        self.mu, self.beta, self.gamma, self.jump = mu, beta, gamma, jump
        self.sigma, self.H, self.hazard = sigma, H, hazard

    def drift(self, h, x, age):
        if h <= 0:
            return 0.0
        a = max(age, 0.1)
        b = a + h
        int_ln = (b * math.log(b) - b) - (a * math.log(a) - a)       # integral of ln(age) over the horizon
        return (self.jump * min(1.0, h)
                + (self.mu + self.beta * (x - CAP_REF) - self.gamma * math.log(AGE_REF)) * h + self.gamma * int_ln)

    def scale(self, h):
        return self.sigma * (h ** self.H) if h > 0 else 0.0

    def p_death(self, h):
        return 1.0 - math.exp(-self.hazard * max(0.0, h))


def solve(A, b):
    """Gaussian elimination with partial pivoting; None when singular."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def huber_wls(X, ys, ws, iters=8):
    """Robust weighted least squares y ~ X coef (X rows include the intercept; Huber weights, MAD scale)."""
    p = len(X[0]) if X else 0
    coef = [0.0] * p
    hw = list(ws)
    for _ in range(iters):
        A = [[sum(w * x[i] * x[j] for w, x in zip(hw, X)) for j in range(p)] for i in range(p)]
        b = [sum(w * x[i] * y for w, x, y in zip(hw, X, ys)) for i in range(p)]
        sol = solve(A, b)
        if sol is None:
            break
        coef = sol
        res = [y - sum(c * xi for c, xi in zip(coef, x)) for x, y in zip(X, ys)]
        s = wmedian([(abs(r), w) for r, w in zip(res, ws)]) or 1e-6
        k = 1.5 * s / 0.6745
        hw = [w * (1.0 if abs(r) <= k else k / abs(r)) for r, w in zip(res, ws)]
    return coef


def estimate_dynamics(times, charts, coins):
    """Fit the Dyn parameters on the history's own (coin, lag) pairs of top-50 coins aged >= 1 h."""
    n = len(times)
    info = {"pairs": 0, "coins": 0, "dropouts": 0.0}
    if n < 2:
        return Dyn(), info
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
    obs = []            # (step, lag_h, d, weight, age, x0)
    expo = dropped = 0.0
    used = set()
    for s in sorted(steps):
        for i in range(n - s):
            j = i + s
            lag = times[j] - times[i]
            if lag <= 0:
                continue
            w = math.exp(-(times[-1] - times[j]) / tau)
            floor_j = charts[j][-1][1] if len(charts[j]) >= 200 else -math.inf
            for mint, lc in charts[i][:TOP_FOR_PAIRS]:
                age = times[i] - coins[mint].created
                if age < MIN_PAIR_AGE_H:
                    continue
                later = pos[j].get(mint)
                used.add(mint)
                if floor_j <= lc - DEATH_DROP:      # a fall of DEATH_DROP would show: the pair is exposed
                    expo += w * lag
                    if later is None:
                        dropped += w
                if later is None:
                    continue
                obs.append((s, lag, later - lc, w, age, lc))
    k = len(used)
    info.update(pairs=len(obs), coins=k, dropouts=round(dropped, 3))
    if not obs or k == 0:
        return Dyn(), info
    a = k / (k + PRIOR_WEIGHT)
    # change d over a lag: robust regression on a one-off jump (min(1, lag)) and per-hour terms lag,
    # lag (x0 - CAP_REF) and lag ln(mid-lag age / AGE_REF), each pair weighted by 1/lag (its variance grows
    # with the lag); the jump and slopes are shrunk toward 0 and mu toward the prior. With a single lag the
    # jump and mu cannot be told apart: no jump then.
    lags = sorted({s for s, *_ in obs})
    X = [(lag, lag * (x0 - CAP_REF), lag * math.log((age + lag / 2.0) / AGE_REF))
         + ((min(1.0, lag),) if len(lags) >= 2 else ()) for _, lag, _, _, age, x0 in obs]
    ys = [d for _, _, d, *_ in obs]
    ws = [w / lag for _, lag, _, w, _, _ in obs]
    coef = huber_wls(X, ys, ws)
    mu_e, beta_e, gamma_e = coef[0], coef[1], coef[2]
    jump_e = coef[3] if len(coef) > 3 else 0.0
    mu = clamp(a * mu_e + (1 - a) * MU_PRIOR, *MU_RANGE)
    beta = clamp(a * beta_e, *BETA_RANGE)
    gamma = clamp(a * gamma_e, *GAMMA_RANGE)
    jump = clamp(a * jump_e, *JUMP_RANGE)
    dyn0 = Dyn(mu, beta, gamma, jump=jump)
    # scale per lag bucket: robust scale of residuals (age-corrected, in t(3) units), then log s = log sigma + H log L
    buckets = {}
    for s, lag, d, w, age, x0 in obs:
        r = (d - dyn0.drift(lag, x0, age)) / age_factor(age) / cap_factor(x0)
        buckets.setdefault(s, []).append((abs(r), w, lag))
    pts = []
    for s in sorted(buckets):
        b = buckets[s]
        if len(b) < 20:
            continue
        sc = wmedian([(r, w) for r, w, _ in b]) / T_MAD
        lag = sum(l for _, _, l in b) / len(b)
        if sc > 0:
            pts.append((math.log(lag), math.log(sc), len(b)))
    if len(pts) >= 2 and max(p[0] for p in pts) - min(p[0] for p in pts) > 0.5:
        H_e, ls1 = None, None
        sw = sum(p[2] for p in pts)
        mx = sum(p[0] * p[2] for p in pts) / sw
        my = sum(p[1] * p[2] for p in pts) / sw
        sxx = sum(p[2] * (p[0] - mx) ** 2 for p in pts)
        H_e = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in pts) / sxx
        ls1 = my - H_e * mx
        H = clamp(a * H_e + (1 - a) * H_PRIOR, *H_RANGE)
        sig_e = math.exp(ls1)
    elif pts:
        H_e = None
        H = H_PRIOR
        sig_e = math.exp(pts[0][1]) / math.exp(pts[0][0]) ** H
    else:
        H_e = None
        H = H_PRIOR
        sig_e = SIGMA_PRIOR
    sigma = clamp(math.exp(a * math.log(max(sig_e, 1e-3)) + (1 - a) * math.log(SIGMA_PRIOR)), *SIGMA_RANGE)
    haz_e = dropped / expo if expo > 0 else HAZARD_PRIOR
    hazard = clamp(a * haz_e + (1 - a) * HAZARD_PRIOR, *HAZARD_RANGE)
    info.update(mu_emp=round(mu_e, 4), beta_emp=round(beta_e, 4), gamma_emp=round(gamma_e, 4),
                jump_emp=round(jump_e, 4), sigma_emp=round(sig_e, 4),
                H_emp=None if H_e is None else round(H_e, 4), hazard_emp=round(haz_e, 4))
    return Dyn(mu, beta, gamma, sigma, H, hazard, jump), info


# ---------------------------------------------------------------------------------------------- simulation

def t_pool(rng, size=1 << 16):
    """A fixed pool of Student-t(3) draws scaled to a normal's MAD and clipped, indexed at random."""
    out = []
    for _ in range(size):
        z = rng.gauss(0.0, 1.0)
        chi = rng.gauss(0.0, 1.0) ** 2 + rng.gauss(0.0, 1.0) ** 2 + rng.gauss(0.0, 1.0) ** 2
        t = z / math.sqrt(max(chi, 1e-12) / NU) * 0.6745 / T_MAD
        out.append(clamp(t, -T_CLIP, T_CLIP))
    return out


def cohort_table(times, charts, coins, width_h):
    """cohorts[c] = [(age_h, caps desc)] per mark, for coins created in [c w, (c+1) w) after t0."""
    cohorts = {}
    for mi, chart in enumerate(charts):
        t = times[mi]
        by = {}
        for mint, lc in chart:
            cr = coins[mint].created
            if cr < 0:
                continue
            by.setdefault(int(math.floor(cr / width_h)), []).append(lc)
        for c in range(int(math.floor(t / width_h))):      # cohorts fully born before this mark
            caps = sorted(by.get(c, []), reverse=True)[:8]
            cohorts.setdefault(c, []).append((t - (c + 0.5) * width_h, caps))
    return cohorts


def newcomer_plan(cohorts, t_last, t_read, width_h):
    """For each future launch slot between the last mark and the read: (inclusion prob, candidates), where a
    candidate is (extra hours to extrapolate, caps, observed age)."""
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
            best = None
            for a, caps in cohorts[c]:
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


def read_hours(task, times):
    t0 = parse_time(task.get("t0"))
    reads = [parse_time(r) for r in task.get("reads") or []] if isinstance(task.get("reads"), list) else []
    out = [(r - t0) / 3600.0 for r in reads if r is not None and t0 is not None]
    if not out:
        hu = parse_time(task.get("history_until"))
        out = [((hu - t0) / 3600.0) if hu is not None and t0 is not None else (times[-1] if times else 0.0)]
    return out


def chart_floor(charts, chart_size):
    """log cap of the answer chart's last coin, from the recent marks (None: every ranked coin is on it)."""
    vals = [c[chart_size - 1][1] for c in charts[-6:] if len(c) >= chart_size]
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def simulate(task, times, charts, floors, coins, rng):
    reads_h = read_hours(task, times)
    places = int(task.get("places") or 5)
    chart_size = int(task.get("chart_size") or 100)
    dyn, info = estimate_dynamics(times, charts, coins)

    t_last = times[-1] if times else 0.0
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    width = 1.0 if span >= 3.0 else max(1.0 / 6.0, span / max(1, len(times) - 1)) if len(times) > 1 else 1.0 / 6.0

    # existing coins: start state at the last mark
    last_chart = charts[-1] if charts else []
    in_last = dict(last_chart)
    floor_last = floors[-1] if floors else None
    low_last = last_chart[-1][1] if last_chart else 1.0
    below = (floor_last if floor_last is not None else low_last) - 0.3
    starts = []
    for mint, c in coins.items():
        if mint in in_last:
            starts.append((in_last[mint], mint))
        elif c.first_rank <= 20:     # off the chart at the last mark: at most its floor; kept if it once ranked high
            starts.append((min(c.obs[-1][1], below), mint))
    starts.sort(key=lambda x: (-x[0], x[1]))
    starts = starts[:MAX_EXISTING]
    mints = [m for _, m in starts]

    params = []       # per coin, per read: (loc, scale, death probability)
    for x0, mint in starts:
        c = coins[mint]
        age_last = max(0.0, t_last - c.created)
        vf = (0.9 if c.venue == "pool" else 1.15) * cap_factor(x0)
        per = []
        for tr in reads_h:
            h = max(0.0, tr - t_last)
            per.append((x0 + dyn.drift(h, x0, age_last), dyn.scale(h) * age_factor(age_last + h / 2.0) * vf, dyn.p_death(h)))
        params.append(per)

    cohorts = cohort_table(times, charts, coins, width)
    plans = []
    for tr in reads_h:
        plan = []
        for frac, cands in newcomer_plan(cohorts, t_last, tr, width):
            prepared = []
            for dt, caps, a_obs in cands:
                if dt > 0.02:
                    sc = dyn.scale(dt) * age_factor(a_obs + dt / 2.0)
                    prepared.append([(lc + dyn.drift(dt, lc, a_obs), sc) for lc in caps] + [dyn.p_death(dt)])
                else:
                    prepared.append([(lc, 0.02) for lc in caps] + [0.0])
            plan.append((frac, prepared))
        plans.append(plan)
    n_new = max((sum(len(c[0]) - 1 if c else 0 for _, c in p) for p in plans), default=0)
    per_sim = len(starts) + max(1, n_new)
    n_sims = int(clamp(WORK_BUDGET // per_sim, MIN_SIMS, MAX_SIMS))
    n_sims -= n_sims % len(reads_h)

    cfloor = chart_floor(charts, chart_size)
    cfl = -math.inf if cfloor is None else cfloor
    tp = t_pool(rng)
    tn = len(tp)
    rnd = rng.random
    hits = [dict() for _ in range(places)]
    new_hits = [0] * places
    onchart = [0] * len(starts)
    nfine = int(round((FINE_HI - FINE_LO) / FINE))
    fine = [[0] * nfine for _ in range(places)]
    filled = [0] * places
    nlargest = heapq.nlargest
    nr = len(reads_h)
    for s in range(n_sims):
        r = s % nr
        vals = []
        app = vals.append
        for i, per in enumerate(params):
            loc, sc, pd = per[r]
            v = loc + sc * tp[int(rnd() * tn)]
            if rnd() < pd:
                v -= 1.0 + rnd()
            if v >= cfl:
                onchart[i] += 1
            app((v, i))
        for frac, cands in plans[r]:
            if frac < 1.0 and rnd() >= frac:
                continue
            cand = cands[int(rnd() * len(cands))]
            pd = cand[-1]
            for j in range(len(cand) - 1):
                loc, sc = cand[j]
                v = loc + sc * tp[int(rnd() * tn)]
                if pd and rnd() < pd:
                    v -= 1.0 + rnd()
                app((v, -1))
        for k, (v, i) in enumerate(nlargest(places, vals)):
            if i < 0:
                new_hits[k] += 1
            else:
                hk = hits[k]
                hk[i] = hk.get(i, 0) + 1
            cell = int((v - FINE_LO) / FINE)
            fine[k][0 if cell < 0 else nfine - 1 if cell >= nfine else cell] += 1
            filled[k] += 1
    info.update(mu=round(dyn.mu, 4), beta=round(dyn.beta, 4), gamma=round(dyn.gamma, 4), jump=round(dyn.jump, 4),
                sigma=round(dyn.sigma, 4), H=round(dyn.H, 4),
                hazard=round(dyn.hazard, 4), sims=n_sims, existing=len(starts), width_h=round(width, 4),
                chart_floor=None if cfloor is None else round(cfloor, 3))
    return dict(mints=mints, starts=starts, hits=hits, new_hits=new_hits, fine=fine, filled=filled,
                onchart=onchart, n_sims=n_sims, places=places, chart_size=chart_size, info=info)


# -------------------------------------------------------------------------------------------------- output

def floor6(p):
    return math.floor(p * MICRO) / MICRO


def expected_score(P, C, p, N, chart_size):
    """The forecast's own expected ln-score at one place for naming coins with beliefs P, chart chances C and
    reported p; N is the belief that an unnamed coin (newcomer or unnamed existing) takes the place."""
    m = max(1.0, chart_size - sum(C))
    u = max(SCORE_FLOOR, (1.0 - sum(p)) / m)
    e = N * math.log(u)
    for Pi, pi in zip(P, p):
        e += Pi * math.log(max(SCORE_FLOOR, pi))
    return e


def order_rows(sim):
    places = sim["places"]
    mints = sim["mints"]
    chart_size = sim["chart_size"]
    n = max(1, sim["n_sims"])
    nc = len(mints)
    prior = [1.0 / (i + 3.0) for i in range(nc)]          # broad prior over existing coins, by rank at the cut
    zp = sum(prior) or 1.0
    prior = [p / zp for p in prior]
    chart_p = [CHART_TRUST * min(1.0, sim["onchart"][i] / n) for i in range(nc)]
    rows = []
    for k in range(places):
        hk = sim["hits"][k]
        mc_new = sim["new_hits"][k] / n
        belief = [(1.0 - EPS_ORDER) * hk.get(i, 0) / n + EPS_ORDER * (1.0 - mc_new) * prior[i] for i in range(nc)]
        p_new = 1.0 - sum(belief)
        # rank candidates by an estimate of their marginal value, then take the best-scoring prefix
        u0 = max(SCORE_FLOOR, p_new / max(1.0, chart_size / 2.0))
        m0 = max(1.0, chart_size / 2.0)

        def gain(i):
            P = belief[i]
            return P * math.log(max(P, SCORE_FLOOR) / u0) + chart_p[i] * p_new / m0

        order = sorted(range(nc), key=lambda i: (-gain(i), mints[i]))[:MAX_ROWS]
        best_len, best_e, best_p = 0, expected_score([], [], [], 1.0, chart_size), []
        P, C = [], []
        for L, i in enumerate(order, 1):
            P.append(belief[i])
            C.append(chart_p[i])
            rep = [max(P_MIN, x) for x in P]
            tot = sum(rep)
            if tot > 1.0 - MIN_UNNAMED:
                rep = [x * (1.0 - MIN_UNNAMED) / tot for x in rep]
            e = expected_score(P, C, rep, 1.0 - sum(P), chart_size)
            if e > best_e + 1e-12:
                best_len, best_e, best_p = L, e, rep
        named = [(order[j], best_p[j]) for j in range(best_len)]
        named.sort(key=lambda t: (-t[1], mints[t[0]]))
        for i, p in named:
            p = floor6(p)
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
    fill = [m for _, m in sim["starts"]] + sorted(coins, key=lambda m: (-coins[m].peak, m)) + PLACEHOLDERS
    for m in fill:
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
    bounds = list(edges) + [math.inf]
    kern = []         # fine cell centre -> its kernel mass in each bin
    for cell in range(nfine):
        x = FINE_LO + (cell + 0.5) * FINE
        row, prev = [], 0.0
        for b in range(nb):
            c = 1.0 if bounds[b] == math.inf else normal_cdf((bounds[b] - x) / SIZE_KERNEL_SD)
            row.append(c - prev)
            prev = c
        kern.append(row)
    rows = []
    for k in range(places):
        tot = sim["filled"][k]
        if tot > 0:
            dens = [0.0] * nb
            for cell, cnt in enumerate(sim["fine"][k]):
                if cnt:
                    kr = kern[cell]
                    for b in range(nb):
                        dens[b] += cnt * kr[b]
            dens = [(1.0 - SIZE_UNIFORM) * d / tot + SIZE_UNIFORM / nb for d in dens]
        else:
            dens = [1.0 / nb] * nb
        for b in range(nb):
            p = floor6(dens[b])
            if p > 0:
                rows.append({"place": k + 1, "bin": b, "p": p})
    return rows


def forecast(task, marks, seed=SEED):
    rng = random.Random(seed)
    _, times, charts, floors, coins = build(task, marks)
    edges = (task.get("size") or {}).get("edges")
    edges = [float(e) for e in edges] if isinstance(edges, list) and edges else [2.0 + 0.05 * i for i in range(81)]
    sim = simulate(task, times, charts, floors, coins, rng)
    rows = order_rows(sim)
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
