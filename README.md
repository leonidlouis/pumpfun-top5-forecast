# pumpfun-top5-forecast-01 entry: a self-calibrating Monte Carlo

A make.fun entry for **pumpfun-top5-forecast-01**: forecast the 5 largest new pump.fun coins (and the size of each
place's market cap) at seven answer reads after the history ends.

- **Python 3.13, standard library only.** One entrypoint, `main.py`, at the repository root.
- **Model-free and offline:** no network, no API keys, no `.env`, no LLM or model calls. Every parameter is fitted
  on the input history itself, shrunk toward fixed priors when the history is short.
- **Deterministic:** `random.Random` with a fixed seed, stable sorts with mint tie-breaks, no wall clock, no
  set-order iteration in the output. Two runs on the same input are byte-identical.

## How to run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=/path/to/kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
# PF_DEBUG=1 additionally prints the fitted parameters to stdout
```

It reads `$MAKEFUN_INPUT_DIR` (default `/input`): `task.json` and `history.ndjson`. It writes
`$MAKEFUN_OUTPUT_DIR` (default `/output`)/`<task.output>` with exactly `probabilities`, `top5` and `sizes`. All
times, reads, places, bin edges and the output name come from `task.json`; nothing is hard-coded.

Tests and tools:

```sh
python3 -m unittest discover -s tests -v                  # schema, limits, determinism, robustness, scorer
python3 tools/synth.py /tmp/syn --seed 1                    # synthetic 66-mark / 11 h history + answers
python3 tools/score.py /tmp/syn/out/forecast.json /tmp/syn/answers.json
python3 tools/bench.py --seeds 1 2 3 4 5 6 7 8              # main.py vs the baseline port on synthetic markets
```

## Method

1. **Parse.** Each mark gives a ranked chart of up to 200 coins. Every coin seen gets a trajectory of
   (hours since t0, log10 cap in SOL). A coin missing from a later full chart is below that mark's 200th cap.
2. **Fit the dynamics** on (coin, lag) pairs: the earlier mark's top 50, coins aged at least 1 h, lags of
   1–6 h (all lags when the history is under an hour). Later marks get more weight.
   - Change over a lag, robust (Huber) regression:
     `d = jump·min(1, lag) + lag·(mu + beta·(cap − 3.5) + gamma·ln(age / 3 h))`.
     The one-off `jump` absorbs the fall-back of coins that were selected for having just pumped, so it is
     not extrapolated as a per-hour trend.
   - Scale: a robust (MAD, t(3) units) residual scale per lag, then `scale(h) = sigma·h^H`. `H < 0.5`
     means changes grow slower than a random walk's.
   - Death hazard: per-hour rate of falling at least 1 log10. A dropout counts only where the later floor
     sits at least 1 log10 below the coin, and exposure is counted the same way.
   - Every estimate is shrunk toward its prior by the number of distinct coins (prior weight 25) and
     clamped. Priors: mu = −0.04/h, sigma = 0.22, H = 0.5, hazard = 0.03/h, beta = gamma = jump = 0.
3. **Existing coins** (only these can be named). Each simulation draws a coin's cap at a read from its last
   value at the cut:
   - Drift is the integral above, taken as the coin ages over the horizon.
   - Noise is Student-t(3), clipped at ±4 scale units.
   - The scale grows for young coins (age factor) and for curve or small coins.
   - With the death probability, the coin drops 1–2 log10.

   Coins that fell off the chart before the last mark start just below its floor, if they once ranked in the
   top 20.
4. **Newcomers** (born after the last mark, and mayhem coins never ranked) come from a cohort bootstrap:
   - History coins are grouped into launch cohorts (1 h; 10 min for histories under 3 h).
   - Each future launch slot between the last mark and the read draws one history cohort observed at the age
     the newcomers will have at that read, and uses its top caps there.
   - Ages older than anything observed are extrapolated from a cohort's oldest observation with the fitted
     dynamics.
5. **Monte Carlo.** About 16,000–21,000 simulations, cycling over the 7 reads. Each one ranks existing and
   newcomer caps and records who holds places 1–5, each place's log cap (0.01 cells), and which existing coins
   clear the answer chart's floor (the 100th cap of the recent marks).
6. **Order probabilities.**
   - Each coin's belief is 0.88 × its MC hit share plus 0.12 × a broad rank prior.
   - Candidates are ranked by their marginal value, and the named set is the prefix of at most 100 rows that
     maximises the forecast's own expected log score.
   - The unnamed share is spread over the answer chart's unnamed coins. Naming a coin that is likely on the
     chart but unlikely at the place, at a small p (≥ 3e-4), raises the share newcomers get. The chart-presence
     chance is discounted (×0.35), because the synthetic runs showed it overstated.
   - Every place leaves at least 8% unnamed.
   - `top5` is chosen greedily: the most probable unused coin at each place.
7. **Sizes.** Each place's simulated log caps are smoothed with a Gaussian kernel (sd 0.15 log10), mixed with
   3% uniform over the 82 bins, and floored to 6 decimals, so every place sums to at most 1.
8. **Guards.**
   - `sanitize` re-enforces every limit before writing: known mints only, ≤ 100 rows per place, sums ≤ 1,
     bins 0–81, 5 distinct `top5` mints.
   - Any exception falls back to a persistence forecast, so a file is always written.
   - Empty marks, a 1-mark history, fewer than 5 coins, null fields and mayhem rows are all handled.

## Measured numbers

Runtime, 1 CPU, measured here:

| input | runtime |
|---|---|
| kit practice input (6 marks) | ~2.0 s |
| synthetic final-shaped input (66 marks over 11 h, 200 rows each) | ~2.3–3.9 s |

The limit is 300 s. Output is about 25 KB, against a 1 MiB limit.

**Synthetic scores.** These are not real data. `tools/synth.py` is deliberately a different process from the
model's: runners peak and then decay, with AR noise, rugs and a graduation cap. Each seed is scored on 7 answer
reads, 7–13 h after the cut, against a Python port of the house baseline (`tools/baseline.py`).

| seeds | role | main.py log (order / size) | baseline log (order / size) |
|---|---|---|---|
| 1–8 | used while developing | −41.12 (−22.73 / −18.39) | −44.94 (−24.57 / −20.37) |
| 9–16 | looked at once during development | −42.22 (−23.36 / −18.86) | −44.01 (−24.67 / −19.34) |
| 17–24 | fresh, never used for any decision | −41.53 (−23.00 / −18.53) | −42.92 (−25.47 / −17.45) |

The even spread scores −45.059447.

Honest reading:

- On these synthetic markets main.py beats persistence on average, mostly on order. Its order score is close
  to the even spread, because the synthetic winners are hard to predict and newcomers take 50–90% of places.
- On seeds 17–24 the baseline's sizes were better than main.py's.
- Calibration choices were not tuned hard to the synthetic generator. Where the more defensible estimator
  scored a little worse on synthetic data (the jump term and the 1-log death hazard), it was kept.
- On the kit's practice history (6 marks, 50 min), every pair is younger than 1 h, so the dynamics fall back
  to the priors. The practice answer isn't in the kit, so no practice score could be computed locally.

## Files

- `main.py`: the entry
- `tools/synth.py`: synthetic market generator
- `tools/score.py`: the RULES log score
- `tools/baseline.py`: the house baseline, ported to Python
- `tools/bench.py`: the comparison harness
- `tests/test_main.py`: unittest suite

Licensed MIT (see `LICENSE`).
