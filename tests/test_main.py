"""Tests for main.py and tools/score.py (stdlib unittest).

    python3 -m unittest discover -s tests -v
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_INPUT = "/workspace/make-fun/kit-pumpfun/pumpfun-top5-forecast-01-kit/input"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import main  # noqa: E402
import score  # noqa: E402
import synth  # noqa: E402

B58 = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


def make_task(t0="2030-01-01T00:00:00.000Z", until="2030-01-01T01:00:00.000Z", reads=None):
    edges = [round(2 + 0.05 * i, 2) for i in range(81)]
    return {"t0": t0, "history_until": until,
            "reads": reads if reads is not None else ["2030-01-01T01:00:00.000Z", "2030-01-01T01:30:00.000Z"],
            "places": 5, "chart_size": 100, "size": {"edges": edges}, "output": "forecast.json"}


def row(rank, mint, sol, created="2030-01-01T00:01:00.000Z", venue="curve", mayhem=False):
    return {"rank": rank, "mint": mint, "market_cap_lamports": str(int(sol * 1e9)), "venue": venue, "slot": 1,
            "name": None, "symbol": None, "created_at": created, "complete": venue == "pool", "mayhem": mayhem,
            "quote_reserves_lamports": None}


def mint_of(i):
    digits = "".join("123456789A"[int(c)] for c in "%03d" % i)     # base58 has no 0
    return ("M" + digits).ljust(40, "x") + "pump"


def run_main(input_dir, output_dir):
    env = dict(os.environ, MAKEFUN_INPUT_DIR=input_dir, MAKEFUN_OUTPUT_DIR=output_dir)
    t = time.perf_counter()
    subprocess.run([sys.executable, os.path.join(ROOT, "main.py")], env=env, check=True)
    secs = time.perf_counter() - t
    with open(os.path.join(output_dir, "forecast.json"), "rb") as f:
        return f.read(), secs


def write_input(d, task, marks):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "task.json"), "w") as f:
        json.dump(task, f)
    with open(os.path.join(d, "history.ndjson"), "w") as f:
        for m in marks:
            f.write(json.dumps(m) + "\n")


class Validate:
    def check(self, out, history_mints=None, places=5):
        self.assertEqual(set(out), {"probabilities", "top5", "sizes"})
        per = {}
        for r in out["probabilities"]:
            self.assertEqual(set(r), {"mint", "place", "p"})
            self.assertIsInstance(r["mint"], str)
            self.assertTrue(32 <= len(r["mint"]) <= 44)
            self.assertIsInstance(r["place"], int)
            self.assertTrue(1 <= r["place"] <= places)
            self.assertTrue(0 <= r["p"] <= 1)
            if history_mints is not None:
                self.assertIn(r["mint"], history_mints)
            per.setdefault(r["place"], []).append(r)
        for k, rows in per.items():
            self.assertLessEqual(len(rows), 100)
            self.assertEqual(len({r["mint"] for r in rows}), len(rows))
            self.assertLessEqual(sum(r["p"] for r in rows), 1.0 + 1e-9)
        self.assertEqual(len(out["top5"]), 5)
        self.assertEqual(len(set(out["top5"])), 5)
        for m in out["top5"]:
            self.assertTrue(32 <= len(m) <= 44 and set(m) <= B58 | set("x"))
        if history_mints is not None and len(history_mints) >= 5:
            for m in out["top5"]:
                self.assertIn(m, history_mints)
        sz = {}
        for r in out["sizes"]:
            self.assertEqual(set(r), {"place", "bin", "p"})
            self.assertTrue(0 <= r["bin"] <= 81)
            self.assertTrue(1 <= r["place"] <= places)
            self.assertTrue(0 <= r["p"] <= 1)
            sz.setdefault(r["place"], {})
            self.assertNotIn(r["bin"], sz[r["place"]])
            sz[r["place"]][r["bin"]] = r["p"]
        for k, bins in sz.items():
            self.assertLessEqual(sum(bins.values()), 1.0 + 1e-9)
            self.assertGreater(sum(bins.values()), 0.99)


@unittest.skipUnless(os.path.isdir(KIT_INPUT), "kit input not present")
class TestKit(unittest.TestCase, Validate):
    def test_schema_and_determinism(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            out1, secs = run_main(KIT_INPUT, a)
            out2, _ = run_main(KIT_INPUT, b)
            self.assertEqual(out1, out2)
            self.assertLess(len(out1), 1 << 20)
            self.assertLess(secs, 60)
        hm = set()
        with open(os.path.join(KIT_INPUT, "history.ndjson")) as f:
            for line in f:
                if line.strip():
                    hm.update(r["mint"] for r in json.loads(line)["rows"])
        self.check(json.loads(out1), hm)


class TestRobustness(unittest.TestCase, Validate):
    def forecast(self, task, marks):
        with tempfile.TemporaryDirectory() as d:
            write_input(os.path.join(d, "in"), task, marks)
            raw, _ = run_main(os.path.join(d, "in"), os.path.join(d, "out"))
            return json.loads(raw)

    def test_one_mark(self):
        rows = [row(i + 1, mint_of(i), 1000 / (i + 1)) for i in range(30)]
        out = self.forecast(make_task(), [{"at": "2030-01-01T00:50:00.000Z", "ok": True, "rows": rows}])
        self.check(out, {r["mint"] for r in rows})

    def test_only_empty_marks(self):
        marks = [{"at": "2030-01-01T00:%02d:00.000Z" % (10 * i), "ok": True, "rows": []} for i in range(3)]
        out = self.forecast(make_task(), marks)
        self.check(out, set())
        self.assertEqual(out["probabilities"], [])

    def test_empty_first_mark_then_rows(self):
        marks = [{"at": "2030-01-01T00:00:04.000Z", "ok": True, "rows": []},
                 {"at": "2030-01-01T00:10:04.000Z", "ok": True,
                  "rows": [row(i + 1, mint_of(i), 300 / (i + 1)) for i in range(12)]}]
        out = self.forecast(make_task(), marks)
        self.check(out, {mint_of(i) for i in range(12)})

    def test_fewer_than_five_coins_and_nulls(self):
        rows = [row(1, mint_of(1), 500), row(2, mint_of(2), 200, venue="pool", mayhem=True),
                {"rank": 3, "mint": mint_of(3), "market_cap_lamports": None, "venue": None, "created_at": None},
                {"rank": 4, "mint": None, "market_cap_lamports": "123"}]
        out = self.forecast(make_task(), [{"at": "2030-01-01T00:50:00.000Z", "ok": True, "rows": rows}])
        self.check(out)
        hist = {mint_of(1), mint_of(2), mint_of(3)}
        for r in out["probabilities"]:
            self.assertIn(r["mint"], hist)
        self.assertTrue({mint_of(1), mint_of(2)} <= set(out["top5"]))

    def test_no_reads_in_task(self):
        rows = [row(i + 1, mint_of(i), 900 / (i + 1)) for i in range(8)]
        out = self.forecast(make_task(reads=[]), [{"at": "2030-01-01T00:50:00.000Z", "ok": True, "rows": rows}])
        self.check(out, {r["mint"] for r in rows})


class TestSyntheticFinalShape(unittest.TestCase, Validate):
    def test_synthetic_runs_fast_and_valid(self):
        with tempfile.TemporaryDirectory() as d:
            synth.generate(d, seed=3)
            raw, secs = run_main(os.path.join(d, "input"), os.path.join(d, "out"))
            with open(os.path.join(d, "answers.json")) as f:
                ans = json.load(f)
            out = json.loads(raw)
            self.check(out, set(ans["history_mints"]))
            self.assertLess(secs, 60)
            s = score.score(out, ans["reads"], ans["history_mints"])
            self.assertTrue(-92.103404 <= s["log_score"] <= 0)


class TestScorer(unittest.TestCase):
    def chart(self, top):
        fill = [{"mint": "X%043d" % i, "market_cap_lamports": str(10 ** 11)} for i in range(100 - len(top))]
        return {"chart": [{"mint": m, "market_cap_lamports": str(int(c * 1e9))} for m, c in top] + fill}

    def test_even_spread(self):
        s = score.score({"probabilities": [], "top5": [], "sizes": []}, [self.chart([("A" * 44, 1800)])])
        self.assertEqual(s["log_score"], -45.059447)
        self.assertAlmostEqual(5 * math.log(1 / 100) + 5 * math.log(1 / 82), -45.059447, places=6)

    def test_rules_worked_example(self):
        A, B, C, F, G = ("A" * 44, "B" * 44, "C" * 44, "F" * 44, "G" * 44)
        fc = {"probabilities": [{"mint": A, "place": 1, "p": 0.5}, {"mint": B, "place": 1, "p": 0.2},
                                {"mint": B, "place": 2, "p": 0.4}],
              "top5": [A, B, C, "D" * 44, "E" * 44],
              "sizes": [{"place": 1, "bin": 26, "p": 0.3}, {"place": 1, "bin": 27, "p": 0.3}]}
        r1 = self.chart([(A, 1800), (C, 1500), (B, 1400), (F, 1300), (G, 1200)])
        r2 = self.chart([(A, 2100), (B, 1500), (C, 1400), (F, 1300), (G, 1200)])
        s1 = score.score(fc, [r1])
        self.assertEqual((s1["log_score"], s1["order"], s1["size"], s1["agreement"]),
                         (-38.445453, -19.614603, -18.83085, 13))
        s = score.score(fc, [r1, r2])
        self.assertEqual((s["log_score"], s["order"], s["agreement"]), (-36.350626, -17.519776, 28))

    def test_unknown_mint_dropped(self):
        known = "K" * 44
        fc = {"probabilities": [{"mint": "U" * 44, "place": 1, "p": 0.9}], "top5": [], "sizes": []}
        s_drop = score.score(fc, [self.chart([(known, 1800)])], history_mints=[known])
        s_none = score.score({"probabilities": [], "top5": [], "sizes": []}, [self.chart([(known, 1800)])])
        self.assertEqual(s_drop["log_score"], s_none["log_score"])


class TestPieces(unittest.TestCase):
    def test_bins(self):
        edges = [round(2 + 0.05 * i, 2) for i in range(81)]
        self.assertEqual(score.size_bin(str(99 * 10 ** 9), edges), 0)
        self.assertEqual(score.size_bin(str(1000 * 10 ** 9), edges), 21)
        self.assertEqual(score.size_bin(str(1800 * 10 ** 9), edges), 26)
        self.assertEqual(score.size_bin(str(10 ** 15), edges), 81)

    def test_drift_integral(self):
        d = main.Dyn(mu=-0.1, beta=0.0, gamma=0.0, jump=0.0)
        self.assertAlmostEqual(d.drift(5.0, 3.0, 2.0), -0.5)
        g = main.Dyn(mu=0.0, gamma=0.1)
        # integral of 0.1 ln(age / 3) from age 3 to 3 + h is positive and matches a Riemann sum
        h, a = 4.0, 3.0
        riemann = sum(0.1 * math.log((a + (i + 0.5) * h / 4000) / 3.0) * h / 4000 for i in range(4000))
        self.assertAlmostEqual(g.drift(h, 3.0, a), riemann, places=5)


if __name__ == "__main__":
    unittest.main()
