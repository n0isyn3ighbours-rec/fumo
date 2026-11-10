"""Self-checks for the scoring maths and one full offline run. No network, no torch.  Run: python test_forecast.py"""
import math
import numpy as np
import pandas as pd
import forecast as f


def test_crps():
    dec = [100.0] * 9
    assert f.crps(dec, 100.0) == 0                      # perfect point forecast
    assert f.crps([90 + 2.5 * i for i in range(9)], 100) < f.crps([60 + 10 * i for i in range(9)], 100)  # tighter, centred band wins
    assert f.crps(dec, 110.0) > f.crps(dec, 105.0)      # further miss costs more


def test_wilson():
    lo, hi = f.wilson(50, 100)
    assert lo < 0.5 < hi and math.isclose(lo + hi, 1.0, abs_tol=1e-3)
    assert f.wilson(0, 0) == [0.0, 1.0]


def test_ewma_band_scales_with_vol():
    calm = 100 * np.exp(np.cumsum(np.full(200, 0.001) * np.tile([1, -1], 100)))
    wild = 100 * np.exp(np.cumsum(np.full(200, 0.02) * np.tile([1, -1], 100)))
    w = lambda d: d[-1] - d[0]
    assert w(f.ewma_deciles(wild, 100)) > 10 * w(f.ewma_deciles(calm, 100))


def test_scoring_and_alert():
    df = f.synthetic_candles(f.CONTEXT + 30)
    hist = [{"t": 1, "target_t": int(df.t.iloc[-1]), "last": 1.0, "kronos": [0.5] * 4 + [2.0] + [1e9] * 4,
             "ewma": [1e9] * 9, "band1": [0, 0]}]
    hist = f.score_history(hist, df)
    h = hist[0]
    assert h["hit"] is True and h["hit_ewma"] is False and h["dir_ok"] is True
    # last hour's band was [0, 0], so the new close is outside it -> alert
    hist.append({"t": int(df.t.iloc[-2]), "target_t": 0, "last": 1.0, "kronos": [1] * 9, "ewma": [1] * 9, "band1": [0, 0]})
    meta = {"name": "test", "params": "-", "context": f.CONTEXT, "samples": 8, "top_p": 1.0}
    out, hist, paper, alert = f.run(df.tail(f.CONTEXT).reset_index(drop=True), f.random_walk_paths(df, 8), hist, meta, "test")
    assert alert and "outside" in alert
    assert out["schema"] == "fumo.kronos.v1" and len(out["forecast"]["p50"]) == f.HOURS
    assert all(a <= b <= c for a, b, c in zip(out["forecast"]["p10"], out["forecast"]["p50"], out["forecast"]["p90"]))
    assert out["lean"]["up"] + out["lean"]["down"] == 8


def test_fake_replay_scores():
    df = f.synthetic_candles(f.CONTEXT + 60)
    meta = {"name": "rw", "params": "-", "context": f.CONTEXT, "samples": 50, "top_p": 1.0}
    hist, paper = [], None
    for i in range(60, -1, -1):
        d = df.iloc[: len(df) - i].tail(f.CONTEXT).reset_index(drop=True)
        out, hist, paper, _ = f.run(d, f.random_walk_paths(d, 50), hist, meta, "synthetic", paper if i < 60 else None)
    s = out["score"]
    assert s["n"] == 60 - f.HOURS + 1 and s["independent"] == 1 and s["too_few"]
    assert 0 <= s["coverage"] <= 1
    lo, hi = s["direction_ci"]
    assert lo < 0.5 < hi  # one independent window can never look significant


def _q(last, med_ret):
    return np.array([[last * 0.98] * f.HOURS, [last * (1 + med_ret)] * f.HOURS, [last * 1.02] * f.HOURS])


def _bar(df, close, lo=None, hi=None):
    t = int(df.t.iloc[-1]) + 3600
    row = {"t": t, "open": close, "high": hi or close, "low": lo or close, "close": close, "volume": 1, "amount": close}
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def step(state, df, q, up, n):
    """Drive paper_step the way run() does for Kronos: median move + share of paths agreeing."""
    med = float(q[1][-1]) / float(df.close.iloc[-1]) - 1
    return f.paper_step(state, df, med, (up if med > 0 else n - up) / n)


def test_paper_stop_and_exit():
    df = f.synthetic_candles(f.CONTEXT)
    last = float(df.close.iloc[-1])
    s = step(None, df, _q(last, 0.02), up=50, n=64)  # +2% median, 78% agree -> LONG
    p = s["pos"]
    assert p["side"] == "LONG" and math.isclose(p["notional"], 1000 * 0.005 / 0.02, rel_tol=1e-9)
    assert step(s, df, _q(last, 0.02), 50, 64) is s and len(s["equity"]) == 1  # same hour: no double run
    df2 = _bar(df, last * 0.97, lo=last * 0.96)  # falls through the 2% stop
    s = step(s, df2, _q(last * 0.97, 0), up=32, n=64)
    t = s["trades"][-1]
    assert t["why"] == "stop" and t["pnl"] < 0 and t["fees"] > 0
    assert abs(t["pnl"] + 0.005 * 1000) < 1.5  # lost ~0.5% of equity plus costs
    assert s["skips"]["median move under 0.5%"] == 1

    df3 = df2
    s = step(s, df3 := _bar(df3, last), _q(last, -0.02), up=10, n=64)  # SHORT
    assert s["pos"]["side"] == "SHORT"
    for _ in range(f.HOURS):
        df3 = _bar(df3, last * 0.99)
        s = step(s, df3, _q(last, 0), 32, 64)
    t = s["trades"][-1]
    assert t["side"] == "SHORT" and t["why"] == "24h" and t["pnl"] > 0
    st = f.paper_stats(s)
    assert st["trades"] == 2 and st["wins"] == 1 and st["max_dd_pct"] > 0 and st["too_few"]


def test_paper_skips_split_paths():
    df = f.synthetic_candles(f.CONTEXT)
    s = step(None, df, _q(float(df.close.iloc[-1]), 0.02), up=34, n=64)  # 53% agree
    assert s["pos"] is None and s["skips"] == {"paths split": 1}


def _scored(n, real_offset):
    """n scored forecasts with a band of +-100 around 1000 and the real price landing at 1000 + offset(i)."""
    return [{"kronos": [900, 0, 0, 0, 1000, 0, 0, 0, 1100], "real": 1000 + real_offset(i)} for i in range(n)]


def test_calibration_learns_the_stretch():
    assert f.calib_factor(_scored(10, lambda i: 0))[0] == 1.0          # too few: no change
    # real lands uniformly out to 2x the band: 80% coverage needs ~1.6x
    k, n = f.calib_factor(_scored(200, lambda i: (i % 100) * 2 * (1 if i % 2 else -1)))
    assert n == 200 and 1.5 < k < 1.75
    assert f.calib_factor(_scored(200, lambda i: (i % 100) * 0.5))[0] < 0.6  # overcovering band gets narrower
    assert f.stretch([900, 1000, 1100], 1000, 2) == [800, 1000, 1200]


def test_momentum_signal_and_benchmark():
    df = f.synthetic_candles(f.CONTEXT)
    df.loc[df.index[-25:], "close"] = np.linspace(80_000, 84_000, 25)  # a clean 24h rally
    move, agree = f.momentum_signal(df)
    assert move > 0.04 and agree == 1.0
    s = f.paper_step(None, df, move, agree, words=("24h", "hourly moves"))
    assert s["pos"]["side"] == "LONG" and "24h +5.00%" in s["log"][-1][2]


def test_run_tracks_both_accounts_and_calibrated_scores():
    df = f.synthetic_candles(f.CONTEXT + 120)
    meta = {"name": "rw", "params": "-", "context": f.CONTEXT, "samples": 40, "top_p": 1.0}
    hist, paper = [], None
    for i in range(120, -1, -1):
        d = df.iloc[: len(df) - i].tail(f.CONTEXT).reset_index(drop=True)
        out, hist, paper, _ = f.run(d, f.random_walk_paths(d, 40), hist, meta, "synthetic", paper)
    assert set(paper) == {"kronos", "momentum"} and out["benchmark"]["name"] == "24h momentum"
    assert out["calibration"]["active"] and out["score"]["coverage_cal"] is not None
    fc = out["forecast"]
    assert all(a <= m <= b for a, m, b in zip(fc["p10_cal"], fc["p50"], fc["p90_cal"]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
