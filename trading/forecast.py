"""Hourly Kronos forecasts for FUMO (BTC, ETH, SOL): fetch candles, forecast 24h, score past forecasts, paper-trade,
and write data/<SYMBOL>/forecast.json for each coin.

Paper-only research tool. Nothing here trades or holds keys.
Run:  python forecast.py                      (live: Binance/Kraken data + Kronos, all coins)
      python forecast.py --symbols BTCUSDT      (just one coin)
      python forecast.py --fake 72              (no network, no torch: synthetic candles + random-walk paths, for UI work)
"""
import argparse, functools, json, math, os, sys, time, urllib.request
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd

HOURS, CONTEXT, KEEP_CANDLES, KEEP_PATHS = 24, 512, 120, 30
SYMBOLS = {"BTCUSDT": "XBTUSDT", "ETHUSDT": "ETHUSDT", "SOLUSDT": "SOLUSDT"}  # Binance symbol -> Kraken fallback pair
FAKE_START = {"BTCUSDT": 80_000.0, "ETHUSDT": 3_600.0, "SOLUSDT": 180.0}
DECILES = np.arange(1, 10) / 10  # p10..p90, used for scoring (CRPS) and the band
MODELS = {  # (model repo, model revision, tokenizer repo, tokenizer revision, params)
    "mini": ("NeoQuasar/Kronos-mini", None, "NeoQuasar/Kronos-Tokenizer-2k", None, "4.1M"),  # ponytail: unpinned; pin the commit hashes after the first download
    "small": ("NeoQuasar/Kronos-small", "901c26c1332695a2a8f243eb2f37243a37bea320",
              "NeoQuasar/Kronos-Tokenizer-base", "0e0117387f39004a9016484a186a908917e22426", "24.7M"),
}
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------- data ----------

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fumo-forecast"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def fetch_fx(previous=None):
    """USD->ZAR for display only (ECB daily rate). Falls back to the last known rate; never blocks a forecast."""
    try:
        r = _get("https://api.frankfurter.dev/v1/latest?from=USD&to=ZAR")
        return {"ZAR": float(r["rates"]["ZAR"]), "date": r["date"], "source": "ECB via frankfurter.dev"}
    except Exception as e:
        print(f"fx failed ({e}); keeping the last rate", file=sys.stderr)
        return previous


def fetch_candles(symbol="BTCUSDT", now=None):
    """Last closed 1h candles, oldest first. Binance market-data host first, Kraken as fallback."""
    now = now or time.time()
    try:
        rows = _get(f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit={CONTEXT + 1}")
        df = pd.DataFrame([[r[0] // 1000, *map(float, r[1:6]), float(r[7]), r[6] // 1000] for r in rows],
                          columns=["t", "open", "high", "low", "close", "volume", "amount", "close_t"])
        source = "binance"
    except Exception as e:
        print(f"binance failed ({e}); using kraken", file=sys.stderr)
        res = _get(f"https://api.kraken.com/0/public/OHLC?pair={SYMBOLS[symbol]}&interval=60")["result"]
        rows = next(v for k, v in res.items() if k != "last")
        # kraken: time, open, high, low, close, vwap, volume, count
        df = pd.DataFrame([[int(r[0]), *map(float, r[1:5]), float(r[6]), float(r[5]) * float(r[6]), int(r[0]) + 3599] for r in rows],
                          columns=["t", "open", "high", "low", "close", "volume", "amount", "close_t"])
        source = "kraken"
    df = df[df.close_t < now].drop(columns="close_t")  # drop the still-open candle
    return df.tail(CONTEXT).reset_index(drop=True), source


def synthetic_candles(n, seed=4, start_price=80_000.0):
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    open_ = np.r_[start_price, close[:-1]]
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    t0 = (int(time.time()) // 3600 - n) * 3600
    vol = rng.uniform(300, 900, n)
    return pd.DataFrame({"t": t0 + 3600 * np.arange(n), "open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close, "volume": vol, "amount": vol * close})


# ---------- paths ----------

def kronos_paths(df, n_paths, size="mini"):
    """n_paths sampled 24h close paths, shape (n_paths, HOURS)."""
    import torch
    torch.manual_seed(int(df.t.iloc[-1]))
    predictor = _predictor(size)
    x = df[["open", "high", "low", "close", "volume", "amount"]]
    xt = pd.Series(pd.to_datetime(df.t, unit="s"))
    yt = pd.Series(pd.to_datetime(df.t.iloc[-1] + 3600 * np.arange(1, HOURS + 1), unit="s"))
    # predict() averages its samples (kronos.py:467), so ask for 1 sample per row and repeat the row instead.
    with torch.no_grad():
        out = predictor.predict_batch([x] * n_paths, [xt] * n_paths, [yt] * n_paths, pred_len=HOURS,
                                      T=1.0, top_k=0, top_p=1.0, sample_count=1, verbose=False)
    return np.stack([o["close"].to_numpy() for o in out])


@functools.lru_cache(maxsize=None)
def _predictor(size):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the vendored Kronos model/ folder
    from model import Kronos, KronosTokenizer, KronosPredictor
    repo, rev, tok_repo, tok_rev, _ = MODELS[size]
    tokenizer = KronosTokenizer.from_pretrained(tok_repo, revision=tok_rev).eval()
    model = Kronos.from_pretrained(repo, revision=rev).eval()  # Kronos never calls eval(): dropout stays on otherwise (issue #415)
    return KronosPredictor(model, tokenizer, device="cpu", max_context=CONTEXT)


def random_walk_paths(df, n_paths, seed=0):
    rng = np.random.default_rng(seed + int(df.t.iloc[-1]))
    sd = np.log(df.close).diff().std()
    return df.close.iloc[-1] * np.exp(np.cumsum(rng.normal(0, sd, (n_paths, HOURS)), axis=1))


# ---------- scoring ----------

def ewma_deciles(closes, last, lam=0.94):
    """Baseline: a normal band from EWMA hourly volatility, scaled to 24h."""
    r = np.diff(np.log(closes))
    var = r[0] ** 2
    for x in r[1:]:
        var = lam * var + (1 - lam) * x * x
    sd = math.sqrt(var * HOURS)
    return [last * math.exp(NormalDist().inv_cdf(q) * sd) for q in DECILES]


def crps(deciles, real):
    """CRPS approximated from p10..p90 (average pinball loss x2), in % of price. Lower is better."""
    loss = [max(q * (real - d), (q - 1) * (real - d)) for q, d in zip(DECILES, deciles)]
    return 200 * float(np.mean(loss)) / real


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


CAL_WINDOW, CAL_MIN = 24 * 30, 48  # learn the band scale from the last 30 days of scored forecasts, once 48 exist


def calib_factor(history):
    """Split-conformal scale k: the smallest stretch of the p10-p90 band (about the median) that would have
    covered 80% of recent scored forecasts. Uses only outcomes already known, so it never sees the future."""
    s = [h for h in history if "real" in h][-CAL_WINDOW:]
    if len(s) < CAL_MIN:
        return 1.0, len(s)
    need = []
    for h in s:
        lo, med, hi, r = h["kronos"][0], h["kronos"][4], h["kronos"][-1], h["real"]
        need.append((r - med) / max(hi - med, 1e-9) if r >= med else (med - r) / max(med - lo, 1e-9))
    level = min(1.0, math.ceil((len(need) + 1) * 0.8) / len(need))
    return max(0.25, float(np.quantile(need, level))), len(s)


def stretch(values, med, k):
    return [med + k * (v - med) for v in values]


def score_history(history, df):
    """Fill in the real outcome for forecasts whose 24h target candle has closed."""
    closes = dict(zip(df.t, df.close))
    for h in history:
        if "real" not in h and h["target_t"] in closes:
            real = closes[h["target_t"]]
            k, e = h["kronos"], h["ewma"]
            c = h.setdefault("kronos_cal", k)  # records from before calibration: the raw band
            real = float(real)
            h.update(real=real, hit=bool(k[0] <= real <= k[-1]), hit_ewma=bool(e[0] <= real <= e[-1]),
                     hit_cal=bool(c[0] <= real <= c[-1]),
                     dir_ok=bool((k[4] - h["last"]) * (real - h["last"]) > 0),
                     crps=crps(k, real), crps_ewma=crps(e, real), crps_cal=crps(c, real))
    return history


def scoreboard(history):
    s = [h for h in history if "real" in h]
    n = len(s)
    if n == 0:
        return {"n": 0}
    dir_k = sum(h["dir_ok"] for h in s)
    m = lambda key: round(float(np.mean([h[key] for h in s])), 4)
    width = lambda key: round(float(np.mean([(h[key][-1] - h[key][0]) / h["last"] * 100 for h in s])), 3)
    # Hourly forecasts of 24h moves overlap: only ~1 in 24 is independent, so the CI and the warning use that count.
    n_ind = max(1, n // HOURS)
    return {"n": n, "independent": n_ind, "target_coverage": 0.8,
            "coverage": m("hit"), "coverage_ewma": m("hit_ewma"), "coverage_cal": m("hit_cal"),
            "crps": m("crps"), "crps_ewma": m("crps_ewma"), "crps_cal": m("crps_cal"),
            "width_pct": width("kronos"), "width_pct_ewma": width("ewma"), "width_pct_cal": width("kronos_cal"),
            "direction": round(dir_k / n, 4), "direction_ci": wilson(dir_k / n * n_ind, n_ind),
            "too_few": n_ind < 100}


# ---------- paper trading ----------

# Paper only. Costs are deliberately pessimistic (exchange taker fee + slippage, both sides).
PAPER = {"start": 1000.0, "fee": 0.001, "slip": 0.0005, "edge": 0.005, "agree": 0.6, "stop": 0.02, "risk": 0.005}


def momentum_signal(df):
    """Benchmark signal: the last 24h move, and the share of the last 24 hourly moves that went the same way."""
    c = df.close.to_numpy()
    move = c[-1] / c[-HOURS - 1] - 1
    agree = float((np.sign(np.diff(c[-HOURS - 1:])) == np.sign(move)).mean()) if move else 0.5
    return float(move), agree


def paper_step(state, df, med, agree, words=("median", "paths")):
    """Advance a paper account one hour: close a position on stop/24h, else maybe open one from the signal.
    med = expected 24h move as a fraction, agree = share of evidence pointing the same way."""
    s = state or {"cash": PAPER["start"], "pos": None, "trades": [], "equity": [], "skips": {}, "log": [], "next_id": 1}
    last_t, last = int(df.t.iloc[-1]), float(df.close.iloc[-1])
    if s["equity"] and s["equity"][-1][0] >= last_t:
        return s  # this hour already processed
    log = lambda kind, text: s["log"].append([last_t, kind, text])

    p = s["pos"]
    if p:
        sign = 1 if p["side"] == "LONG" else -1
        for t, hi, lo, c in df[df.t > p["t"]][["t", "high", "low", "close"]].itertuples(index=False):
            stopped = lo <= p["stop"] if sign > 0 else hi >= p["stop"]
            if stopped or t >= p["t"] + 3600 * HOURS:
                px = (p["stop"] if stopped else float(c)) * (1 - sign * PAPER["slip"])
                fee_out = p["units"] * px * PAPER["fee"]
                gross = sign * p["units"] * (px - p["entry"])
                s["cash"] += gross - fee_out
                net = gross - fee_out - p["fee_in"]
                s["trades"].append({"id": p["id"], "side": p["side"], "t_in": p["t"], "t_out": int(t), "entry": round(p["entry"], 2),
                                    "exit": round(px, 2), "notional": round(p["notional"], 2), "fees": round(p["fee_in"] + fee_out, 2),
                                    "pnl": round(net, 2), "why": "stop" if stopped else "24h"})
                log("win" if net > 0 else "loss",
                    f"#{p['id']} {p['side']} closed ({'stop' if stopped else '24h'}) at {px:,.0f} · {'+' if net >= 0 else '-'}${abs(net):,.2f} after ${p['fee_in'] + fee_out:,.2f} fees")
                p = None
                break
        s["pos"] = p

    equity = s["cash"] + (0 if not p else (1 if p["side"] == "LONG" else -1) * p["units"] * (last - p["entry"]))
    if p:
        reason = "position open"
    elif abs(med) < PAPER["edge"]:
        reason = f"{words[0]} move under 0.5%"
    elif agree < PAPER["agree"]:
        reason = f"{words[1]} split"
    else:
        reason = None
        sign = 1 if med > 0 else -1
        entry = last * (1 + sign * PAPER["slip"])
        notional = equity * PAPER["risk"] / PAPER["stop"]  # lose ~0.5% of equity if the 2% stop hits
        fee_in = notional * PAPER["fee"]
        s["cash"] -= fee_in
        s["pos"] = {"id": s["next_id"], "side": "LONG" if sign > 0 else "SHORT", "t": last_t, "entry": entry,
                    "units": notional / entry, "notional": notional, "fee_in": fee_in, "stop": entry * (1 - sign * PAPER["stop"])}
        log("in", f"#{s['next_id']} {s['pos']['side']} ${notional:,.0f} at {entry:,.0f} · {words[0]} {med:+.2%} · {agree:.0%} of {words[1]} agree · stop {s['pos']['stop']:,.0f}")
        s["next_id"] += 1
        equity -= fee_in
    if reason:
        s["skips"][reason] = s["skips"].get(reason, 0) + 1
        if reason != "position open":
            log("skip", f"skip · {reason} ({words[0]} {med:+.2%}, {agree:.0%} agree)")

    s["equity"] = (s["equity"] + [[last_t, round(equity, 2)]])[-24 * 90:]
    s["log"], s["trades"] = s["log"][-40:], s["trades"][-500:]
    return s


def paper_stats(s):
    tr, eq = s["trades"], [e for _, e in s["equity"]] or [PAPER["start"]]
    wins = sum(t["pnl"] > 0 for t in tr)
    peak, dd = eq[0], 0.0
    for e in eq:
        peak = max(peak, e)
        dd = max(dd, (peak - e) / peak)
    p = s["pos"]
    return {"start": PAPER["start"], "equity": eq[-1], "return_pct": round((eq[-1] / PAPER["start"] - 1) * 100, 2),
            "trades": len(tr), "wins": wins, "win_rate": round(wins / len(tr), 4) if tr else None, "win_ci": wilson(wins, len(tr)),
            "pnl": round(sum(t["pnl"] for t in tr), 2), "fees": round(sum(t["fees"] for t in tr), 2),
            "max_dd_pct": round(dd * 100, 2), "too_few": len(tr) < 30,
            "open": p and {k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items() if k != "units"},
            "rules": PAPER}


# ---------- one run ----------

def run(df, paths, history, model_meta, source, paper=None, timings=None, symbol="BTCUSDT"):
    """Pure step: candles + paths + state -> (forecast json, history, paper states, alert or None).
    paper = {"kronos": account, "momentum": benchmark account} or None to start fresh."""
    last_t, last = int(df.t.iloc[-1]), float(df.close.iloc[-1])
    t0 = time.perf_counter()
    history = score_history(history, df)
    t_score = time.perf_counter() - t0

    alert = None
    prev = next((h for h in reversed(history) if h["t"] == last_t - 3600), None)
    if prev and not prev["band1"][0] <= last <= prev["band1"][1]:
        alert = f"{symbol[:-4]} closed {last:,.2f}, outside last hour's p10-p90 band ({prev['band1'][0]:,.0f}-{prev['band1'][1]:,.0f})"

    q = np.quantile(paths, [0.1, 0.5, 0.9], axis=0)
    final_dec = np.quantile(paths[:, -1], DECILES).tolist()
    ewma = ewma_deciles(df.close.to_numpy(), last)
    up = int((paths[:, -1] > last).sum())
    k, k_n = calib_factor(history)
    # ponytail: k is learned at the 24h step and applied to every step; fine for display, re-fit per step if early steps matter
    p10c = [m - k * (m - lo) for lo, m in zip(q[0], q[1])]
    p90c = [m + k * (hi - m) for hi, m in zip(q[2], q[1])]
    history.append({"t": last_t, "target_t": last_t + 3600 * HOURS, "last": last,
                    "kronos": [round(x, 2) for x in final_dec], "ewma": [round(x, 2) for x in ewma],
                    "kronos_cal": [round(x, 2) for x in stretch(final_dec, final_dec[4], k)],
                    "band1": [round(p10c[0], 2), round(p90c[0], 2)]})

    t0 = time.perf_counter()
    paper = paper or {}
    med = float(q[1][-1]) / last - 1
    paper["kronos"] = paper_step(paper.get("kronos"), df, med, (up if med > 0 else len(paths) - up) / len(paths))
    paper["momentum"] = paper_step(paper.get("momentum"), df, *momentum_signal(df), words=("24h", "hourly moves"))
    t_paper = time.perf_counter() - t0

    seen = {h["t"] for h in history}
    rets = (paths[:, -1] / last - 1) * 100
    lim = max(1.0, math.ceil(float(np.abs(rets).max())))
    counts, edges = np.histogram(rets, bins=24, range=(-lim, lim))
    timings = timings or {}
    pipeline = {
        "stages": [{"name": "fetch candles", "s": timings.get("fetch")}, {"name": "forecast", "s": timings.get("forecast")},
                   {"name": "score past forecasts", "s": round(t_score, 3)}, {"name": "paper account", "s": round(t_paper, 3)}],
        "runs": [[last_t - 3600 * i, (last_t - 3600 * i) in seen] for i in range(47, -1, -1)],  # a gap = a missed hourly run
        "returns_hist": {"edges": edges.round(2).tolist(), "counts": counts.tolist()},
        "drift": {"all": scoreboard(history), "week": scoreboard(history[-24 * 7:])},
        "skips": dict(sorted(paper["kronos"]["skips"].items(), key=lambda kv: -kv[1])),
    }

    out = {
        "schema": "fumo.kronos.v1",
        "generated_at": int(time.time()),
        "symbol": symbol, "interval": "1h", "source": source, "mode": "paper",
        "model": model_meta,
        "last": {"t": last_t, "close": last},
        "candles": df[["t", "open", "high", "low", "close"]].tail(KEEP_CANDLES).round(2).values.tolist(),
        "forecast": {
            "t": [last_t + 3600 * i for i in range(1, HOURS + 1)],
            "p10": q[0].round(2).tolist(), "p50": q[1].round(2).tolist(), "p90": q[2].round(2).tolist(),
            "p10_cal": [round(v, 2) for v in p10c], "p90_cal": [round(v, 2) for v in p90c],
            "paths": paths[:KEEP_PATHS].round(0).tolist(),
        },
        "lean": {"up": up, "down": len(paths) - up, "n": len(paths)},
        "alert": alert,
        "calibration": {"k": round(k, 3), "from_scored": k_n, "active": k_n >= CAL_MIN, "window": CAL_WINDOW},
        "ewma_band": [round(ewma[0], 2), round(ewma[-1], 2)],
        "score": scoreboard(history),
        "paper": paper_stats(paper["kronos"]) | {"trade_list": paper["kronos"]["trades"][-20:],
                                                 "equity_curve": paper["kronos"]["equity"][-24 * 30:], "log": paper["kronos"]["log"][-20:]},
        "benchmark": paper_stats(paper["momentum"]) | {"name": "24h momentum", "equity_curve": paper["momentum"]["equity"][-24 * 30:]},
        "pipeline": pipeline,
        "recent": [{key: h[key] for key in ("t", "last", "real", "hit", "hit_cal", "dir_ok")} | {"p10": h["kronos"][0], "p50": h["kronos"][4], "p90": h["kronos"][-1]}
                   for h in history if "real" in h][-12:],
        "log": [f"{source}: {len(df)} closed 1h candles, last {datetime.fromtimestamp(last_t, timezone.utc):%Y-%m-%d %H:%M} UTC",
                f"{model_meta['name']}: {len(paths)} paths x {HOURS}h, top_p {model_meta['top_p']}",
                f"median {q[1][-1]:,.0f} · p10 {q[0][-1]:,.0f} · p90 {q[2][-1]:,.0f} · {up}/{len(paths)} paths up",
                f"scored {sum('real' in h for h in history)} past forecasts against real closes",
                f"band calibration x{k:.2f} " + (f"from last {k_n} scored" if k_n >= CAL_MIN else f"inactive until {CAL_MIN} scored ({k_n} so far)")],
    }
    return out, history, paper, alert


# ---------- io ----------

def load_history(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save(folder, out, history, paper):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "forecast.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    with open(os.path.join(folder, "history.jsonl"), "w") as f:
        f.writelines(json.dumps(h, separators=(",", ":")) + "\n" for h in history[-24 * 400:])  # ~13 months
    with open(os.path.join(folder, "paper.json"), "w") as f:
        json.dump(paper, f, separators=(",", ":"))


def ntfy(msg):
    topic = os.environ.get("NTFY_TOPIC")
    if topic and msg:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode(), method="POST",
                                     headers={"Title": "FUMO forecast", "Tags": "chart_with_upwards_trend"})
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:  # an alert failing must not lose the forecast
            print(f"ntfy failed: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", type=int, metavar="HOURS", help="offline demo: replay this many synthetic hours")
    ap.add_argument("--model", choices=MODELS, default=os.environ.get("KRONOS_MODEL", "mini"))
    ap.add_argument("--paths", type=int, default=int(os.environ.get("KRONOS_PATHS", 64)))
    ap.add_argument("--symbols", default=os.environ.get("SYMBOLS", ",".join(SYMBOLS)), help="comma list, e.g. BTCUSDT,ETHUSDT")
    a = ap.parse_args()
    symbols = [x.strip().upper() for x in a.symbols.split(",") if x.strip()]
    unknown = [x for x in symbols if x not in SYMBOLS]
    if unknown:
        sys.exit(f"unknown symbols {unknown}; choose from {list(SYMBOLS)}")

    if a.fake:
        fx = {"ZAR": 16.5382, "date": "2026-11-06", "source": "fixed demo rate (ECB 6 Nov 2026)"}
        for n, sym in enumerate(symbols):
            candles = synthetic_candles(CONTEXT + a.fake, seed=4 + n, start_price=FAKE_START[sym])
            meta = {"name": "random-walk (FAKE)", "params": "-", "context": CONTEXT, "samples": a.paths, "top_p": 1.0}
            history, paper = [], None
            for i in range(a.fake, -1, -1):  # replay hour by hour so the scoreboard and paper account have data
                df = candles.iloc[: len(candles) - i].tail(CONTEXT).reset_index(drop=True)
                out, history, paper, alert = run(df, random_walk_paths(df, a.paths), history, meta, "synthetic", paper,
                                                 {"fetch": 0.0, "forecast": 0.0}, sym)
            save(os.path.join(DATA, sym), out | {"fx": fx}, history, paper)
            print(f"fake {sym}: {out['score']['n']} scored, band x{out['calibration']['k']}, "
                  f"kronos {out['paper']['return_pct']}% vs momentum {out['benchmark']['return_pct']}%")
        return

    fx, failed = None, []
    for sym in symbols:  # one coin failing must not stop (or lose) the others
        folder = os.path.join(DATA, sym)
        try:
            prev = load_json(os.path.join(folder, "forecast.json")) or {}
            fx = fx or fetch_fx(prev.get("fx"))
            t0 = time.time()
            df, source = fetch_candles(sym)
            t1 = time.time()
            paths = kronos_paths(df, a.paths, a.model)
            repo, _, _, _, params = MODELS[a.model]
            meta = {"name": repo.split("/")[1], "params": params, "context": CONTEXT, "samples": a.paths, "top_p": 1.0,
                    "seconds": round(time.time() - t0, 1)}
            out, history, paper, alert = run(df, paths, load_history(os.path.join(folder, "history.jsonl")), meta, source,
                                             load_json(os.path.join(folder, "paper.json")),
                                             {"fetch": round(t1 - t0, 2), "forecast": round(time.time() - t1, 2)}, sym)
            save(folder, out | {"fx": fx}, history, paper)
            ntfy(alert)
            print(f"[{sym}] " + " | ".join(out["log"]) + f" | took {meta['seconds']}s" + (f" | ALERT: {alert}" if alert else ""))
        except Exception as e:
            failed.append(sym)
            print(f"[{sym}] FAILED: {e!r}", file=sys.stderr)
    if failed:
        ntfy(f"Forecast job failed for {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
