# FUMO trading desk

Open it from FUMO's sidebar (**Trading**), or go to `https://<you>.github.io/fumo/trading/`. The page has a **← FUMO** link back.

Paper only. No exchange or wallet keys anywhere, and nothing trades. A research tool, not financial advice.

## What happens each hour

The *Trading forecast* workflow (`.github/workflows/trading-forecast.yml`) runs at 7 minutes past each hour. For BTC, ETH and SOL it:

1. Fetches the last 512 one-hour candles from Binance's market-data host, with Kraken as fallback.
2. Runs Kronos-mini for 64 sampled paths of the next 24 hours.
3. Scores past forecasts against what really happened, compared with an EWMA baseline, and recalibrates the band.
4. Steps two paper accounts, with the same rules and costs: one follows the Kronos signal, the other 24h momentum.

ZAR is for display only (ECB rate).

The results go to this repo's **`trading-data`** branch, one folder per coin. That branch is replaced with a single commit every run, so FUMO's `main` branch and its Pages site don't fill up with hourly commits. The page reads the files straight from that branch.

The first run takes 5–10 minutes while it downloads the model. Later runs are quicker. Until then, the page says there's no data yet.

To run it now: **Actions → Trading forecast → Run workflow**.

## Settings

In the workflow's `env:`:

| Setting | Default | Options |
|---|---|---|
| `KRONOS_MODEL` | `mini` | or `small` |
| `KRONOS_PATHS` | `64` | number of sampled paths |
| `SYMBOLS` | `BTCUSDT,ETHUSDT,SOLUSDT` | any of these three |

## Phone alerts (off)

The job can send an ntfy push whenever a coin closes outside the previous hour's p10–p90 band. That happens often: about 15 times a day across the three coins. So it's off by default.

To turn it on, add a repository secret named `NTFY_TOPIC`. Use a **different** topic from The Daily Edition's morning summary, or the alerts will bury it.

## Try it on your computer (optional)

```
cd trading
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python test_forecast.py          # 10 self-checks, no network
python forecast.py --fake 400    # fake data, no model download, to see the page filled
python -m http.server 8000       # from the fumo folder, then open http://localhost:8000/trading/
```

Locally the page reads `trading/data/`. That folder is git-ignored, so fake data never gets pushed. Use Python 3.11: the pinned numpy and pandas have no wheels for newer versions.

The model code in `model/` is Kronos (MIT licence), vendored unchanged from commit `67b630e`.
