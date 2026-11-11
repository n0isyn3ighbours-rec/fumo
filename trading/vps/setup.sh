#!/usr/bin/env bash
# FUMO trading desk on your VPS: an hourly Kronos forecast for BTC, ETH and SOL, uploaded to FUMO cloud.
# Run once:  curl -fsSL <your FUMO site>/trading/vps/setup.sh | bash
# Paper only: no exchange keys anywhere. Re-running it updates the code and keeps your key.
set -euo pipefail
SITE="${FUMO_SITE:-https://fumo-hq.netlify.app}"
DIR="$HOME/fumo-trading"
mkdir -p "$DIR/model" "$DIR/data"
cd "$DIR"

if [ ! -s .env ]; then
  echo "Paste the value of 'fumo_trading_key' (Supabase dashboard -> your fumo project -> Vault), then press Enter:"
  read -rs KEY < /dev/tty
  [ -n "$KEY" ] || { echo "No key given."; exit 1; }
  ( umask 077; printf 'FUMO_TRADING_KEY=%s\n' "$KEY" > .env )
fi

echo "Downloading the forecast job…"
for f in forecast.py upload.py requirements.txt model/__init__.py model/kronos.py model/module.py; do
  curl -fsSL "$SITE/trading/$f" -o "$f"
done

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv 2>/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv .venv; }
fi
echo "Installing PyTorch (CPU) and the other libraries — a few minutes the first time…"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -r requirements.txt

cat > run.sh <<'RUN'
#!/usr/bin/env bash
# One hourly run: forecast, then upload. Output goes to run.log (last 2,000 lines kept).
cd "$(dirname "$0")"
set -a; . ./.env; set +a
{ date -u; .venv/bin/python forecast.py; .venv/bin/python upload.py; } >> run.log 2>&1
tail -n 2000 run.log > run.log.tmp && mv run.log.tmp run.log
RUN
chmod +x run.sh
( crontab -l 2>/dev/null | grep -v 'fumo-trading/run.sh' || true; echo "7 * * * * $DIR/run.sh" ) | crontab -

echo "First run now (5–10 minutes: it downloads the model once)…"
./run.sh || true
tail -n 8 run.log
echo
echo "Done. It runs at 7 minutes past every hour. Open FUMO's Trading page to see it. Log: $DIR/run.log"
