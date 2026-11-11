"""Upload this run's forecasts to FUMO cloud (Supabase).

Needs FUMO_TRADING_KEY: the `fumo_trading_key` value from your Supabase project's Vault. That key can only
upload trading forecasts. Run after forecast.py; setup.sh's hourly job does both.
"""
import json, os, pathlib, sys, urllib.error, urllib.request

SB_URL = "https://shodagzqvfueenvrjati.supabase.co"
SB_KEY = "sb_publishable_dUfMqtXP23Wx3dZYIuggoA__GBwVRI-"  # public by design; the trading key is what authorises uploads
DATA = pathlib.Path(__file__).resolve().with_name("data")


def main():
    key = os.environ.get("FUMO_TRADING_KEY", "").strip()
    if not key:
        sys.exit("FUMO_TRADING_KEY is not set")
    files = sorted(DATA.glob("*USDT/forecast.json"))
    if not files:
        sys.exit("no forecasts to upload yet")
    ok = 0
    for f in files:
        body = json.dumps({"k": key, "sym": f.parent.name, "payload": json.loads(f.read_text())}).encode()
        req = urllib.request.Request(f"{SB_URL}/rest/v1/rpc/fumo_trading_upload", data=body,
                                     headers={"apikey": SB_KEY, "Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=60).read()
            ok += 1
            print("uploaded", f.parent.name)
        except urllib.error.HTTPError as e:
            print("upload failed", f.parent.name, e.code, e.read()[:200].decode(errors="replace"), file=sys.stderr)
        except OSError as e:
            print("upload failed", f.parent.name, e, file=sys.stderr)
    sys.exit(0 if ok == len(files) else 1)


if __name__ == "__main__":
    main()
