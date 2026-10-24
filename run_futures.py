"""Standalone RPT Futures morning scan. NEVER writes stock pages or stock JSON.

    python run_futures.py             # Yahoo 30m, 4 futures, no Discord
    python run_futures.py --demo      # synthetic test data, no Discord

The independent GitHub workflow commits futures.html and futures_watchlist.txt,
then notifies Discord through notify_futures.py after the pages are committed.
"""
import argparse
import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scanner import futures as F
from scanner.report import CSS  # ORIGINAL STOCK/FUTURES CSS — never restyled


def fetch_30m():
    import yfinance as yf
    result = {}
    for sym, name, tv in F.FUTURES:
        try:
            d = yf.download(sym, period="30d", interval="30m", auto_adjust=False,
                            prepost=True, threads=False, progress=False, timeout=25)
            if d is None or d.empty:
                print(f"WARNING: {sym} 30m Yahoo returned no data")
                continue
            if isinstance(d.columns, pd.MultiIndex):
                if sym in d.columns.get_level_values(1):
                    d = d.xs(sym, axis=1, level=1)
                elif sym in d.columns.get_level_values(0):
                    d = d.xs(sym, axis=1, level=0)
            d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
            result[sym] = d
            print(f"{sym}: {len(d)} bars, newest bar starts {d.index[-1]}")
        except Exception as ex:
            print(f"WARNING: {sym} fetch error: {type(ex).__name__}: {ex}")
    return result


def demo_30m(now):
    rng = np.random.default_rng(42)
    end = pd.Timestamp(now).tz_convert("America/Chicago").floor("30min") - pd.Timedelta(minutes=30)
    index = pd.date_range(end=end, periods=550, freq="30min")
    out = {}
    for i, (sym, _, _) in enumerate(F.FUTURES):
        drift = (-0.06 if i % 2 == 0 else 0.05)
        steps = rng.normal(drift, .7, len(index))
        c = 100 + np.cumsum(steps)
        o = c - steps
        hi, lo = np.maximum(o, c) + .18, np.minimum(o, c) - .18
        out[sym] = pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c,
                                 "Volume": rng.integers(100, 500, len(index))}, index=index)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic bars to check presentation")
    args = ap.parse_args()
    now_ct = dt.datetime.now(ZoneInfo("America/Chicago"))
    bars = demo_30m(now_ct) if args.demo else fetch_30m()
    table = F.in_play_table(bars, now=now_ct)
    date = now_ct.strftime("%Y-%m-%d")
    label = (f"DEMO · {now_ct:%I:%M %p} CT" if args.demo else f"Morning scan · {now_ct:%I:%M %p} CT")
    note = "DEMO DATA — not a live signal" if args.demo else "Market data from Yahoo Finance; prices may be delayed"
    os.makedirs("docs", exist_ok=True)
    with open("docs/futures.html", "w", encoding="utf-8") as f:
        f.write(F.build_page(table, date, CSS, note, label))
    with open("docs/futures_watchlist.txt", "w", encoding="utf-8") as f:
        f.write(F.build_watchlist(table))
    payload = {"date": date, "session_label": label, "data_source": "Yahoo Finance 30m continuous contracts",
               "demo": args.demo, "futures": json.loads(table.replace({np.nan: None}).to_json(orient="records"))}
    with open("docs/futures_latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    n_valid = int((table["status"] != "NO DATA").sum())
    print(f"done: {n_valid}/4 scored, outputs futures-only; stock files untouched")
    print(table[["name", "status", "direction", "score", "asof"]].to_string(index=False))


if __name__ == "__main__":
    main()
