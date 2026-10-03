"""RPT Scanner — nightly runner.

    python run.py            # live: Nasdaq universe + Yahoo bars
    python run.py --demo     # offline: synthetic data, same report

Writes docs/index.html, docs/latest.json, docs/history/<date>.html and posts to Discord
when DISCORD_WEBHOOK is set in the environment.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import pandas as pd

from scanner import config as C
from scanner import data as D
from scanner.setups import indicators
from scanner.report import regime, group_rank, setup_rows, build_html, discord_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap the universe (testing)")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs("docs/history", exist_ok=True)

    if args.demo:
        universe = D.demo_universe()
        bars = D.demo_bars(universe)
        exchanges = {}
        note = "DEMO DATA — synthetic bars, for layout only."
    else:
        universe = D.fetch_universe()
        if args.limit:
            universe = universe.sort_values("market_cap", ascending=False).head(args.limit)
        print(f"universe: {len(universe)} stocks ≥ ${C.MIN_MARKET_CAP/1e9:.0f}B")
        bars = D.fetch_bars(list(universe["symbol"]) + D.BENCHMARKS + D.SECTOR_ETFS)
        print(f"bars: {len(bars)} symbols in {time.time()-t0:.0f}s")
        exchanges = {}
        note = ""

    # per-stock stats, with the liquidity screen
    stats = {}
    for s in universe["symbol"]:
        if s not in bars:
            continue
        d = indicators(bars[s])
        if d is None or d["price"] < C.MIN_PRICE or d["avg_dollar"] < C.MIN_DOLLAR_VOLUME:
            continue
        if not (C.ADR_MIN <= d["adr_pct"] <= C.ADR_MAX):
            continue
        stats[s] = d
    universe = universe[universe["symbol"].isin(stats)].reset_index(drop=True)
    print(f"qualified: {len(stats)}")

    reg = regime(bars, stats)
    groups = group_rank(universe, stats)
    top = set(groups.head(C.TOP_GROUPS)["industry"])
    if not args.demo:
        hits = {s for s, d in stats.items() if d["setups"]}
        exchanges = D.fetch_exchanges(sorted(hits))
    setups = setup_rows(universe, stats, exchanges, top)

    run_date = (bars[next(iter(bars))].index[-1]).strftime("%Y-%m-%d")
    page = build_html(reg, groups, setups, universe, run_date, note)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(page)
    with open(f"docs/history/{run_date}.html", "w", encoding="utf-8") as f:
        f.write(page)
    out = dict(date=run_date, regime={k: v for k, v in reg.items() if k != "sectors"}, sectors=reg["sectors"],
               groups=groups.head(C.TOP_GROUPS).to_dict("records"),
               setups=setups.to_dict("records") if not setups.empty else [])
    with open("docs/latest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, default=str, indent=1)
    page_url = os.environ.get("PAGE_URL", "")
    discord_post(os.environ.get("DISCORD_WEBHOOK", ""), reg, setups, groups, page_url, run_date)
    print(f"done: {len(setups)} setups, regime {reg['label']}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
