"""Regime strip, group ranking, HTML page, JSON, Discord post."""
import datetime as dt
import html
import json
import os

import numpy as np
import pandas as pd
import requests

from . import config as C
from .data import BENCHMARKS, SECTOR_ETFS

SECTOR_NAMES = {"XLK": "Tech", "XLY": "Discretionary", "XLC": "Comm", "XLI": "Industrials", "XLF": "Financials",
                "XLB": "Materials", "XLE": "Energy", "XLV": "Health", "XLP": "Staples", "XLU": "Utilities", "XLRE": "Real Estate"}
OFFENSE = {"XLK", "XLY", "XLC", "XLI", "XLF", "XLB"}


def tv_link(symbol, exchange):
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{symbol}"


# ------------------------------------------------------------------ regime
def regime(bars: dict, stats: dict) -> dict:
    out = {}
    for b in BENCHMARKS:
        if b not in bars:
            continue
        c = bars[b]["Close"]
        m50, m200 = c.rolling(50).mean(), c.rolling(200).mean()
        out[b] = dict(price=float(c.iloc[-1]), above_50=bool(c.iloc[-1] > m50.iloc[-1]),
                      rising_50=bool(m50.iloc[-1] > m50.iloc[-6]),
                      ma50_over_200=bool(m50.iloc[-1] > m200.iloc[-1]) if len(c) >= 200 else True,
                      chg_1w=float((c.iloc[-1] / c.iloc[-6] - 1) * 100))
    n = max(len(stats), 1)
    up4 = sum(1 for s in stats.values() if s["up4"])
    dn4 = sum(1 for s in stats.values() if s["down4"])
    above50 = sum(1 for s in stats.values() if s["above_50"]) / n * 100
    above200 = sum(1 for s in stats.values() if s["above_200"]) / n * 100
    sec = []
    for e in SECTOR_ETFS:
        if e in bars:
            c = bars[e]["Close"]
            sec.append((e, float((c.iloc[-1] / c.iloc[-6] - 1) * 100)))
    sec.sort(key=lambda t: -t[1])
    offense_lead = sum(1 for e, _ in sec[:4] if e in OFFENSE) >= 3

    spy, qqq = out.get("SPY", {}), out.get("QQQ", {})
    idx_ok = spy.get("above_50", False) and spy.get("rising_50", False) and qqq.get("above_50", False)
    score = (2 if idx_ok else 0) + (1 if above50 >= 50 else 0) + (1 if offense_lead else 0) + (1 if up4 >= dn4 else 0)
    color = "green" if score >= 4 else "yellow" if score >= 2 else "red"
    label = {"green": "RISK ON", "yellow": "MIXED", "red": "RISK OFF"}[color]
    return dict(color=color, label=label, score=score, indexes=out, up4=up4, down4=dn4, pct_above_50=above50,
                pct_above_200=above200, sectors=sec, offense_leading=offense_lead)


# ------------------------------------------------------------------ groups
def group_rank(universe: pd.DataFrame, stats: dict) -> pd.DataFrame:
    rows = []
    for s, d in stats.items():
        rows.append(dict(symbol=s, rs_1m=d["rs_1m"], rs_3m=d["rs_3m"], hi20=d["new_20d_high"], a50=d["above_50"]))
    df = pd.DataFrame(rows).merge(universe[["symbol", "industry"]], on="symbol")
    g = df.groupby("industry").agg(members=("symbol", "count"), rs_1m=("rs_1m", "median"), rs_3m=("rs_3m", "median"),
                                   pct_hi20=("hi20", "mean"), pct_a50=("a50", "mean")).reset_index()
    g = g[g["members"] >= 3]
    g["pct_hi20"] *= 100
    g["pct_a50"] *= 100
    g["score"] = g["rs_1m"].rank(pct=True) * 40 + g["pct_hi20"].rank(pct=True) * 40 + g["pct_a50"].rank(pct=True) * 20
    return g.sort_values("score", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ setups table
def setup_rows(universe: pd.DataFrame, stats: dict, exchanges: dict, top_groups: set):
    from .setups import score
    meta = universe.set_index("symbol")
    rows = []
    for s, d in stats.items():
        for st in d["setups"]:
            kind, side, level, stop = st
            sc = score(st, d)
            if meta.loc[s, "industry"] in top_groups:
                sc += 10
            rows.append(dict(symbol=s, exchange=exchanges.get(s, meta.loc[s, "exchange"]), name=meta.loc[s, "name"],
                             industry=meta.loc[s, "industry"], setup=kind, side=side, score=sc,
                             price=d["price"], level=level, stop=stop, risk_pct=abs(d["price"] - stop) / d["price"] * 100,
                             adr=d["adr_pct"], rvol=d["rvol"], chg=d["chg_pct"], ext=d["ext_atr50"],
                             dollar_vol=d["avg_dollar"], chase=d["ext_atr50"] >= C.CHASE_ATR and kind != "EP",
                             top_group=meta.loc[s, "industry"] in top_groups))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("score", ascending=False)
    longs = df[df.side == "long"].head(C.MAX_SETUPS_PER_SIDE)
    shorts = df[df.side == "short"].head(C.MAX_SETUPS_PER_SIDE)
    return pd.concat([longs, shorts])


# ------------------------------------------------------------------ html
CSS = """
:root{--bg:#0e0f12;--card:#16181d;--txt:#e6e6e6;--mut:#8a8f98;--grn:#2ecc71;--red:#ff5252;--yel:#f1c40f;--ora:#ff9800;--pur:#e040fb;--cyn:#26c6da}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:18px}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px;margin-bottom:16px}
.strip{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:var(--card);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.pill{font-weight:700;padding:4px 12px;border-radius:999px;color:#000}.green{background:var(--grn)}.yellow{background:var(--yel)}.red{background:var(--red)}
.kv{color:var(--mut);font-size:13px}.kv b{color:var(--txt)}
h2{font-size:15px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;margin:18px 0 8px}
table{width:100%;border-collapse:collapse;background:var(--card);border-radius:10px;overflow:hidden}
th,td{padding:8px 10px;text-align:left;white-space:nowrap}th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase}
tr+tr td{border-top:1px solid #23262d}td.num{text-align:right;font-variant-numeric:tabular-nums}
a.t{color:var(--cyn);font-weight:700;text-decoration:none}a.t:hover{text-decoration:underline}
.tag{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:700;color:#000}
.EP{background:var(--ora)}.HVC{background:var(--pur)}.SECOND_CHANCE{background:var(--cyn)}.BREAKOUT{background:var(--grn)}.BREAKOUT_READY{background:#9ccc65}
.short .tag{background:var(--red)}.chase{color:var(--red);font-weight:700}.grp{color:var(--mut);font-size:12px}
.scroll{overflow-x:auto}.foot{color:var(--mut);font-size:12px;margin-top:20px}
@media(max-width:700px){td,th{padding:6px 7px;font-size:13px}}
"""


def fmt_dollar(v):
    return f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M"


def build_html(reg, groups, setups, universe, run_date, note=""):
    e = html.escape
    sec_top = ", ".join(f"{SECTOR_NAMES.get(k,k)} {v:+.1f}%" for k, v in reg["sectors"][:3])
    sec_bot = ", ".join(f"{SECTOR_NAMES.get(k,k)} {v:+.1f}%" for k, v in reg["sectors"][-3:])
    idx = " · ".join(f"{b} {'above' if v['above_50'] else 'BELOW'} 50 ({v['chg_1w']:+.1f}% 1w)" for b, v in reg["indexes"].items())
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             f"<title>{C.REPORT_TITLE} — {run_date}</title><style>{CSS}</style></head><body><div class='wrap'>",
             f"<h1>{C.REPORT_TITLE}</h1><div class='sub'>After the close · {run_date} · {len(universe)} stocks screened · tap a ticker to open it in TradingView</div>",
             f"<div class='strip'><span class='pill {reg['color']}'>{reg['label']}</span>",
             f"<span class='kv'>{idx}</span>",
             f"<span class='kv'>Above 50-day: <b>{reg['pct_above_50']:.0f}%</b> · Above 200: <b>{reg['pct_above_200']:.0f}%</b> · Up 4%+: <b>{reg['up4']}</b> / Down 4%+: <b>{reg['down4']}</b></span>",
             f"<span class='kv'>Sectors 1w — leading: <b>{e(sec_top)}</b> · lagging: <b>{e(sec_bot)}</b> · offense {'leading' if reg['offense_leading'] else 'not leading'}</span></div>"]
    # groups
    parts.append("<h2>Top groups (1-month RS, % at 20-day highs, % above 50)</h2><div class='scroll'><table><tr><th>#</th><th>Group</th><th class='num'>Members</th><th class='num'>RS 1m</th><th class='num'>RS 3m</th><th class='num'>At 20d highs</th><th class='num'>Above 50</th></tr>")
    for i, r in groups.head(C.TOP_GROUPS).iterrows():
        parts.append(f"<tr><td>{i+1}</td><td>{e(str(r['industry']))}</td><td class='num'>{int(r['members'])}</td><td class='num'>{r['rs_1m']:+.1f}%</td><td class='num'>{r['rs_3m']:+.1f}%</td><td class='num'>{r['pct_hi20']:.0f}%</td><td class='num'>{r['pct_a50']:.0f}%</td></tr>")
    parts.append("</table></div>")

    def table(df, title):
        parts.append(f"<h2>{title}</h2>")
        if df.empty:
            parts.append("<div class='kv'>Nothing qualified today.</div>")
            return
        parts.append("<div class='scroll'><table><tr><th>Ticker</th><th>Setup</th><th class='num'>Price</th><th class='num'>Level</th><th class='num'>Stop</th><th class='num'>Risk</th><th class='num'>ADR</th><th class='num'>RVOL</th><th class='num'>Today</th><th class='num'>vs 50</th><th class='num'>$ Vol</th><th>Group</th></tr>")
        for _, r in df.iterrows():
            chase = " <span class='chase'>DON'T CHASE</span>" if r["chase"] else ""
            star = " ★" if r["top_group"] else ""
            parts.append(f"<tr class='{r['side']}'><td><a class='t' href='{tv_link(r['symbol'], r['exchange'])}' target='_blank'>{e(r['symbol'])}</a><div class='grp'>{e(str(r['name']))[:28]}</div></td>"
                         f"<td><span class='tag {r['setup']}'>{r['setup'].replace('_',' ')}</span>{chase}</td>"
                         f"<td class='num'>{r['price']:.2f}</td><td class='num'>{r['level']:.2f}</td><td class='num'>{r['stop']:.2f}</td><td class='num'>{r['risk_pct']:.1f}%</td>"
                         f"<td class='num'>{r['adr']:.1f}%</td><td class='num'>{r['rvol']:.1f}x</td><td class='num'>{r['chg']:+.1f}%</td><td class='num'>{r['ext']:+.1f} ATR</td><td class='num'>{fmt_dollar(r['dollar_vol'])}</td><td class='grp'>{e(str(r['industry']))}{star}</td></tr>")
        parts.append("</table></div>")

    if setups.empty:
        table(setups, "Long setups")
    else:
        table(setups[setups.side == "long"], "Long setups")
        table(setups[setups.side == "short"], "Short setups (below the 200-day only)")
    parts.append("<h2>How to read it</h2><div class='kv'>EP = gap on volume that held (buy the opening-range high next session, stop the day's low). HVC = high-volume close near the high (buy the hold, stop the low). SECOND CHANCE = price back at a recent EP/HVC close on quiet volume (buy the reclaim, stop under the level). BREAKOUT READY = 30%+ run, tight base, within 5% of the base high (watch for the close through it). BREAKOUT = closed through the base high on volume today. ★ = in a top-10 group. Level = the price that matters; Stop = low/high of the day or just past the level; Risk = distance to the stop. Nothing here is a recommendation; it is a list of charts to open.</div>")
    if note:
        parts.append(f"<div class='foot'>{e(note)}</div>")
    parts.append(f"<div class='foot'>RPT Scanner · rules mirror the RPT Stock Chart Pine · generated {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC</div></div></body></html>")
    return "".join(parts)


# ------------------------------------------------------------------ discord
def discord_post(webhook, reg, setups, groups, page_url, run_date):
    if not webhook:
        return
    emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[reg["color"]]
    lines = [f"**RPT Scanner · {run_date}** {emoji} {reg['label']} · above 50-day {reg['pct_above_50']:.0f}% · up4 {reg['up4']} / down4 {reg['down4']}",
             "Top groups: " + ", ".join(groups.head(5)["industry"].tolist())]
    for side, head in (("long", "**Longs**"), ("short", "**Shorts**")):
        sub = setups[setups.side == side].head(8) if not setups.empty else setups
        if sub.empty:
            continue
        lines.append(head)
        for _, r in sub.iterrows():
            lines.append(f"`{r['symbol']:<6}` {r['setup'].replace('_',' '):<14} lvl {r['level']:.2f} · stop {r['stop']:.2f} · ADR {r['adr']:.1f}% · RVOL {r['rvol']:.1f}x{' ⚠️ extended' if r['chase'] else ''}")
    lines.append(f"Full page: {page_url}")
    try:
        requests.post(webhook, json={"content": "\n".join(lines)[:1900]}, timeout=20)
    except Exception as ex:
        print("discord post failed:", ex)
