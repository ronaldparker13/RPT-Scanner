"""Futures 'in play' ranking: which markets are waking up.

Three signs, all from daily bars (simple-average ATRs here; the stock scanner and the Pine use Wilder's ATR):
  1. Volatility expanding  : ATR(20) / ATR(100) >= 1.3
  2. Leaving its range     : close within 0.5% of a 50-day high / low, OR closes outside the prior 20-day range on
                             3 of the last 5 days
  3. Participation         : volume(20) / volume(100)  (scored, not required)
IN PLAY = 1 and 2. WAKING = score >= 40 (one of the two, or both partially) - volume is scored but not required.
Side: close above (below) the 20-day with the 20-day higher (lower) than 3 days ago.
Markets with incomplete bars (missing high / low / close in the last 5 days) or stale data (last bar more than
one session behind the freshest market) are reported as "no data" rather than scored.
"""
import numpy as np
import pandas as pd

# Yahoo continuous-contract tickers. Label, TradingView symbol for the link.
FUTURES = [   # TopStep-tradable only: CME, CBOT, COMEX, NYMEX. No ICE (coffee, sugar, cocoa, dollar index).
    ("NQ=F", "Nasdaq 100", "CME_MINI:MNQ1!"),
    ("ES=F", "S&P 500", "CME_MINI:MES1!"),
    ("RTY=F", "Russell 2000", "CME_MINI:M2K1!"),
    ("YM=F", "Dow", "CBOT_MINI:MYM1!"),
    ("GC=F", "Gold", "COMEX:MGC1!"),
    ("SI=F", "Silver", "COMEX:SIL1!"),
    ("HG=F", "Copper", "COMEX:MHG1!"),
    ("PL=F", "Platinum", "NYMEX:PL1!"),
    ("CL=F", "Crude oil", "NYMEX:MCL1!"),
    ("NG=F", "Natural gas", "NYMEX:MNG1!"),
    ("RB=F", "Gasoline", "NYMEX:RB1!"),
    ("HO=F", "Heating oil", "NYMEX:HO1!"),
    ("BTC=F", "Bitcoin", "CME:MBT1!"),
    ("ETH=F", "Ether", "CME:MET1!"),
    ("ZB=F", "30-yr bond", "CBOT:ZB1!"),
    ("ZN=F", "10-yr note", "CBOT:ZN1!"),
    ("ZF=F", "5-yr note", "CBOT:ZF1!"),
    ("6E=F", "Euro", "CME:M6E1!"),
    ("6J=F", "Yen", "CME:6J1!"),
    ("6B=F", "British pound", "CME:M6B1!"),
    ("6A=F", "Australian dollar", "CME:M6A1!"),
    ("6C=F", "Canadian dollar", "CME:6C1!"),
    ("ZC=F", "Corn", "CBOT:ZC1!"),
    ("ZS=F", "Soybeans", "CBOT:ZS1!"),
    ("ZW=F", "Wheat", "CBOT:ZW1!"),
    ("LE=F", "Live cattle", "CME:LE1!"),
    ("HE=F", "Lean hogs", "CME:HE1!"),
]


def in_play_table(bars: dict) -> pd.DataFrame:
    rows = []
    latest = max((bars[s].index[-1] for s, _, _ in FUTURES if s in bars and len(bars[s])), default=None)
    for yf_sym, name, tv in FUTURES:
        df = bars.get(yf_sym)
        if df is None or len(df) < 120 or df[["High", "Low", "Close"]].tail(5).isna().any().any() or (latest is not None and (latest - df.index[-1]).days > 3):
            rows.append(dict(symbol=yf_sym, name=name, tv=tv, price=np.nan, vol_ratio=np.nan, part=np.nan, breakout=False, direction="no data", ext_atr50=np.nan, chg20=np.nan, score=-1.0, in_play=False, asof=None))
            continue
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"].replace(0, np.nan)
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr20, atr100 = tr.rolling(20).mean(), tr.rolling(100).mean()
        vol20, vol100 = v.rolling(20).mean(), v.rolling(100).mean()
        ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
        hi50, lo50 = h.rolling(50).max(), l.rolling(50).min()
        hi20p, lo20p = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
        px = float(c.iloc[-1])
        vol_ratio = float(atr20.iloc[-1] / atr100.iloc[-1]) if atr100.iloc[-1] > 0 else np.nan
        part = float(vol20.iloc[-1] / vol100.iloc[-1]) if vol100.iloc[-1] and vol100.iloc[-1] > 0 else np.nan
        above = (c > hi20p).tail(5).sum(); below = (c < lo20p).tail(5).sum()
        at_hi50 = px >= hi50.iloc[-1] * 0.995; at_lo50 = px <= lo50.iloc[-1] * 1.005
        breakout = bool(at_hi50 or at_lo50 or above >= 3 or below >= 3)
        direction = "long" if px > ma20.iloc[-1] and ma20.iloc[-1] > ma20.iloc[-4] else "short" if px < ma20.iloc[-1] and ma20.iloc[-1] < ma20.iloc[-4] else "flat"
        ext = float((px - ma50.iloc[-1]) / atr20.iloc[-1]) if atr20.iloc[-1] > 0 else np.nan
        chg20 = float((px / c.iloc[-21] - 1) * 100)
        # score: volatility expansion carries the most weight, then breakout, then participation
        score = 0.0
        if not np.isnan(vol_ratio): score += min(max(vol_ratio - 1.0, 0), 1.0) * 50
        if breakout: score += 30
        if not np.isnan(part): score += min(max(part - 1.0, 0), 1.0) * 20
        in_play = (not np.isnan(vol_ratio) and vol_ratio >= 1.3) and breakout
        rows.append(dict(symbol=yf_sym, name=name, tv=tv, price=px, vol_ratio=vol_ratio, part=part, breakout=breakout,
                         direction=direction, ext_atr50=ext, chg20=chg20, score=round(score, 1), in_play=in_play, asof=df.index[-1].strftime("%Y-%m-%d")))
    t = pd.DataFrame(rows)
    return t.sort_values("score", ascending=False).reset_index(drop=True) if not t.empty else t


def latest_date(t: pd.DataFrame):
    return max((a for a in t["asof"].dropna()), default=None) if not t.empty else None


def html_section(t: pd.DataFrame) -> str:
    """the table only (used inside the standalone futures page)"""
    if t.empty:
        return ""
    import html as H
    out = ["<h2>Futures: what's in play (volatility expanding + leaving its range + volume)</h2><div class='scroll'><table>",
           "<tr><th>#</th><th>Ticker</th><th>Market</th><th>Status</th><th>Side</th><th class='num'>ATR 20/100</th><th class='num'>Vol 20/100</th><th class='num'>20d move</th><th class='num'>vs 50</th><th class='num'>Score</th></tr>"]
    for i, r in t.iterrows():
        if r["direction"] == "no data":
            out.append(f"<tr><td>{i+1}</td><td>{H.escape(r['tv'].split(':')[1])}</td><td>{H.escape(r['name'])}</td><td class='grp'>no data</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>")
            continue
        status = "<span class='tag BREAKOUT'>IN PLAY</span>" if r["in_play"] else ("<span class='tag BREAKOUT_READY'>waking</span>" if r["score"] >= 40 else "<span class='grp'>quiet</span>")
        side = r["direction"]
        tick = r["tv"].split(":")[1]
        part_txt = 'n/a' if np.isnan(r['part']) else f"{r['part']:.2f}"
        out.append(f"<tr><td>{i+1}</td><td><a class='t' href='https://www.tradingview.com/chart/?symbol={H.escape(r['tv'])}' target='tradingview'>{H.escape(tick)}</a></td><td>{H.escape(r['name'])}</td>"
                   f"<td>{status}</td><td class='{side}'>{side}</td><td class='num'>{r['vol_ratio']:.2f}</td><td class='num'>{part_txt}</td>"
                   f"<td class='num'>{r['chg20']:+.1f}%</td><td class='num'>{r['ext_atr50']:+.1f} ATR</td><td class='num'>{r['score']:.0f}</td></tr>")
    out.append("</table></div><div class='kv'>IN PLAY = 20-day ATR at least 30% above the 100-day ATR AND price leaving its range (close within 0.5% of a 50-day high / low, or outside the prior 20-day range on 3 of the last 5 days). Waking = score 40+ (partway there; volume is scored, not required). Side = close vs the 20-day, with the 20-day higher or lower than 3 days ago. Trade the side, not the market.</div>")
    return "".join(out)


def build_page(t: pd.DataFrame, run_date: str, css: str, note: str = "", session_label: str = "After the close") -> str:
    """Standalone futures board."""
    import datetime as dt
    n_play = int(t["in_play"].sum()) if not t.empty else 0
    body = html_section(t) if not t.empty else "<div class='kv'>No futures data tonight.</div>"
    return ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>RPT Futures — {run_date}</title><style>{css}</style></head><body><div class='wrap'>"
            f"<h1>RPT Futures</h1><div class='sub'>{session_label} · {run_date} · {len(t)} TopStep markets · {n_play} in play · "
            f"<a class='dl' href='futures_watchlist.txt' download='RPT Futures {run_date}.txt'>⬇ Download watchlist</a> · "
            f"<a class='dl' href='index.html'>→ Stock board</a></div>"
            + body +
            "<h2>How to read it</h2><div class='kv'>A market is IN PLAY when its 20-day ATR is at least 30% above its 100-day ATR and price is leaving its range (a close within 0.5% of a 50-day high / low, or 3 of the last 5 closes outside the prior 20-day range). Waking = score 40+, partway there; volume raises the score but is not required. Side = price vs a rising or falling 20-day; trade with it. vs 50 is the stretch from the 50-day in ATR, same gauge as the stock readout. Tap a market to open the micro in TradingView.</div>"
            + (f"<div class='foot'>{note}</div>" if note else "") +
            f"<div class='foot'>RPT Scanner · futures board · generated {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC</div></div></body></html>")


def build_watchlist(t: pd.DataFrame) -> str:
    """TradingView import format: IN PLAY, WAKING, then the rest, each in score order."""
    if t.empty:
        return "###FUTURES"
    parts = []
    t = t[t["direction"] != "no data"]
    for head, mask in (("IN PLAY", t["in_play"]), ("WAKING", (~t["in_play"]) & (t["score"] >= 40)), ("QUIET", (~t["in_play"]) & (t["score"] < 40))):
        sub = t[mask]
        if sub.empty:
            continue
        parts.append(f"###{head}")
        parts.extend(sub["tv"].tolist())
    return ",".join(parts)


def discord_post(webhook, t: pd.DataFrame, page_url: str, run_date: str):
    """Second embed of the night: the top six futures."""
    import datetime as dt, requests
    if not webhook or t.empty:
        return
    nice = dt.datetime.strptime(run_date, "%Y-%m-%d").strftime("%b %-d")
    top = t[t["direction"] != "no data"].head(6)
    n_play = int(t["in_play"].sum())
    color = 0xff9800 if n_play else 0x8a8f98
    lines = []
    for _, r in top.iterrows():
        flag = "🔥" if r["in_play"] else ("◐" if r["score"] >= 40 else "·")
        side = {"long": "LONG", "short": "SHORT"}.get(r["direction"], "flat")
        tick = r["tv"].split(":")[1]
        link = f"https://www.tradingview.com/chart/?symbol={r['tv']}"
        # one heading line per market (Discord's largest embed text), numbers on the line below
        lines.append(f"### {flag} [{tick}]({link}) {r['name']} · {side}\nATR x{r['vol_ratio']:.2f} · {r['chg20']:+.1f}% over 20d · {r['ext_atr50']:+.1f} ATR vs 50")
    embed = {"title": f"RPT Futures · {nice} · {n_play} in play", "url": page_url or None, "color": color,
             "description": "🔥 in play (ATR expanding + leaving its range) · ◐ waking · · quiet\n" + "\n".join(lines),
             "footer": {"text": "Trade the side, not the market · tap the title for the full board"}}
    try:
        requests.post(webhook, json={"embeds": [embed]}, timeout=20)
    except Exception as ex:
        print("futures discord post failed:", ex)
