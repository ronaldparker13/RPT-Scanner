"""30-minute futures continuation ranking for the four RPT trading markets.

Uses only completed 30m candles. The 65 SMA supplies multi-session bias;
causally confirmed price swings identify the countertrend reset and its first
opposing CHOCH. 20 SMA alignment is a bonus, never a prerequisite for CHOCH.

This is an opportunity watchlist, NOT a backtested trading signal. Yahoo futures
quotes can be delayed, and full-size Yahoo symbols proxy the micro contracts.
"""
import datetime as dt
import html as H
import math

import numpy as np
import pandas as pd

# Yahoo continuous contract -> displayed market -> TradingView MICRO chart.
FUTURES = [
    ("NQ=F", "Nasdaq 100", "CME_MINI:MNQ1!"),
    ("CL=F", "Crude oil", "NYMEX:MCL1!"),
    ("GC=F", "Gold", "COMEX:MGC1!"),
    ("BTC=F", "Bitcoin", "CME:MBT1!"),
]

PIVOT_LEFT = 2
PIVOT_RIGHT = 2  # a pivot is only available two FULL candles later
MIN_BARS = 190
RESET_WINDOW = 96
CHOCH_LOOKBACK = 16
MAX_DATA_AGE_MINUTES = 180


def _base(yf_sym, name, tv):
    return dict(symbol=yf_sym, name=name, tv=tv, price=np.nan,
                direction="no data", status="NO DATA", setup="insufficient data",
                score=np.nan, trend_score=0, reset_score=0, choch_score=0,
                alignment_score=0, slope_atr=np.nan, side_pct=np.nan,
                ma20=np.nan, ma65=np.nan, asof=None, lag_min=np.nan,
                signal_time=None, pivot_level=np.nan)


def _pivots(high: np.ndarray, low: np.ndarray):
    """Return (pivot index, value, confirmation index) for fully-confirmed swings."""
    highs, lows = [], []
    width = PIVOT_LEFT + PIVOT_RIGHT
    for i in range(PIVOT_LEFT, len(high) - PIVOT_RIGHT):
        h, l = high[i], low[i]
        if np.isfinite(h) and h > max(high[i-PIVOT_LEFT:i]) and h >= max(high[i+1:i+PIVOT_RIGHT+1]):
            highs.append((i, float(h), i + PIVOT_RIGHT))
        if np.isfinite(l) and l < min(low[i-PIVOT_LEFT:i]) and l <= min(low[i+1:i+PIVOT_RIGHT+1]):
            lows.append((i, float(l), i + PIVOT_RIGHT))
    return highs, lows


def _reset_at(j, side, close, ph, pl):
    """Return (structure valid, break level) using pivots already confirmed at j."""
    start = max(0, j - RESET_WINDOW)
    highs = [p for p in ph if start <= p[0] < j and p[2] <= j]
    lows = [p for p in pl if start <= p[0] < j and p[2] <= j]
    if len(highs) < 2 or len(lows) < 2:
        return False, np.nan
    h0, h1 = highs[-2], highs[-1]
    l0, l1 = lows[-2], lows[-1]
    # The final pivot belongs to the completed countertrend impulse. For a
    # bearish continuation, that impulse consists of HH/HL; bullish is LH/LL.
    if side == "short":
        valid = h1[1] > h0[1] and l1[1] > l0[1]
        # Require the structure to have emerged in the recent pullback.
        valid = valid and j - h1[0] <= 32 and j - l1[0] <= 40
        return valid, l1[1]
    if side == "long":
        valid = h1[1] < h0[1] and l1[1] < l0[1]
        valid = valid and j - h1[0] <= 40 and j - l1[0] <= 32
        return valid, h1[1]
    return False, np.nan


def _structure(side, close, high, low):
    """Find first break after each causally identifiable countertrend reset."""
    ph, pl = _pivots(high, low)
    n = len(close)
    # A candidate break only happens on its first close through the pivot.
    events = []
    for j in range(max(MIN_BARS, n - RESET_WINDOW), n):
        valid, level = _reset_at(j, side, close, ph, pl)
        if not valid or not np.isfinite(level):
            continue
        crossed = (close[j] < level <= close[j-1]) if side == "short" else (close[j] > level >= close[j-1])
        if crossed and (not events or j - events[-1][0] > 16):
            events.append((j, level))
    # A fresh countertrend reset must still be unbroken if it's a WATCH.
    valid_now, level_now = _reset_at(n-1, side, close, ph, pl)
    unbroken = (close[-1] >= level_now) if side == "short" else (close[-1] <= level_now)
    if not valid_now or not np.isfinite(level_now):
        unbroken = False
    recent = events[-1] if events and n - 1 - events[-1][0] <= CHOCH_LOOKBACK else None
    return bool(valid_now and unbroken), recent


def _classify(df: pd.DataFrame, now: dt.datetime):
    if df is None or len(df) < MIN_BARS:
        return None, "insufficient 30m history"
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex) or d.index.tz is None:
        return None, "missing bar timezone"
    d.index = d.index.tz_convert("America/Chicago")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    required = ("Open", "High", "Low", "Close")
    if any(col not in d for col in required):
        return None, "missing OHLC columns"
    d = d.dropna(subset=list(required))
    now_ct = pd.Timestamp(now).tz_convert("America/Chicago")
    # Yahoo timestamps are bar-start timestamps. Never score an unfinished bar.
    d = d[d.index + pd.Timedelta(minutes=30) <= now_ct - pd.Timedelta(minutes=1)]
    if len(d) < MIN_BARS:
        return None, "insufficient completed 30m bars"
    lag = (now_ct - (d.index[-1] + pd.Timedelta(minutes=30))).total_seconds() / 60
    if lag < 0 or lag > MAX_DATA_AGE_MINUTES:
        return None, f"stale Yahoo candles ({lag:.0f}m since last completed bar)"
    # If an intraday data window contains a large discontinuity near the end,
    # historical trend context remains available, but signal freshness is not.
    if (d.index[-1] - d.index[-2]).total_seconds() / 60 > 180:
        return None, "missing recent consecutive 30m bars"
    return d, ""


def in_play_table(bars: dict, now=None) -> pd.DataFrame:
    """Rank MNQ/MCL/MGC/MBT from 0 to 100; incomplete markets show NO DATA."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    rows = []
    for yf_sym, name, tv in FUTURES:
        r = _base(yf_sym, name, tv)
        d, error = _classify(bars.get(yf_sym), now)
        if d is None:
            r["setup"] = error
            rows.append(r)
            continue
        c, h, l = d["Close"], d["High"], d["Low"]
        ma65, ma20 = c.rolling(65).mean(), c.rolling(20).mean()
        tr = pd.concat([h - l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        if not np.isfinite(atr) or atr <= 0:
            r["setup"] = "invalid ATR"
            rows.append(r)
            continue
        slope = float((ma65.iloc[-1] - ma65.iloc[-97]) / atr)
        sign = 1 if slope > 0 else -1
        directions = []
        for step in (0, 24, 48, 72):
            delta = float(ma65.iloc[-1-step] - ma65.iloc[-25-step])
            directions.append(1 if delta > 0 else -1 if delta < 0 else 0)
        consistency = sum(x == sign for x in directions) / len(directions)
        side = ((c.iloc[-96:] > ma65.iloc[-96:]) if sign == 1 else (c.iloc[-96:] < ma65.iloc[-96:]))
        persistence = float(side.mean())
        strong_direction = abs(slope) >= 0.40 and consistency >= 0.75 and persistence >= 0.55
        direction = ("long" if sign == 1 else "short") if strong_direction else "flat"
        # Continuous 0..45 trend-quality score; no per-market point thresholds.
        trend = round(17 * min(abs(slope) / 2.0, 1.0) + 14 * consistency + 14 * min(persistence / 0.8, 1.0))
        if not strong_direction:
            trend = min(trend, 24)
        reset, recent = _structure(direction, c.to_numpy(), h.to_numpy(), l.to_numpy())
        reset_points = 25 if reset or recent else 0
        event_age = len(d) - 1 - recent[0] if recent else None
        event_points = (20 if event_age is not None and event_age <= 4 else 12 if event_age is not None else 0)
        below20 = float(ma20.iloc[-1]) > float(c.iloc[-1])
        below65 = float(ma65.iloc[-1]) > float(c.iloc[-1])
        curl = float(ma20.iloc[-1] - ma20.iloc[-5])
        aligned = (direction == "short" and below20 and below65 and curl < 0) or (direction == "long" and not below20 and not below65 and curl > 0)
        curling = (direction == "short" and curl < 0) or (direction == "long" and curl > 0)
        align_points = 10 if aligned else 5 if curling else 0
        score = int(max(0, min(100, trend + reset_points + event_points + align_points)))
        if recent:
            status = "CHOCH"
            setup = f"First 30m CHOCH · {event_age} bars ago"
        elif reset:
            status = "WATCH"
            setup = "Countertrend reset · waiting for first CHOCH"
        elif strong_direction and aligned:
            status = "ALIGNED"
            setup = "Trend aligned · no fresh reset"
        elif strong_direction:
            status = "TREND"
            setup = "Established trend · no fresh CHOCH"
        else:
            status = "CHOP"
            setup = "Weak or mixed 65 SMA trend"
        end_ct = d.index[-1] + pd.Timedelta(minutes=30)
        r.update(price=float(c.iloc[-1]), direction=direction, status=status,
                 setup=setup, score=score, trend_score=int(trend),
                 reset_score=int(reset_points), choch_score=int(event_points),
                 alignment_score=int(align_points), slope_atr=round(slope, 2),
                 side_pct=round(persistence * 100), ma20=round(float(ma20.iloc[-1]), 4),
                 ma65=round(float(ma65.iloc[-1]), 4), lag_min=round((pd.Timestamp(now).tz_convert('America/Chicago') - end_ct).total_seconds()/60),
                 asof=end_ct.strftime("%Y-%m-%d %H:%M CT"),
                 signal_time=(d.index[recent[0]] + pd.Timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M CT") if recent else None,
                 pivot_level=float(recent[1]) if recent else np.nan)
        rows.append(r)
    table = pd.DataFrame(rows)
    return table.sort_values(["score", "name"], ascending=[False, True], na_position="last").reset_index(drop=True)


def latest_date(t: pd.DataFrame):
    vals = [str(s).split(" ")[0] for s in t.get("asof", pd.Series(dtype=str)).dropna()]
    return max(vals) if vals else None


def _tick(r):
    return r["tv"].split(":")[-1]


def _link(r):
    return f"https://www.tradingview.com/chart/?symbol={r['tv']}"


def html_section(t: pd.DataFrame) -> str:
    """Retain existing dark-page table wrappers, CSS classes and link styling."""
    if t.empty:
        return ""
    out = ["<h2>Futures: 30m 65 SMA trend + countertrend resumption opportunities</h2><div class='scroll'><table>",
           "<tr><th>#</th><th>Ticker</th><th>Market</th><th>Status</th><th>Side</th><th class='num'>65 trend</th><th class='num'>Price-side</th><th>Countertrend / CHOCH</th><th class='num'>Data as of</th><th class='num'>Score</th></tr>"]
    for i, r in t.iterrows():
        tick, link = H.escape(_tick(r)), H.escape(_link(r), quote=True)
        name = H.escape(r["name"])
        anchor = f"<a class='t' href='{link}' target='tradingview'>{tick}</a>"
        if r["status"] == "NO DATA":
            out.append(f"<tr><td>{i+1}</td><td>{anchor}</td><td>{name}</td><td class='grp'>no data</td><td></td><td></td><td></td><td>{H.escape(r['setup'])}</td><td></td><td>—</td></tr>")
            continue
        status = H.escape(r["status"])
        cls = "BREAKOUT" if r["status"] == "CHOCH" else "BREAKOUT_READY" if r["status"] in ("WATCH", "ALIGNED") else ""
        chip = f"<span class='tag {cls}'>{status}</span>" if cls else f"<span class='grp'>{status.lower()}</span>"
        slope = float(r["slope_atr"])
        out.append(f"<tr><td>{i+1}</td><td>{anchor}</td><td>{name}</td><td>{chip}</td>"
                   f"<td class='{r['direction']}'>{r['direction']}</td><td class='num'>{slope:+.2f} ATR</td>"
                   f"<td class='num'>{int(r['side_pct'])}%</td><td>{H.escape(r['setup'])}</td>"
                   f"<td class='num'>{H.escape(r['asof'])}</td><td class='num'>{int(r['score'])}/100</td></tr>")
    out.append("</table></div><div class='kv'>RPT Futures: 30m 65 SMA determines established trend bias; a confirmed countertrend swing sequence and its first CHOCH indicate resumption potential. The 20 SMA alignment raises readiness but is never required for CHOCH. Scores are 0–100: trend 45, countertrend reset 25, recent CHOCH 20, 20 SMA alignment 10. Only completed 30m bars; Yahoo quotes may be delayed. A CHOCH is a chart-review alert, not an automatic trade entry.</div>")
    return "".join(out)


def build_page(t: pd.DataFrame, run_date: str, css: str, note: str = "", session_label: str = "8:00 AM CT morning scan") -> str:
    """Use precisely the old page wrapper and original scanner.report.CSS."""
    n_choch = int((t["status"] == "CHOCH").sum()) if not t.empty else 0
    body = html_section(t) if not t.empty else "<div class='kv'>No futures data this morning.</div>"
    now = dt.datetime.now(dt.timezone.utc)
    return ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>RPT Futures — {run_date}</title><style>{css}</style></head><body><div class='wrap'>"
            f"<h1>RPT Futures</h1><div class='sub'>{H.escape(session_label)} · {run_date} · 4 Topstep micro markets · {n_choch} recent CHOCH · "
            f"<a class='dl' href='futures_watchlist.txt' download='RPT Futures {run_date}.txt'>⬇ Download watchlist</a> · "
            f"<a class='dl' href='index.html'>→ Stock board</a></div>" + body +
            "<h2>How to read it</h2><div class='kv'>Higher scores mean better alignment with the RPT countertrend-failure thesis, not higher proven win rates. "
            "Strong 65 trend + price persistence set the bias; WATCH means countertrend pivots are confirmed; CHOCH means the first countertrend structural level broke on a completed 30m candle. "
            "20 SMA curling back with the trend adds points, but does not gate a CHOCH. Inspect 2m for execution; use the 10m 20 SMA for trade management. "
            "Times and possible feed delays are displayed. Yahoo full-size continuous futures proxy the micro TradingView charts.</div>" +
            (f"<div class='foot'>{H.escape(note)}</div>" if note else "") +
            f"<div class='foot'>RPT Scanner · futures board · generated {now:%Y-%m-%d %H:%M} UTC</div></div></body></html>")


def build_watchlist(t: pd.DataFrame) -> str:
    """Unchanged TradingView comma-separated import format, four markets only."""
    if t.empty:
        return "###FUTURES"
    groups = (("CHOCH", t["status"] == "CHOCH"), ("WATCH", t["status"] == "WATCH"),
              ("TREND", t["status"].isin(["ALIGNED", "TREND"])),
              ("QUIET", t["status"].isin(["CHOP", "NO DATA"])))
    out = []
    for name, mask in groups:
        sub = t[mask]
        if not sub.empty:
            out.append("###" + name)
            out.extend(sub["tv"].tolist())
    return ",".join(out)


def discord_post(webhook, t: pd.DataFrame, page_url: str, run_date: str):
    """Same Discord embed and blue markdown chart links, now all four ranked."""
    import requests
    if not webhook or t.empty or not (t["status"] != "NO DATA").any():
        return False
    when = dt.datetime.strptime(run_date, "%Y-%m-%d").strftime("%b %d").replace(" 0", " ")
    n_choch = int((t["status"] == "CHOCH").sum())
    color = 0xff9800 if n_choch else 0x8a8f98
    lines = []
    for i, r in t.iterrows():
        flag = "🔥" if r["status"] == "CHOCH" else "◐" if r["status"] == "WATCH" else "·"
        side = {"long": "LONG", "short": "SHORT"}.get(r["direction"], "FLAT")
        score = f"{int(r['score'])}/100" if pd.notna(r["score"]) else "no data"
        lines.append(f"### {flag} #{i+1} [{_tick(r)}]({_link(r)}) {r['name']} · {side}\n"
                     f"Score {score} · {r['setup']}" +
                     (f" · Data {r['asof']}" if r["asof"] else ""))
    embed = {"title": f"RPT Futures · {when} · 4 markets ranked", "url": page_url or None,
             "color": color,
             "description": "🔥 recent first CHOCH · ◐ countertrend watch · · other / no data\n" + "\n".join(lines),
             "footer": {"text": "Ranked 0–100 · Yahoo 30m data may be delayed · tap the title for the full board"}}
    resp = requests.post(webhook, json={"embeds": [embed]}, timeout=20)
    resp.raise_for_status()
    return True
