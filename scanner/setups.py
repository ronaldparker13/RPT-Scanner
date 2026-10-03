"""Per-stock indicators and setup detection. Same rules as the RPT Stock Chart Pine.

Setups (long):  EP (gap on volume, held), HVC (high-volume close near the high),
                SECOND_CHANCE (back at a recent EP/HVC close), BREAKOUT_READY (tight base after a
                big move, near the base high), BREAKOUT (closed through the base high on volume).
Setups (short): the mirrors, only below the 200-day.
"""
import numpy as np
import pandas as pd

from . import config as C


def indicators(df: pd.DataFrame) -> dict | None:
    """Return a dict of the numbers the report needs, or None if the stock does not qualify."""
    c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    n = len(df)
    if n < 60:
        return None
    px = float(c.iloc[-1])
    ma10, ma20, ma50 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean()
    ma200 = c.rolling(200).mean() if n >= 200 else pd.Series(np.nan, index=c.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    adr_pct = ((h / l - 1.0).rolling(20).mean() * 100.0)
    avg_vol = v.rolling(20).mean()
    avg_dollar = (v * c).rolling(20).mean()
    hi52 = h.rolling(min(252, n)).max()
    lo52 = l.rolling(min(252, n)).min()

    d = dict(
        price=px, ma10=float(ma10.iloc[-1]), ma20=float(ma20.iloc[-1]), ma50=float(ma50.iloc[-1]),
        ma200=float(ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else np.nan,
        atr=float(atr.iloc[-1]), adr_pct=float(adr_pct.iloc[-1]),
        avg_vol=float(avg_vol.iloc[-2]), avg_dollar=float(avg_dollar.iloc[-1]),
        vol_today=float(v.iloc[-1]), rvol=float(v.iloc[-1] / avg_vol.iloc[-2]) if avg_vol.iloc[-2] > 0 else np.nan,
        chg_pct=float((c.iloc[-1] / c.iloc[-2] - 1) * 100),
        off_52w_high_pct=float((px / hi52.iloc[-1] - 1) * 100), up_52w_low_pct=float((px / lo52.iloc[-1] - 1) * 100),
        ext_atr50=float((px - ma50.iloc[-1]) / atr.iloc[-1]) if atr.iloc[-1] > 0 else np.nan,
        rs_1m=float((c.iloc[-1] / c.iloc[-22] - 1) * 100) if n > 22 else np.nan,
        rs_3m=float((c.iloc[-1] / c.iloc[-64] - 1) * 100) if n > 64 else np.nan,
        new_20d_high=bool(c.iloc[-1] >= c.iloc[-21:].max()),
        above_50=bool(px > ma50.iloc[-1]), above_200=bool(px > ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else True,
        up4=bool(c.iloc[-1] / c.iloc[-2] - 1 >= 0.04), down4=bool(c.iloc[-1] / c.iloc[-2] - 1 <= -0.04),
        low_today=float(l.iloc[-1]), high_today=float(h.iloc[-1]),
    )
    # ------------------------------------------------------------- setups
    setups = []
    prev_c = float(c.iloc[-2])
    rng_pos = (px - d["low_today"]) / (d["high_today"] - d["low_today"]) if d["high_today"] > d["low_today"] else 0.5
    vol_ep = d["rvol"] >= C.EP_VOL_MULT
    vol_hvc = d["rvol"] >= C.HVC_VOL_MULT
    healthy = d["above_200"]
    g = C.EP_GAP_PCT / 100.0

    # EP long / short (today)
    gap_up = float(o.iloc[-1]) >= prev_c * (1 + g)
    holds_up = px >= prev_c * (1 + g) and rng_pos >= 0.3
    gap_dn = float(o.iloc[-1]) <= prev_c * (1 - g)
    holds_dn = px <= prev_c * (1 - g) and rng_pos <= 0.7
    ep_long = vol_ep and gap_up and holds_up
    ep_short = vol_ep and gap_dn and holds_dn
    hvc_long = (not ep_long) and vol_hvc and px > prev_c and rng_pos >= 0.75
    hvc_short = (not ep_short) and vol_hvc and px < prev_c and rng_pos <= 0.25

    if ep_long and healthy:
        setups.append(("EP", "long", px, d["low_today"]))
    if hvc_long and healthy:
        setups.append(("HVC", "long", px, d["low_today"]))
    if ep_short and not healthy:
        setups.append(("EP", "short", px, d["high_today"]))
    if hvc_short and not healthy:
        setups.append(("HVC", "short", px, d["high_today"]))

    # Second chance: a recent EP/HVC close (long) or EP/HVC-short close (short) within SECOND_CHANCE_PCT
    rv_hist = v / avg_vol.shift()
    lvl_long, lvl_short, lvl_days = None, None, None
    for k in range(C.SECOND_CHANCE_MIN_DAYS, min(C.SECOND_CHANCE_MAX_DAYS, n - 2) + 1):
        i = n - 1 - k
        pc, cc, oo, hh, ll = c.iloc[i - 1], c.iloc[i], o.iloc[i], h.iloc[i], l.iloc[i]
        rp = (cc - ll) / (hh - ll) if hh > ll else 0.5
        rvk = rv_hist.iloc[i]
        was_ep = rvk >= C.EP_VOL_MULT and oo >= pc * (1 + g) and cc >= pc * (1 + g) and rp >= 0.3
        was_hvc = rvk >= C.HVC_VOL_MULT and cc > pc and rp >= 0.75
        was_eps = rvk >= C.EP_VOL_MULT and oo <= pc * (1 - g) and cc <= pc * (1 - g) and rp <= 0.7
        was_hvcs = rvk >= C.HVC_VOL_MULT and cc < pc and rp <= 0.25
        if (was_ep or was_hvc) and lvl_long is None:
            lvl_long, lvl_days = float(cc), k
        if (was_eps or was_hvcs) and lvl_short is None:
            lvl_short = float(cc)
    if healthy and lvl_long and abs(px / lvl_long - 1) * 100 <= C.SECOND_CHANCE_PCT and not ep_long and not hvc_long and d["rvol"] < C.HVC_VOL_MULT:
        setups.append(("SECOND_CHANCE", "long", lvl_long, min(d["low_today"], lvl_long * 0.98)))
    if (not healthy) and lvl_short and abs(px / lvl_short - 1) * 100 <= C.SECOND_CHANCE_PCT and not ep_short and not hvc_short and d["rvol"] < C.HVC_VOL_MULT:
        setups.append(("SECOND_CHANCE", "short", lvl_short, max(d["high_today"], lvl_short * 1.02)))

    # Breakout-ready / breakout (long only): prior move then a tight base riding the 10/20
    if healthy and n >= 120:
        best = None
        for base_len in range(C.BASE_MIN_DAYS, C.BASE_MAX_DAYS + 1, 5):
            base = df.iloc[-base_len - 1:-1]
            base_hi, base_lo = float(base["High"].max()), float(base["Low"].min())
            start = n - base_len - 1
            prior = c.iloc[max(0, start - 60):start]
            if len(prior) < 20:
                continue
            prior_move = (c.iloc[start] / prior.min() - 1) * 100
            depth = (base_hi / base_lo - 1) * 100
            tight = depth <= max(12.0, 2.5 * d["adr_pct"])
            riding = bool((base["Close"] > ma20.iloc[-base_len - 1:-1]).mean() >= 0.6)
            if prior_move >= C.BREAKOUT_PRIOR_MOVE_PCT and tight and riding:
                score = prior_move / max(depth, 1)
                if best is None or score > best[0]:
                    best = (score, base_hi, base_lo, base_len, prior_move, depth)
        if best:
            _, base_hi, base_lo, base_len, prior_move, depth = best
            if px > base_hi and d["rvol"] >= 1.5 and rng_pos >= 0.5:
                setups.append(("BREAKOUT", "long", base_hi, d["low_today"]))
            elif px >= base_hi * (1 - C.BASE_NEAR_HIGH_PCT / 100) and px <= base_hi:
                setups.append(("BREAKOUT_READY", "long", base_hi, max(base_lo, px - 1.0 * d["atr"])))
            d.update(base_high=base_hi, base_low=base_lo, base_len=base_len, prior_move=prior_move, base_depth=depth)

    d["setups"] = setups
    return d


def score(setup: tuple, d: dict) -> float:
    """Rank within the list. Higher = look first."""
    kind, side, level, stop = setup
    s = {"EP": 90, "BREAKOUT": 80, "SECOND_CHANCE": 75, "HVC": 65, "BREAKOUT_READY": 55}[kind]
    if side == "long":
        s += min(d.get("rs_1m", 0) or 0, 30) * 0.3
        if d.get("ext_atr50", 0) >= C.CHASE_ATR and kind != "EP":
            s -= 25
        if 5 <= d["adr_pct"] <= 8:
            s += 5
    else:
        s += min(-(d.get("rs_1m", 0) or 0), 30) * 0.3
    s += min(d.get("rvol", 1), 5) * 2
    return round(s, 1)
