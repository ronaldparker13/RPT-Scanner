"""Universe and daily bars.

Universe: the public Nasdaq screener download (every US-listed common stock with
sector, industry, market cap and exchange). Bars: Yahoo daily OHLCV via yfinance.
Both are free. To move to a paid feed later (Polygon / EODHD) only this file changes.
"""
import io
import time
import numpy as np
import pandas as pd
import requests

from . import config as C

NASDAQ_SCREENER = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true"
HEADERS = {"User-Agent": "Mozilla/5.0 (RPT Scanner)", "Accept": "application/json"}

EXCHANGE_MAP = {"NASDAQ": "NASDAQ", "NYSE": "NYSE", "AMEX": "AMEX", "NYSE ARCA": "AMEX", "NYSE AMERICAN": "AMEX"}

BENCHMARKS = ["SPY", "QQQ", "IWM"]
SECTOR_ETFS = ["XLK", "XLY", "XLC", "XLI", "XLF", "XLB", "XLE", "XLV", "XLP", "XLU", "XLRE"]


def fetch_universe() -> pd.DataFrame:
    """Return DataFrame[symbol, name, exchange, sector, industry, market_cap]."""
    r = requests.get(NASDAQ_SCREENER, headers=HEADERS, timeout=60)
    r.raise_for_status()
    rows = r.json()["data"]["rows"]
    df = pd.DataFrame(rows)
    df = df.rename(columns={"symbol": "symbol", "name": "name", "marketCap": "market_cap", "sector": "sector", "industry": "industry"})
    df["market_cap"] = pd.to_numeric(df["market_cap"].astype(str).str.replace(",", ""), errors="coerce")
    df = df[df["market_cap"] >= C.MIN_MARKET_CAP]
    # drop units, warrants, preferreds, odd tickers
    df = df[~df["symbol"].str.contains(r"[\^\./]", regex=True)]
    df = df[df["symbol"].str.len() <= 5]
    df["sector"] = df["sector"].replace("", "Unknown").fillna("Unknown")
    df["industry"] = df["industry"].replace("", "Unknown").fillna("Unknown")
    # exchange: the screener download does not carry it; yfinance supplies it later. default NASDAQ.
    df["exchange"] = "NASDAQ"
    return df[["symbol", "name", "exchange", "sector", "industry", "market_cap"]].reset_index(drop=True)


def fetch_bars(symbols, days=C.HISTORY_DAYS, batch=200, pause=1.0) -> dict:
    """Return {symbol: DataFrame[Open, High, Low, Close, Volume]} indexed by date."""
    import yfinance as yf
    out = {}
    period = f"{max(days, 250)}d"
    for i in range(0, len(symbols), batch):
        chunk = list(symbols[i:i + batch])
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                                  auto_adjust=False, threads=True, progress=False)
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
                raw = None
        if raw is None or raw.empty:
            continue
        if len(chunk) == 1:
            raw = {chunk[0]: raw}
        for s in chunk:
            try:
                d = raw[s].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(d) >= 60:
                out[s] = d[["Open", "High", "Low", "Close", "Volume"]].copy()
        time.sleep(pause)
    return out


def fetch_exchanges(symbols) -> dict:
    """Best-effort exchange lookup for TradingView links (NASDAQ / NYSE / AMEX)."""
    import yfinance as yf
    out = {}
    for s in symbols:
        try:
            ex = yf.Ticker(s).fast_info.get("exchange", "")
        except Exception:
            ex = ""
        ex = str(ex).upper()
        out[s] = "NASDAQ" if ex in ("NMS", "NGM", "NCM", "NASDAQ") else "NYSE" if ex in ("NYQ", "NYSE") else "AMEX" if ex in ("PCX", "ASE", "AMEX", "NYSEARCA") else "NASDAQ"
    return out


# ---------------------------------------------------------------- demo mode
def demo_universe(n=60, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    industries = ["Semiconductors", "Software - Infrastructure", "Biotechnology", "Oil & Gas E&P",
                  "Aerospace & Defense", "Consumer Electronics", "Medical Devices", "Specialty Retail"]
    rows = []
    for i in range(n):
        ind = industries[i % len(industries)]
        rows.append(dict(symbol=f"T{i:03d}", name=f"Demo Co {i}", exchange="NASDAQ" if i % 2 else "NYSE",
                         sector="Demo", industry=ind, market_cap=float(rng.uniform(2e9, 2e11))))
    return pd.DataFrame(rows)


def demo_bars(universe: pd.DataFrame, days=300, seed=7) -> dict:
    """Synthetic daily bars with a few planted setups so the report has content offline."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    days = len(idx)
    out = {}
    for k, s in enumerate(universe["symbol"]):
        drift = rng.normal(0.0008, 0.0004)
        vol = rng.uniform(0.015, 0.035)
        r = rng.normal(drift, vol, days)
        kind = k % 6
        if kind == 1:   # EP today
            r[-1] = 0.12
        if kind == 2:   # HVC 6 days ago then drift back = second chance
            r[-7] = 0.07
            r[-6:] = rng.normal(-0.004, 0.01, 6)
        if kind == 3:   # 40% run then 25-day tight base near highs
            r[-85:-25] = rng.normal(0.006, 0.015, 60)
            r[-25:] = rng.normal(0.0, 0.006, 25)
        if kind == 4:   # downtrend + gap down today (short EP)
            r[-120:] = rng.normal(-0.004, 0.02, 120)
            r[-1] = -0.11
        close = 40 * np.exp(np.cumsum(r))
        openp = close / np.exp(r) * np.exp(rng.normal(0, 0.003, days))
        if kind == 1:
            openp[-1] = close[-2] * 1.09
        if kind == 4:
            openp[-1] = close[-2] * 0.92
        high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.008, days)))
        low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.008, days)))
        volume = rng.lognormal(16.5, 0.3, days)
        if kind in (1, 4):
            volume[-1] *= 3.5
        if kind == 2:
            volume[-7] *= 3.0
        out[s] = pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
    # benchmarks
    for b in BENCHMARKS + SECTOR_ETFS:
        r = rng.normal(0.0006, 0.01, days)
        close = 400 * np.exp(np.cumsum(r))
        out[b] = pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close,
                               "Volume": rng.lognormal(18, 0.2, days)}, index=idx)
    return out
