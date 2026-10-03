# RPT Scanner

Nightly stock scanner (Ariel Hernandez / Qullamaggie setups) that publishes a one-page report to
GitHub Pages and posts a summary to Discord. Rules mirror the **RPT Stock Chart** Pine indicator.

**Page:** https://ronaldparker13.github.io/rpt-scanner/

## One-time setup (5 minutes)
1. Create the repo `rpt-scanner` on GitHub and upload these files.
2. Settings → Pages → Source: *Deploy from a branch* → Branch `main`, folder `/docs` → Save.
3. Settings → Secrets and variables → Actions → *New repository secret*: name `DISCORD_WEBHOOK`,
   value = your Discord webhook URL. (Skip this and it just won't post.)
4. Actions tab → *nightly scan* → *Run workflow* for the first run. After that it runs every weekday
   after the close by itself.

## Files
- `run.py` — the runner. `python run.py --demo` builds the page from synthetic data (offline test).
- `scanner/config.py` — every threshold (gap %, volume multiples, ADR range, list sizes).
- `scanner/data.py` — universe (Nasdaq screener download) and bars (Yahoo). Swap this file for a paid feed later.
- `scanner/setups.py` — EP / HVC / second chance / breakout-ready / breakout, long and short, scoring.
- `scanner/report.py` — regime strip, group ranking, HTML, Discord.
- `docs/` — the published page (`index.html`), `latest.json`, and a `history/` folder with one page per day.

## Data note
The free feeds (Nasdaq list + Yahoo bars) are for personal use. Once the page is shared with a
community, move `data.py` to a licensed daily feed (EODHD ~$20/mo or Polygon Starter ~$29/mo); the
rest of the code does not change. Never pull TradingView data — tickers link *to* TradingView instead.
