"""Independent futures Discord notification, after futures page is published."""
import json
import os

import pandas as pd

from scanner import futures as F


def main():
    with open("docs/futures_latest.json", encoding="utf-8") as f:
        report = json.load(f)
    if report.get("demo"):
        print("DEMO: Discord notification intentionally suppressed")
        return
    webhook = os.getenv("DISCORD_WEBHOOK_FUTURES") or os.getenv("DISCORD_WEBHOOK")
    if not webhook:
        print("No configured futures or fallback Discord webhook; skipped")
        return
    table = pd.DataFrame(report.get("futures", []))
    if table.empty or not (table["status"] != "NO DATA").any():
        print("No fresh futures rows; suppressing a misleading Discord ranking")
        return
    page = os.getenv("PAGE_URL", "").rstrip("/") + "/futures.html"
    F.discord_post(webhook, table, page, report["date"])
    print("Futures-only Discord ranking posted")


if __name__ == "__main__":
    main()
