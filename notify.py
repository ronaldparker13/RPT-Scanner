"""Post the Discord cards from docs/latest.json. Runs after the page is pushed, so the links work when the card lands."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scanner.report import discord_post
from scanner import futures as F

d = json.load(open("docs/latest.json"))
page_url = os.environ.get("PAGE_URL", "")
webhook = os.environ.get("DISCORD_WEBHOOK", "")
webhook_fut = os.environ.get("DISCORD_WEBHOOK_FUTURES", "") or webhook
reg = d["regime"]; reg["sectors"] = [tuple(x) for x in d["sectors"]]
setups = pd.DataFrame(d["setups"]); groups = pd.DataFrame(d["groups"]); fut = pd.DataFrame(d["futures"])
discord_post(webhook, reg, setups, groups, page_url, d["date"])
F.discord_post(webhook_fut, fut, (page_url.rstrip("/") + "/futures.html") if page_url else "", d.get("futures_date", d["date"]))
print("posted")
