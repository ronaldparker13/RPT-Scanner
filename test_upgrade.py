"""Offline, deterministic checks for futures-only upgrade. Run: python test_upgrade.py"""
import datetime as dt
import os
import sys
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scanner import futures as F

NOW = dt.datetime(2026, 10, 23, 8, 0, tzinfo=ZoneInfo('America/Chicago'))


def sample(n=450, trend=-0.07, seed=9, end=NOW):
    rng = np.random.default_rng(seed)
    ix = pd.date_range(end=pd.Timestamp(end) - pd.Timedelta(minutes=30), periods=n, freq='30min')
    c = np.cumsum(rng.normal(trend, .15, n)) + 120
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o,c)+.08
    lo = np.minimum(o,c)-.08
    return pd.DataFrame({'Open':o,'High':hi,'Low':lo,'Close':c,'Volume':200}, index=ix)


class FuturesUpgradeTests(unittest.TestCase):
    def test_four_micro_markets_only(self):
        self.assertEqual([x[2].split(':')[-1] for x in F.FUTURES], ['MNQ1!', 'MCL1!', 'MGC1!', 'MBT1!'])

    def test_0_to_100_rank(self):
        bars = {s:sample(seed=i+3, trend=(-.07 if i%2==0 else .08)) for i,(s,_,_) in enumerate(F.FUTURES)}
        t = F.in_play_table(bars, now=NOW)
        self.assertEqual(len(t),4)
        self.assertTrue(t['score'].between(0,100).all())
        self.assertEqual(list(t['score']), sorted(t['score'],reverse=True))
        self.assertTrue(set(t['direction']) <= {'long','short','flat','no data'})
        self.assertTrue((t['asof'].str.contains('2026-10-23')).all())

    def test_partial_bar_excluded(self):
        bars={F.FUTURES[0][0]:sample()}
        # Append the still-open 8:00-8:30 candle with implausible price.
        extra=bars[F.FUTURES[0][0]].iloc[[-1]].copy()
        extra.index=pd.DatetimeIndex([pd.Timestamp(NOW)])
        extra.loc[:,'Close']=1_000_000
        bars[F.FUTURES[0][0]]=pd.concat([bars[F.FUTURES[0][0]],extra])
        t=F.in_play_table(bars,now=NOW)
        mnq=t[t['symbol']=='NQ=F'].iloc[0]
        self.assertLess(mnq['price'],500)
        self.assertEqual(mnq['asof'], '2026-10-23 07:30 CT')

    def test_stale_is_no_data_not_zero(self):
        t=F.in_play_table({'NQ=F':sample(end=NOW-dt.timedelta(days=2))},now=NOW)
        self.assertEqual(t.iloc[0]['status'],'NO DATA')
        self.assertTrue(np.isnan(t.iloc[0]['score']))

    def test_unchanged_visual_links_and_score(self):
        bars={s:sample(seed=i+3) for i,(s,_,_) in enumerate(F.FUTURES)}
        t=F.in_play_table(bars,now=NOW)
        html=F.build_page(t,'2026-10-23','/* ORIGINAL CSS */')
        self.assertIn("<div class='wrap'>",html)
        self.assertIn("<div class='scroll'><table>",html)
        self.assertIn("<a class='t' href=",html)
        self.assertIn('MNQ1!',html)
        self.assertIn('/100',html)
        self.assertIn('→ Stock board',html)
        self.assertEqual(html.count('target=\'tradingview\''),4)
        self.assertEqual(F.build_watchlist(t).count('1!'),4)

    def test_discord_has_four_blue_clickable_links_and_rank(self):
        t=F.in_play_table({s:sample(seed=i+3) for i,(s,_,_) in enumerate(F.FUTURES)},now=NOW)
        class Dummy:
            def raise_for_status(self):pass
        with patch('requests.post',return_value=Dummy()) as post:
            self.assertTrue(F.discord_post('https://fake.invalid',t,'https://example.invalid/futures.html','2026-10-23'))
            msg=post.call_args.kwargs['json']['embeds'][0]
            self.assertEqual(msg['description'].count('https://www.tradingview.com/chart/?symbol='),4)
            self.assertIn('0–100',msg['footer']['text'])
            self.assertIn('#1 [',msg['description'])

    def test_choch_without_20_65_cross_requirement(self):
        d = sample(n=450, trend=-.14, seed=11)
        anchor = float(d["Close"].iloc[-34])
        path = [0,1,2,3,2,1,.5,1,2,4,3,2,1.5,2,4,5,4,3,2.8,4,6,5,4,3.5,4.5,6.8,5.6,4.2,2.5,.5,-1,-2.5,-3.5]
        for j, val in enumerate(path):
            k = len(d) - len(path) + j
            value = anchor + val
            prev = anchor + (path[j-1] if j else 0)
            d.iloc[k, d.columns.get_loc('Close')] = value
            d.iloc[k, d.columns.get_loc('Open')] = prev
            d.iloc[k, d.columns.get_loc('High')] = max(prev,value) + .1
            d.iloc[k, d.columns.get_loc('Low')] = min(prev,value) - .1
        t = F.in_play_table({'NQ=F':d}, now=NOW)
        r = t[t['symbol']=='NQ=F'].iloc[0]
        self.assertEqual(r['status'],'CHOCH')
        self.assertEqual(r['direction'],'short')
        # This example's fast MA remains ABOVE the slow MA!
        self.assertGreater(r['ma20'],r['ma65'])
        self.assertTrue(pd.notna(r['signal_time']))
        self.assertEqual(int(r['choch_score']),20)

    def test_mixed_data_discord_safely_includes_no_data(self):
        t=F.in_play_table({'NQ=F':sample()}, now=NOW)
        t=pd.read_json(__import__('io').StringIO(t.to_json(orient='records')),orient='records')
        class Dummy:
            def raise_for_status(self): pass
        with patch('requests.post',return_value=Dummy()) as post:
            self.assertTrue(F.discord_post('https://fake.invalid',t,'https://example.invalid/futures.html','2026-10-23'))
            self.assertEqual(post.call_args.kwargs['json']['embeds'][0]['description'].count('https://www.tradingview.com/chart/?symbol='),4)

    def test_pivots_are_not_available_early(self):
        # Center peak at 3; its causal confirmation is 5.
        h=np.array([1,2,3,7,3,2,1,0,1],dtype=float)
        l=h-0.1
        highs,_=F._pivots(h,l)
        self.assertIn((3,7.,5),highs)
        self.assertTrue(all(confirmed >= i + F.PIVOT_RIGHT for i,_,confirmed in highs))

    def test_chopped_market_not_strong_signal(self):
        d=sample(trend=0,seed=26)
        t=F.in_play_table({'NQ=F':d},now=NOW)
        r=t[t['symbol']=='NQ=F'].iloc[0]
        self.assertLessEqual(r['score'],100)

    def test_stock_runner_guard_only(self):
        from pathlib import Path
        source=Path('run.py').read_text(encoding='utf-8')
        self.assertIn('RPT_STOCK_ONLY',source)
        self.assertIn('if not stock_only:',source)
        self.assertIn('page = build_html(reg, groups, setups, universe, run_date, note, session_label)',source)
        nightly=Path('.github/workflows/nightly.yml').read_text(encoding='utf-8')
        self.assertIn('RPT_STOCK_ONLY: "1"',nightly)
        futures=Path('.github/workflows/futures_morning.yml').read_text(encoding='utf-8')
        self.assertIn('timezone: "America/Chicago"',futures)
        self.assertIn('cron: "0 8 * * 1-5"',futures)


if __name__=='__main__':
    unittest.main(verbosity=2)
