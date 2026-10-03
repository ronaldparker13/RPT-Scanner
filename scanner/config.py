"""RPT Scanner settings. Edit numbers here, nothing else."""

# ---- universe (Ariel's screen) ----
MIN_MARKET_CAP = 1_000_000_000      # $1B
MIN_DOLLAR_VOLUME = 60_000_000      # $60M/day, 20-day average
MIN_PRICE = 10.0
ADR_MIN, ADR_MAX = 3.0, 12.0        # % — 5-8 is the sweet spot; 3-12 keeps the list from starving
HISTORY_DAYS = 320                  # calendar days of daily bars to pull (need 200+ trading days)

# ---- setup rules (same as the RPT Stock Chart Pine) ----
EP_GAP_PCT = 5.0
EP_VOL_MULT = 1.5
HVC_VOL_MULT = 2.0
SECOND_CHANCE_PCT = 2.0             # price within this % of an EP/HVC close from the last 5..30 days
SECOND_CHANCE_MIN_DAYS, SECOND_CHANCE_MAX_DAYS = 3, 30
BREAKOUT_PRIOR_MOVE_PCT = 30.0      # prior move over 60 trading days before the base
BASE_MIN_DAYS, BASE_MAX_DAYS = 10, 40
BASE_NEAR_HIGH_PCT = 5.0            # within this % of the base high = breakout-ready
CHASE_ATR = 4.0                     # > 4 ATR above the 50 = don't chase

# ---- output ----
MAX_SETUPS_PER_SIDE = 12
TOP_GROUPS = 10
REPORT_TITLE = "RPT Scanner"
