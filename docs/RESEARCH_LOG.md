# Full Research Log

This is the complete, original research log for this project — all 13
experiments with real numbers, tables, and the exact commands used to
reproduce them. For the English summary and quick-start guide, see the main
[README.md](../README.md).

This project is an advanced framework for quantitative strategies and
machine-learning-based algorithms for trading financial markets.

## Implemented strategies

1. **Machine Learning (XGBoost)**: uses advanced technical indicators (MACD,
   RSI, ATR, Bollinger Bands via `pandas-ta`) to train a powerful **XGBoost**
   model to predict the direction of the next candle. Note: predicting the
   next candle's direction directly on real data typically achieves accuracy
   close to 50% (random) — see the Meta-Labeling section below for a more
   credible approach.
2. **Grid Trading**: one of the safest algorithms for range-bound (neutral)
   markets. This bot grids a price range, buying in steps as price falls and
   selling as it rises, to capture continuous profit from small oscillations.
3. **Statistical Arbitrage (Pairs Trading)**: finds pairs of assets that are
   statistically highly correlated (cointegrated) and trades based on the
   Z-score deviation of their spread.
4. **Meta-Labeling (recommended)**: instead of predicting price direction
   directly, a simple rule-based signal (Donchian breakout or RSI reversion)
   generates candidate trades, and the XGBoost model only predicts whether
   this specific trade will hit its profit target or its stop (the
   Triple-Barrier method). Data from all assets is pooled, and validation is
   walk-forward (multiple retrains over successive time windows).

## Setup

Install the required libraries first:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Historical data collection

`ccxt` is used to download OHLCV data.

```bash
# Download daily Bitcoin data from Kraken
python scripts/download_data.py --exchange kraken --symbol BTC/USDT --timeframe 1d --limit 1000

# Download Ethereum data for pairing
python scripts/download_data.py --exchange kraken --symbol ETH/USDT --timeframe 1d --limit 1000
```
*Note: the Binance exchange may be geo-blocked in some regions, which is why Kraken is used as the default for testing.*

### 2. Machine learning strategy (XGBoost)

First train the model on one asset's data. This script builds dozens of
professional features and trains and saves the model:
```bash
python scripts/train_ml.py --data kraken_BTC_USDT_1d.csv
```

Then run the backtest on out-of-sample data:
```bash
python scripts/backtest_ml.py --data kraken_BTC_USDT_1d.csv
```

### 3. Grid trading strategy

To check the grid bot's performance on range-bound markets, run the script
below. Settings like grid count and range width are configurable:
```bash
python scripts/backtest_grid.py --data kraken_BTC_USDT_1d.csv --grids 10 --range-pct 0.2
```

### 4. Statistical arbitrage (pairs trading)

First find pairs with statistical correlation (cointegration):
```bash
python scripts/find_pairs.py
```

Then backtest the strategy on a pair:
```bash
python scripts/backtest_pairs.py --asset1 kraken_BTC_USDT_1d.csv --asset2 kraken_ETH_USDT_1d.csv
```

### 5. Meta-labeling (recommended for a higher win rate)

**Step zero — deep data (the most important step):** the Kraken exchange
only returns about 720 4-hour candles (~4 months) for that timeframe, which
produces only ~60 candidate trades — hopelessly little for training a model.
Binance's public archive (`data.binance.vision`) provides full history:

```bash
for s in BTCUSDT ETHUSDT SOLUSDT LINKUSDT AVAXUSDT DOTUSDT ADAUSDT XRPUSDT DOGEUSDT LTCUSDT ATOMUSDT; do
  python scripts/download_klines_vision.py --symbol $s --timeframe 4h --start-date 2020-01-01
done
```
Result: ~14,400 candles instead of 721 (20x), and ~12,500 candidate trades instead of a few hundred.

**Step zero-b — filling the current-month gap:** the `data.binance.vision`
archive only publishes *completed* months, so it always stops at the end of
last month and is missing up to 31 days of the most recent data (which is
the most important out-of-sample data). If you grabbed a fresh CSV from the
Binance API or dashboard, use this script to safely append it to your
existing data:

```bash
python scripts/merge_klines.py --new-dir /path/to/new_csvs --dry-run   # check first
python scripts/merge_klines.py --new-dir /path/to/new_csvs             # then merge
```

Before writing, this script compares the *overlapping* candles from both
sources, and **rejects** the merge if prices don't match (e.g. spot vs.
perpetual, or a different pair). Blindly concatenating two price series
creates an artificial jump at the seam, which a breakout strategy will
discover as a "signal" — worse than having no data at all.

Train the model (BTC is passed in as market-regime context, not as a traded asset):
```bash
python scripts/train_meta_ml.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv binance_LINK_USDT_4h.csv \
         binance_AVAX_USDT_4h.csv binance_DOT_USDT_4h.csv binance_ADA_USDT_4h.csv \
         binance_XRP_USDT_4h.csv binance_DOGE_USDT_4h.csv binance_LTC_USDT_4h.csv \
         binance_ATOM_USDT_4h.csv \
  --btc-regime-file binance_BTC_USDT_4h.csv \
  --signal reversion --pt-mult 2.0 --sl-mult 2.0 --target-precision 0.60
```

Quick single-asset backtest (small sample, quick sanity check only — the calibrated threshold is read automatically):
```bash
python scripts/backtest_meta_ml.py --data binance_ETH_USDT_4h.csv --btc-regime-file binance_BTC_USDT_4h.csv
```

The real, trustworthy validation (walk-forward — judge results by this output, not the single-asset backtest above). The ablation table shows each component's real contribution:
```bash
python scripts/backtest_meta_ml_walkforward.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv binance_LINK_USDT_4h.csv \
         binance_AVAX_USDT_4h.csv binance_DOT_USDT_4h.csv binance_ADA_USDT_4h.csv \
         binance_XRP_USDT_4h.csv binance_DOGE_USDT_4h.csv binance_LTC_USDT_4h.csv \
         binance_ATOM_USDT_4h.csv \
  --btc-regime-file binance_BTC_USDT_4h.csv \
  --signal reversion --pt-mult 2.0 --sl-mult 2.0 --target-precision 0.60
```

### 6. The "90% win rate" goal — the real, measured result

This is the project's most important finding, and it answers your question directly.

**A 90% win rate is achievable, but nearly worthless.** The reason is a
mathematical identity, not model quality:

```
breakeven_win_rate = 1 / (1 + pt/sl)
```

That is, win rate is fundamentally determined by the **profit-target/stop-loss
geometry**, not by the model. Set your profit target close and your stop far,
and you "buy" a high win rate — but every loss becomes several times the size
of every win.

Real output from `scripts/sweep_barrier_geometry.py` (10 assets, 12,500
trades, purged walk-forward, after a 0.4% fee):

| Signal | pt/sl | Breakeven WR | Win rate (top 10%) | Profit Factor | Verdict |
|---|---|---|---|---|---|
| breakout | 0.25/3.0 | **92.3%** | **91.6%** | **0.56** | **loses money** |
| breakout | 0.33/3.0 | 90.1% | **91.0%** | 0.82 | still unprofitable |
| breakout | 0.5/3.0 | 85.7% | 89.0% | 1.10 | Marginal |
| **reversion** | **2.0/2.0** | **50.0%** | **58.8%** | **1.28** | **Best profitability** |

**A 91% win rate at profit factor 0.56 means blowing up the account.** In
contrast, a 58.8% win rate with a balanced payoff is actually profitable.

To see this table on your own data:
```bash
python scripts/sweep_barrier_geometry.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv ... \
  --btc-regime-file binance_BTC_USDT_4h.csv
```

**So what's the right target?** Instead of "90% win rate," state the target jointly:
> win rate X, with **profit factor > 1.3** after costs, over **at least 200 walk-forward trades**.

### 7. Four win-rate-improvement components, and each one's real contribution

All four ideas are implemented and independently toggleable, so their
contribution is **measured**, not assumed:

| Component | File | What it does |
|---|---|---|
| 1. Explicit market-regime alignment (BTC alignment) | `src/regime.py` | Builds the feature `btc_alignment = sign(side) == sign(BTC_trend)` **explicitly**, instead of hoping the tree discovers this interaction on its own |
| 2. Dynamic confidence threshold | `src/gating.py` | Against BTC's direction → higher threshold; aligned with it → lower |
| 3. Precision-based threshold calibration | `src/calibration.py` | Replaces `scoring='f1'`; no custom-gradient risk |
| 4. OOD filter via leaf-frequency | `src/novelty.py` | Uses `pred_leaf` from the same XGBoost model — no second model |

**Real ablation table** (reversion, pt/sl=2.0/2.0, 10 assets, 3,429
walk-forward trades through 2026-08-30, after costs):

| Configuration | Count | Win rate | PF | Return |
|---|---|---|---|---|
| Primary signal only (no model) | 2778 | 48.8% | 1.16 | +15.5% |
| + Fixed threshold 0.55 (previous method) | 414 | 58.9% | 1.31 | +26.7% |
| + Calibrated threshold (idea 3) | 323 | 57.3% | 1.26 | +18.7% |
| + Dynamic BTC threshold (ideas 1+2) | 87 | 60.9% | 1.86 | +17.6% |
| + OOD filter (idea 4) | 322 | 57.5% | 1.27 | +19.2% |
| **+ Everything (full gate)** | **85** | **62.4%** | **1.96** | +18.7% |

Apparent improvement: **48.8% → 62.4% win rate** and **PF 1.16 → 1.96**.

### Important correction — this improvement failed significance testing

The table above was re-examined with significance testing
(`src/significance.py`). The result contradicts the earlier claim:

| Configuration | n | win% | 95% CI | p (corrected) | Verdict |
|---|---|---|---|---|---|
| Primary signal only | 2778 | 48.8% | [47%,51%] | — | — |
| **Fixed threshold 0.55** | **414** | **58.9%** | **[54%,64%]** | **0.0013** | **Significant** |
| Calibrated threshold (idea 3) | 323 | 57.3% | [52%,63%] | 0.0376 | Significant |
| Dynamic BTC threshold (ideas 1+2) | 87 | 60.9% | [50%,71%] | 0.1612 | **Rejected** |
| OOD filter (idea 4) | 322 | 57.5% | [52%,63%] | 0.0317 | Significant |
| **Full gate** | **85** | **62.4%** | **[51%,73%]** | **0.0931** | **Rejected** |

And on the 1-hour timeframe (4x the data, 12,760 trades), only **one** row
survived:

| Configuration | n | win% | PF | p (corrected) | Verdict |
|---|---|---|---|---|---|
| **Fixed threshold 0.55** | **1002** | **55.4%** | **1.16** | **0.0034** | **Significant** |
| Calibrated threshold | 892 | 51.3% | 1.15 | 0.8327 | Rejected |
| Dynamic BTC threshold | 127 | 59.1% | 1.17 | 0.1568 | Rejected |
| OOD filter | 840 | 50.1% | 1.00 | 0.9896 | Rejected |
| Full gate | 105 | 57.1% | **0.98** | 0.4410 | Rejected |

**Conclusion:** that 62.4% was noise. Three independent pieces of evidence:

1. **The [51%, 73%] confidence interval** — the breakeven point is 50.2%, so
   the interval nearly touches breakeven.
2. **With 8 folds instead of 5**: the full gate collapses to 53.9% and
   **PF 0.94** (unprofitable).
3. **On the 1-hour timeframe**: the full gate yields **PF 0.98** — a loss
   after costs, while its win rate is 57%.

Case 3 is exactly the trap warned about from the start: **a high win rate
and a PF below 1 can coexist.**

**Also, `corr(p_win, label)` dropped to 0.032 on the 1-hour timeframe**
(from 0.082 on 4-hour). More data did not raise predictive power — meaning
the 4-hour signal was also mostly noise, and the 0.082 estimate on the
smaller sample was optimistic.

**The only thing that actually survived:** the fixed threshold of 0.55 — the
simplest option, significant on both timeframes and every fold count, with
1,002 trades (not 85). The three more complex ideas (calibration, BTC
regime, OOD filter) fell apart at larger sample size.

### 8. The real root cause: transaction cost, not the model

Three new primary signals with distinct economic logic were built (`src/labeling.py`):

| Signal | Logic |
|---|---|
| `vol_breakout` | Breakout only when volatility is **compressed** (most breakouts in already-volatile markets are false) |
| `trend_pullback` | Buy a pullback **within** an uptrend (not against the trend) |
| `range_fade` | Fade the band edge only when volatility is **contracting** |

Then 5 signals × 5 barrier geometries = **25 combinations** were run with
significance testing. Result:

```
        signal  pt  sl  base_n  base_wr  base_pf  base_exp   base_p
  vol_breakout 2.0 1.0    3223   36.9%     0.84   -0.251%   0.9995
    range_fade 2.0 2.0    3924   51.6%     0.88   -0.259%   1.0000
      breakout 2.0 2.0    6563   51.7%     0.87   -0.296%   1.0000
     reversion 2.0 1.0    3220   34.0%     0.74   -0.566%   1.0000
```

**All 25 combinations have negative expectancy. `base_p ≈ 1.00` means zero evidence of any edge.**

But breaking out the cost shows the real picture:

| | Value |
|---|---|
| **Raw** edge of the best signal (`vol_breakout` 2:1) | **+0.149%** per trade |
| Assumed cost (0.1% fee + 0.1% slippage, both sides) | **−0.400%** |
| Net result | **−0.251%** |

**19 of 25 combinations have a positive raw edge — but the cost is 2.7x the largest edge.**

Maximum tolerable cost: **0.149% round trip** = 0.074% per side.

**This explains why none of the ML work paid off.** The model was working on
a signal whose raw edge was smaller than the cost. Filtering out bad trades
helps, but when every trade costs 0.4% and the best edge is 0.15%, **no
filter can save it** — this is arithmetic, not model quality.

**Real paths forward from here (in order of payoff):**

1. **Reduce cost** — the only thing that flips the math:
   - **Limit/maker** orders instead of market (Binance: 0.02% maker with BNB
     vs. 0.1% taker) → cost drops to ~0.08%, which is **below** the 0.149%
     edge
   - Longer holding (`--max-holding` larger): cost is fixed, so a larger
     per-trade edge covers it

2. **Larger barrier geometry** — `pt/sl = 4:2` or `6:3`. A bigger move means
   relatively lower cost. Note: trade count drops.

3. **Abandon the 4-hour timeframe for this strategy.** The 0.15% edge isn't
   extractable at 4-hour resolution with retail cost. The daily timeframe
   might have a bigger per-trade edge.

**What NOT to do:** add another feature, model, or knob. The problem isn't the model layer.

#### Experimental confirmation of the cost hypothesis

The `vol_breakout` signal at `pt/sl = 2.0/1.0` on 10 assets, **varying only the cost** (same data, same model, same 3,223 trades, same 36.9% win rate):

| Round-trip cost | Raw signal PF | PF with full gate | Gate return |
|---|---|---|---|
| 0.40% (taker, prior assumption) | **0.90** | **0.99** | −1% |
| 0.20% | 1.03 | 1.12 | +9% |
| **0.08% (maker + low slippage)** | **1.13** | **1.21** | **+15.5%** |

Win rate is **exactly the same** in all three rows (36.9% raw / 40.3%
gated). Cost is the only variable. This is the precise definition of "the
problem isn't in the model layer."

And at maker-level cost, the raw signal becomes significant **on its own**
(corrected p = 0.0001, 3,223 trades) — meaning for the first time in this
project we have a real, statistically valid edge that doesn't depend on the
model either.

An important note on the breakeven point: at `pt/sl = 2:1`, breakeven is
**33%**, not 50%. So a 36.9% win rate is a winning strategy. This is the
same point as section 6 — **win rate is meaningless without the barrier
geometry.**

**Execution warning:** the 0.08% cost assumes **maker** orders get
filled. A breakout with a limit order might not fill at all (adverse
selection) — meaning you miss the winning trades and get filled on the
losing ones. This number is an **optimistic ceiling**, and validating it for
real requires either a queue-fill simulation or a small live test.

Evidence that the ideas actually work (from these same runs):
- **5 of the model's 10 most important features are BTC regime features**
  (`btc_above_sma`, `btc_ret_5_aligned`, `btc_vol`, `btc_alignment`,
  `btc_alignment_strength`) — idea 1 confirmed.
- Trades aligned with BTC: win rate **53.4%** / against it: **48.9%** — idea 2 confirmed.
- Familiar leaf paths: **49.5%** / rare paths: **41.1%** — idea 4 confirmed;
  the model genuinely performs worse in unfamiliar conditions.

**Scientific-honesty warning:** `corr(p_win, label)` is about **0.08**.
That is, the model's ranking power is real but **weak**. With few folds,
part of the improvement above could be luck — which is why the "count"
column is always printed next to win rate. 85 trades is too few for a
definitive conclusion. Note that the full gate cuts trade volume from 2,778
to 85 (~3%) — that's "selectivity," not magic: the model rejects most
opportunities to raise the quality of what's left.

Cross-sectional strategy (ranking assets against each other using the
model's score, long-short between the best and worst):
```bash
python scripts/backtest_cross_sectional.py --data kraken_BTC_USDT_4h.csv kraken_ETH_USDT_4h.csv kraken_SOL_USDT_4h.csv --long-short
```

To add a funding-rate feature (alternative data, independent of price/volume) from Binance's public archive:
```bash
python scripts/download_funding_vision.py --symbol BTCUSDT --start-date 2022-01-01 --end-date 2026-01-01 --out kraken_BTC_USDT_4h_funding.csv
python scripts/train_meta_ml.py --use-funding --data kraken_BTC_USDT_4h.csv ...
```

### 9. Maker order-fill simulation — is the 0.08% edge actually reachable?

Section 8 closed with an execution warning about the 0.08% cost (maker +
low slippage): that number assumes a limit order always fills, while a
breakout with a passive order might never fill at all — which can select
exactly for losing trades over winning ones (adverse selection). This
section replaces that assumption with a real queue simulation, not a
constant.

**Method (`src/execution.py`, `scripts/backtest_maker_fill.py`):** instead
of assuming an instant fill at the signal candle's close, a passive limit
order is placed `0.15 × ATR` better than the signal price (for a long,
below the price — exactly the direction that earns the maker fee, but
requires price to come back). The order stays open for up to 3 more
candles; if no candle's high/low touches that price, the order is cancelled
and **no trade is recorded** — it is not assumed filled. This simulation
runs on OHLC data, not a real order book, so it's an **optimistic ceiling**
(a touched price doesn't guarantee a fill in a real queue).

Real output on the same signal (`vol_breakout`, pt/sl=2.0/1.0), 10 assets,
freshly downloaded data through 2026-07-31 (n here differs from section 8's
3,223 because that number came from a more up-to-date data source — through
2026-08-30; internally consistent here for the taker/maker comparison):

| | n | Win rate | PF | exp/trade | p |
|---|---|---|---|---|---|
| TAKER (prior assumption: instant fill) | 4938 | 36.4% | 0.85 | −0.278% | 1.0000 |
| **MAKER (queue simulation, filled only)** | **4474** | **36.9%** | **1.07** | **+0.112%** | **0.0300** |

Fill rate: **90.6%** (4474 of 4938 candidates, within 3 candles).

**Adverse-selection check** (real win rate of each group, on a taker basis):

| Group | n | Would-be win rate |
|---|---|---|
| Filled | 4474 | 31.5% |
| Unfilled | 464 | **83.8%** |

A gap of **+52.3%** — meaning the exact thing section 8's warning predicted
is real: trades that don't fill were dramatically more likely to have been
winners. The maker edge isn't a complete illusion (after removing the 9.4%
that didn't fill, PF is still >1 and positive), but part of its strength is
lost for exactly the reason warned about.

**Robustness check** (3 other configurations on the same pool, without
changing the data or signal — only the queue parameters):

| queue_timeout | offset_mult | Fill rate | PF | p |
|---|---|---|---|---|
| 1 candle | 0.15 ATR | 84.5% | 1.07 | 0.0295 |
| 6 candles | 0.15 ATR | 93.7% | 1.07 | 0.0165 |
| 3 candles | 0.30 ATR | 80.8% | 1.10 | 0.0055 |

PF is stable between 1.07 and 1.10 across all 4 configurations (default +
these 3) — encouraging. But all 4 p-values being below 0.05 does not mean
the edge was independently confirmed 4 times; per the same
`deflated_pvalue` logic from section 7, checking 4 configurations requires a
multiple-testing correction: with the worst p (0.03) and 4 configurations,
the corrected p = **0.115** — **rejected** at the 0.05 threshold.

**Honest conclusion:** the maker edge at the default configuration (an
a-priori decision, not chosen after seeing the result) is significant
(p=0.03), but this significance is much weaker than the p=0.0001 section 8
reported under the "everything fills" assumption, and once robustness
across configurations is accounted for, it is no longer clearly significant
either. This is exactly what section 8 described as "an optimistic ceiling,
pending a queue simulation or a live test" — the simulation has now been
done, and the result is neither a full confirmation nor a full rejection.
**The real next step isn't more backtesting: it's a small live test against
a real exchange's order queue, because no OHLC-based simulation can
substitute for real order-book data.**

Run on your own data:
```bash
python scripts/backtest_maker_fill.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv ... \
  --signal vol_breakout --pt-mult 2.0 --sl-mult 1.0
```

### 10. Daily timeframe — hypothesis rejected

Section 8 had suggested "abandon the 4-hour timeframe... the daily
timeframe has a bigger per-trade edge" as an alternative path around the
cost problem, without measuring it. This section measures that same
hypothesis with the same signal and geometry (`vol_breakout`, pt/sl=2.0/1.0)
on daily data for the same 10 assets (this project's pooling/purging
infrastructure is already generic — bar width is auto-detected, not
hardcoded to 4 hours).

| Timeframe | n | TAKER PF | MAKER PF (simulated, 92.3% fill) | MAKER p |
|---|---|---|---|---|
| 4-hour | 4938 | 0.85 | 1.07 | 0.0300 |
| **Daily** | **826** | **0.85** | **0.96** | **0.6992** |

**Hypothesis rejected.** The daily per-trade edge is not bigger — it's
essentially the same or slightly worse than 4-hour, and even at the
optimistic maker cost, PF stays below 1 (unprofitable) with p=0.70, nowhere
near significant. Additionally, a 20-candle lookback for Donchian/squeeze on
daily data means candidate signals are extremely scarce: all 6.5 years of
data across 10 assets produced only 826 candidates (vs. 4938 on 4-hour) —
below the 200-walk-forward-trade floor section 6 set as the minimum for any
definitive conclusion, and that's before any walk-forward split.

This experiment tested only one signal/geometry combination (the one with
an edge on 4-hour), not a full sweep. A larger barrier geometry (e.g.,
pt/sl=4:2) or a different signal might do better on daily — but that's a new,
untested hypothesis, not something this measurement supports. What was
measured was: **the simple path of "take the same strategy to a bigger
timeframe" didn't work.**

Run on your own data:
```bash
python scripts/backtest_maker_fill.py \
  --data binance_ETH_USDT_1d.csv binance_SOL_USDT_1d.csv ... \
  --signal vol_breakout --pt-mult 2.0 --sl-mult 1.0
```

### 11. The OBI (order-flow) signal — trying a genuinely different data source, result: no edge

Sections 9 and 10 both circled the same two axes: cost and parameters. But
there's a more fundamental problem — **all five of this project's signals**
(`breakout`, `reversion`, `vol_breakout`, `trend_pullback`, `range_fade`)
**are built from OHLC alone**, the same public data every other quant sees,
who can arbitrage it long before a 4-hour candle even closes. This section
tries a genuinely different data source: order-book imbalance (OBI), from
live order-book data rather than price history.

**Infrastructure:** `scripts/download_l2_obi.py` already existed in the
project but was only used as an auxiliary ML feature. Now `obi_momentum_entries`
(`src/labeling.py`) turns it into an **independent primary signal**: when
OBI crosses above its own rolling upper quantile (dominant buying pressure)
→ long; when it crosses below the lower quantile → short. Reference: Cont,
Kukanov & Stoikov (2014), "The Price Impact of Order Book Events."

**Data:** since Binance's bookTicker archive only exists for futures (not
spot), `download_klines_vision.py` got a `--market futures` flag so prices
pair with the same instrument (not spot) — otherwise the futures order-flow
signal would be tested against a different instrument's price. Data:
BTCUSDT futures, 5-minute, 2023-11-01 to 2024-02-15 (~4.5 months, 5,261 candidates).

Raw result (no cost, pt/sl=2.0/1.0, lookback=288 candles=1 day, max-holding=18 candles=90 minutes):

| Direction | n | Win rate | Breakeven | Raw mean ret | p |
|---|---|---|---|---|---|
| Dominant buying pressure → long (main hypothesis) | 5261 | 34.8% | 33.3% | −0.002% | 0.673 |
| Dominant buying pressure → short (reversed-direction check) | 5261 | 35.2% | 33.3% | +0.001% | 0.398 |

**No edge in either direction.** Win rate in both directions sits right at
the breakeven point (33.3%), and the raw mean return (even before any cost)
is essentially zero — this is a statistical coin flip, not a weak edge.
(With a typical 0.4% taker cost it gets even worse, since the raw average
win is ~0.29% — smaller than the cost itself — meaning on a 5-minute
candle, the typical retail cost is even larger than the size of the price
move; a structural finding in its own right.)

**This direction check (main hypothesis long vs. short) was a principled
test, not another knob to chase p<0.05 with** — the only way to find out
whether the direction hypothesis was wrong or there was simply nothing to
find. Both directions were rejected, so continuing to search by tuning
other parameters (quantile, lookback) would be exactly the p-hacking that
section 9 showed destroys significance.

**A plausible interpretation, not a proven finding:** the order-flow-
imbalance literature predicts the effect on a horizon of **seconds**, not
minutes. `download_l2_obi.py` **sums** the imbalance over the entire
5-minute window (to fix the mean-of-means bias), which could erase exactly
that fast signal before it ever reaches this barrier framework. Testing the
literature's real hypothesis needs sub-minute data and a much shorter
holding period — different infrastructure (HFT-style backtesting), not
another parameter tweak on this same framework.

**Conclusion:** this experiment does not rule out an OBI edge in general; it
only rules out this specific operationalization (5-minute aggregation,
1-day quantile window, 2:1 geometry) on BTC futures over this window. The
code and infrastructure (signal, futures downloader, `--obi-data` wiring)
remain in place for a future tick-level experiment.

Run on your own data:
```bash
python scripts/download_klines_vision.py --symbol BTCUSDT --timeframe 5m --market futures --start-date 2023-11-01 --end-date 2024-02-15
python scripts/download_l2_obi.py --symbol BTCUSDT --start-date 2023-11-01 --end-date 2024-02-15 --timeframe 5m
python scripts/backtest_maker_fill.py \
  --data binance_futures_BTC_USDT_5m.csv --obi-data binance_l2obi_BTCUSDT_5m.csv \
  --signal obi_momentum --lookback 288 --pt-mult 2.0 --sl-mult 1.0 --max-holding 18
```

### 11-b. The tick-level test — the edge was real, we were just looking at the wrong horizon

Section 11 guessed that the OBI signal was erased by 5-minute aggregation,
because the literature describes it on a **seconds** horizon. This section
actually tests that hypothesis, rather than just guessing at it.

**New infrastructure:** no kline archive exists below 1-minute resolution,
so `download_l2_obi.py` now also builds an OHLC candle (based on mid-price)
directly from the bookTicker ticks themselves — no separate download
needed. Sub-minute timeframes (`1s`, `5s`, `10s`, `30s`) were added.

**Data:** BTCUSDT futures, 5-second, 2024-01-15 to 01-28 (14 days, 241,920
candles). Signal: the same `obi_momentum`, with lookback=720 candles (1
hour — tuned to the faster horizon, not 1 day) and pt/sl=2.0/1.0,
max-holding=24 candles (**2 minutes**).

Raw result (no cost):

| | n | Win rate | Breakeven | Raw mean ret | p |
|---|---|---|---|---|---|
| Dominant buying pressure → long (main hypothesis) | 34473 | **47.9%** | 33.3% | +0.0035% | **0.0003** |
| Dominant buying pressure → short (reversed-direction check) | 34473 | 31.7% | 33.3% | ~0.000% | 0.487 |

This time the edge is **real**: the main direction is strongly significant
(p=0.0003 — the strongest number in this entire project), the reversed
direction is a clean null. Splitting the data into two independent 7-day
halves (with no parameter tuning at all) showed both independently
significant (first half: win 47.1%, p=0.0003 — second half: win 48.7%,
p=0.0003) — meaning one odd day isn't driving the whole result.

**But: the edge isn't economical.** The average raw win per trade is only
**0.023%** (average loss −0.014%, average hold ~21 seconds). Even section
9's **optimistic maker cost (0.08%)** — which was barely enough for the
4-hour timeframe — **is more than three times larger than this strategy's
average win.** No ordinary limit/maker order clears that; only a real
market maker with a rebate (not just zero cost, but a **negative** one) and
low-latency (colocated) execution could have any shot at profitability —
completely outside the scope of this project (a retail-accessible
backtester).

**Final conclusion for this line of research:** section 11's guess was
right — the OBI signal is **real and statistically valid**, just on a
horizon (seconds) that 5-minute aggregation was destroying. But the size of
the move at that horizon (0.02%) is so small that no cost structure in this
project (retail taker or even retail maker) can extract it. This differs
from the findings in sections 8-10: there, the edge was close to competitive
with cost (~0.15% vs. 0.08-0.40%); here, the edge is an order of magnitude
smaller than even the smallest conceivable cost. The right home for this
edge is market-making infrastructure, not a directional strategy.

Run on your own data:
```bash
python scripts/download_l2_obi.py --symbol BTCUSDT --start-date 2024-01-15 --end-date 2024-01-28 --timeframe 5s
python scripts/backtest_maker_fill.py \
  --data binance_l2obi_BTCUSDT_5s.csv --obi-data binance_l2obi_BTCUSDT_5s.csv \
  --signal obi_momentum --lookback 720 --pt-mult 2.0 --sl-mult 1.0 --max-holding 24
```

### 12. Live shadow paper-test — zero financial risk, collecting samples

Section 9's result was borderline (p=0.03 at one default configuration, but
p=0.115 after multiple-testing correction across 4 configurations) because
that maker-fill simulation only used historical OHLC — an optimistic
ceiling, not real queue data. This section checks the same hypothesis
against **real live quotes**, without ever sending a real order and without
any financial risk.

**Infrastructure (`src/paper_trading.py`, `scripts/paper_test_live.py`):**
every 4 hours, checks all 10 assets for whether the `vol_breakout` signal
just fired on a freshly closed candle. If it did, it prices a hypothetical
limit order exactly like section 9's `simulate_maker_fills`, and polls the
live price (not a historical 4-hour candle) to check whether it would
actually have filled — more precise than the OHLC simulation because it
polls every time, not just once at the end of each candle.

**Infrastructure note:** Kraken is used instead of Binance directly, because
Binance's live API returns HTTP 451 (geo-restriction) from this
environment — the exact same restriction `scripts/download_data.py` already
documented.

State is stored in `data/paper_trades.csv` (gitignored). Because the
`vol_breakout` signal is rare (~0.2 signals/day/asset, per section 9),
collecting a meaningful sample takes **weeks**; this section will be
updated with the real result once enough samples accumulate.

Manual run:
```bash
python scripts/paper_test_live.py
```

### 13. Market-making — rejected, but the reason it was rejected is instructive

Section 11-b found a real statistical edge in OBI (p=0.0003) that was too
small for a directional trade (0.023% vs. a cost floor even the optimistic
maker fee couldn't clear). The obvious theoretical fix: instead of betting
on direction (which always fights cost), **quote both sides** and capture
the spread — be the market maker, not the taker. This section tests that
idea against real data, rather than just proposing it.

**Infrastructure:** `download_l2_obi.py` now also keeps the **real** last
bid/ask of each bar (`bid_close`/`ask_close`, not just a derived mid) — a
market-making backtest needs the real spread, not an averaged price.
`src/market_making.py` is a bar-by-bar quote/fill/inventory simulator with
two independent skews: inventory (classic risk control — the more one-sided
the position, the more both quotes lean back toward flat) and OBI (quotes
lean in the direction the OBI signal predicts).

**First reality check:** the real average top-of-book spread on BTC futures
is only **0.0002%** of price — an order of magnitude **smaller** than the
0.02% maker cost assumed throughout this project. That means simply
"joining the best price" is a guaranteed loser. So the simulator prices its
quote based on recent volatility instead of the raw spread — trading a
lower fill rate for capturing a spread that's actually larger than the cost.

**Real result** (the same 14 days of 5-second data from section 11-b;
`k_spread` was chosen from cost, not from looking at the result — a
half-spread of 3x the average bar range, i.e. ~2x the round-trip cost):

| Configuration | Round trips | Total PnL |
|---|---|---|
| No skew (naive symmetric) | 6747 | **−144,980** |
| + inventory skew | 7904 | −167,573 |
| + inventory + OBI skew | 7834 | −162,164 |

Disastrous. Diagnosis: inventory sits pinned near its cap (±5) almost the
entire time (std=3.43 — i.e. it practically never stays near zero), and a
forward-return check confirms this is real adverse selection, not a bug:
the average 1-minute return right after each buy fill is more negative than
the market's unconditional average — meaning we get filled buying exactly
when the market is about to turn against us.

**Does simply widening the quote fix it?** Sweeping `k_spread` (3→6→10→20)
reduces the loss monotonically (−145k → −34k → −11.6k → −14k), but it never
turns positive and **floors** around −11 to −14k. At k_spread=10, inventory
skew genuinely helps (halves the loss: −11.6k → −5.2k) — classic risk
control working as expected. But OBI skew makes it slightly worse (−6.4k) —
a real negative finding, reported rather than hidden.

**The real root cause, not a bug:** quotes only update once per bar (5
seconds), based on the previous bar's close — meaning they stay "stale" for
the entire window. In a market with momentum continuation (routine in
crypto), that means orders get filled exactly when price has broken past
the quote level and keeps going — precisely what real market makers avoid
with millisecond-level requoting. **This project's bar-by-bar backtesting
framework — which worked perfectly for the hold-to-barrier directional
strategies elsewhere in this project — is not the right tool for
market-making:** market making is inherently a continuous-time strategy,
not a discrete per-bar decision.

**Conclusion:** "just be the counterparty" is not a simple fix for the cost
problem. Both the directional OBI edge (section 11-b) and market-making
here hit the same wall from different angles: the closer you get to the
faster horizons that actually have an edge, the more you need continuous,
real-time execution infrastructure — something no bar-based backtest
(whether 4-hour or 5-second) can honestly simulate. The real next step for
this line of research, if pursued further, is building continuous
tick-level quoting infrastructure, not more bar-based backtesting.

Run on your own data:
```bash
python scripts/backtest_market_making.py --data binance_l2obi_BTCUSDT_5s.csv --k-spread 10.0
```

### 14. The "3-4 signals a day" request — frequency solved, profitability not

Every section above optimises a single strategy for *edge*. This section
answers a different, product-shaped question that keeps coming up: *"just
give me 3-4 tradeable signals every day."* It is worth answering precisely,
because the naive way to build it is a trap this whole log has been
documenting.

**The trap.** You cannot manufacture "3-4 signals a day" by loosening a
filter until enough trades appear. Section 8 established the binding
constraint — the raw per-trade edge (~0.15%) is smaller than a retail taker
round-trip (~0.40%) — so *firing more often just buys more
negative-expectancy trades faster.* A daily-quota product has to be built the
opposite way: generate a **wide** candidate pool, then spend a **fixed daily
budget** of N slots on the highest-conviction candidates only.

**Design (`src/daily_signals.py`, `scripts/daily_signal_report.py`).**

1. *Candidate pool* — every rule-based primary signal already in
   `labeling.py` (reversion, vol_breakout, trend_pullback), pooled across an
   8-asset 1-hour universe (ETH, SOL, LINK, AVAX, ADA, XRP, DOGE, LTC; BTC as
   regime context, not traded). 2022-01 to 2025-08, **17,500 candidates over
   1,334 days**.
2. *Conviction score* — a **fixed, unfitted** weighting of only the three
   effects earlier sections actually *measured* as real (not the ones §7
   retracted): BTC-regime alignment (§8: aligned 53.4% vs against 48.9%),
   volatility state matched to each signal's own economic rationale, and
   own-asset trend agreement signed by signal family. **No parameter in the
   score is tuned to a backtest number** — that is the §7 failure mode, by
   construction avoided.
3. *Selection* — two methods, reported side by side: **top-N-per-day** (the
   literal digest; a bounded ≤24h same-day ranking lookahead) and a
   fully-causal **frequency-calibrated threshold** (a single constant,
   analogous to §7's surviving fixed-0.55 gate, calibrated to hit the target
   rate — never to trade outcomes). The threshold method is the honest
   economics headline.

**Results** (pt/sl = 2/2, breakeven 50%; maker cost 0.08% is §8's optimistic
floor, taker 0.40% the realistic retail figure):

| Universe / signals | selection | signals/day | win% | maker PF | maker exp/trade | perm_p (ranking) |
|---|---|---|---|---|---|---|
| 3-asset mix | threshold | 4.00 | 49.8% | 0.90 | −0.105% | 0.017 |
| 6-asset mix | threshold | 4.00 | 52.2% | 1.01 | +0.012% | **0.0002** |
| 8-asset mix | threshold | 4.00 | 51.2% | 0.97 | −0.026% | 0.015 |
| 8-asset mix | top-4/day | 3.91 | 49.6% | 0.93 | −0.072% | 0.779 |
| **reversion only** | **top-4/day** | **2.86** | **51.2%** | **1.00** | **−0.000%** | **0.0002** |
| reversion only | threshold (forced 4/day) | 4.00 | 48.0% | 0.82 | −0.232% | 0.974 |

**Two things are true at once, and they are in direct conflict:**

1. **The conviction ranking is real** — when the digest is *allowed to be
   selective*, the permutation test (does this pick beat a random pick of the
   same size from the pool?) is strongly significant: p=0.0002 for reversion
   top-4/day and for the 6-asset threshold. The score is not noise; it does
   pick better-than-average trades.
2. **The daily quota destroys that selectivity.** Reversion alone only fires
   ~2.86 high-conviction times a day, and at that rate it lands *exactly* at
   breakeven (PF 1.00) at the optimistic maker cost. The moment you *force*
   the rate up to a guaranteed 4/day (drop the threshold to 0.455, admitting
   almost the whole pool), the ranking edge vanishes (perm_p 0.97) and PF
   collapses to 0.82. Adding DOGE/LTC to widen the pool had the same diluting
   effect the log has seen before: the 6-asset edge (perm_p 0.0002) faded to
   perm_p 0.015 at 8 assets — apparent edges shrink as the sample grows.

**Verdict.** The *frequency* target is trivially achievable — the tool emits
a clean, ranked 3-4-signal digest every day (`--today`). The *profitability*
target is not met: at realistic retail (taker) cost every configuration
loses (exp −0.32% to −0.55%/trade), and even at the optimistic maker floor
the best the quota allows is **breakeven, not profit** — and net-expectancy>0
is *not* statistically established (bootstrap p well above 0.05 everywhere,
before even correcting for the ~6 configurations examined here). This does
not overturn §8; it reproduces it from the product angle. The one genuinely
positive result — that the unfitted conviction score has real ranking power —
only survives *at the selectivity the quota forbids*. A daily-quota signal
service is therefore honest only as *"here are today's best-ranked setups,
which reach breakeven at maker cost if they fill"* — not as a profit promise.
The §9 adverse-selection risk (do the passive orders that back the maker cost
actually fill, or do you fill the losers and miss the winners?) still applies
and is the first thing a live test would have to settle.

Run on your own data:
```bash
for s in BTCUSDT ETHUSDT SOLUSDT LINKUSDT AVAXUSDT ADAUSDT XRPUSDT DOGEUSDT LTCUSDT; do
  python scripts/download_klines_vision.py --symbol $s --timeframe 1h --start-date 2022-01-01
done

# Full backtest (honest economics: taker AND maker, with significance tests)
python scripts/daily_signal_report.py \
  --data binance_ETH_USDT_1h.csv binance_SOL_USDT_1h.csv binance_LINK_USDT_1h.csv \
         binance_AVAX_USDT_1h.csv binance_ADA_USDT_1h.csv binance_XRP_USDT_1h.csv \
         binance_DOGE_USDT_1h.csv binance_LTC_USDT_1h.csv \
  --btc-regime-file binance_BTC_USDT_1h.csv \
  --signals reversion vol_breakout trend_pullback --pt-mult 2.0 --sl-mult 2.0 --per-day 4

# Today's ranked digest (the product itself)
python scripts/daily_signal_report.py --data <same> --btc-regime-file binance_BTC_USDT_1h.csv --today
```

### 15. Daily cross-sectional momentum — the first strategy that clears cost with a real Sharpe

§14 ended at breakeven because it kept betting the same *absolute, intraday,
single-asset direction* the whole log had already shown is smaller than
cost. A researcher does not stop at breakeven — they change the bet. This
section changes it on the one axis every prior section held fixed: from an
**absolute** bet (does ETH go up?) to a **relative, market-neutral** one
(does ETH outperform the universe?), held for a full day instead of a few
hours.

**Why this escapes the §8 cost wall by construction, before any fitting:**

1. *Relative, not absolute.* Long the strongest M assets, short the weakest
   M, dollar-neutral. The common daily move — the BTC beta that dominates
   every asset's return and that §8's regime features kept re-discovering —
   **cancels** in a long/short book. What is left is cross-sectional
   dispersion, a less-arbitraged signal than the OHLC patterns of §11.
2. *Daily hold -> cost is a small fraction of the move.* A crypto name moves
   ~2-4% a day; an 0.08-0.40% round-trip is a far smaller tax on that than
   on the 2xATR intraday barrier §8 fought. And a book that only *re-ranks*
   daily turns over just the names that changed side — realised turnover for
   the winning config is **0.44/day**, so cost scales with that, not with a
   flat per-name charge.
3. *Exactly N signals a day, structurally.* M longs + M shorts = 2M
   positions every single day. M=2 -> **4 signals/day**, no threshold to
   force. The §14 quota-vs-edge conflict simply does not arise.

**Method (`src/cross_sectional_daily.py`, `scripts/cross_sectional_report.py`):**
resample the same 8-asset 1h universe to daily, rank each day on a
*pre-registered* factor, form the dollar-neutral book, and charge cost on
realised turnover. Two factors were tested as a principled **direction
check** (does the crypto cross-section trend or mean-revert at a daily
horizon?), not a fishing sweep — the same logic §11 used: momentum vs its
exact mirror, short-term reversal.

**Result** (8 assets, 1,338 days, 2022-01 -> 2025-08, M=2 -> 4 signals/day):

| Factor | lookback | maker ann. | maker Sharpe | total | maxDD | turn/day | taker Sharpe |
|---|---|---|---|---|---|---|---|
| **momentum** | **14d** | **+34.5%** | **0.99** | **+152%** | −36% | 0.44 | 0.14 |
| momentum | 30d | +19.2% | 0.59 | +62% | −34% | 0.29 | 0.03 |
| momentum | 7d | +13.7% | 0.39 | +31% | −43% | 0.60 | −0.68 |
| reversal | all | negative | <0 | loss | — | — | negative |

**Two clean, mutually-reinforcing findings:**

1. **The crypto cross-section trends, it does not mean-revert.** Momentum is
   positive and monotonic in lookback (best at 14-30d); reversal *loses at
   every lookback*. That mirror-image split (the identical machinery, sign
   flipped, gives the opposite result) is the §11-b direction-check pattern
   — strong evidence this is structure, not a lucky fit.
2. **Momentum-14d is stable out of sample.** Split the 1,338 days into two
   independent halves with no re-tuning: H1 Sharpe **0.95** (ann +31.6%), H2
   Sharpe **1.03** (ann +37.5%). Same sign, same magnitude, same ~1.0 Sharpe
   in both — one lucky period is not driving it.

**Honest verdict — the best result in this project, with its limits stated:**

- On its own pre-registered terms (14-day cross-sectional momentum is the
  canonical Jegadeesh-Titman lookback, not a discovered one), the full-sample
  bootstrap **p = 0.026** — significant. Correcting conservatively for all 8
  factor×lookback cells examined here deflates it to **p = 0.19**, and
  neither half alone clears 0.05 (0.098, 0.083) — so this is a *real, stable,
  economically-coherent* edge, **not yet a bulletproof one**. An honest
  A-not-quite-B: the direction and stability are convincing; the strict
  significance is borderline.
- It is **cost-fragile the right way up**: profitable and ~Sharpe 1 at maker
  cost, roughly flat at taker cost. Unlike §14's intraday breakout, the maker
  assumption here is *far* more defensible — a daily rebalance can rest
  patient limit orders across the whole day rather than chase a breakout, so
  the §9 adverse-selection risk is much smaller (though not zero).
- **Funding cost — now modelled, and it barely dents the result.** The
  honest worry was that shorting perps to build the short leg would bleed
  funding. `--funding` (real Binance funding history via
  `scripts/download_funding_vision.py`, 8 assets, longs pay / shorts receive,
  3 charges/day) puts a number on it: momentum-14d goes from +34.5% ->
  **+32.3%/yr** and Sharpe 0.99 -> **0.94**. A ~2.2%/yr drag, not a killer —
  precisely because the book is dollar-neutral, so the funding the longs pay
  is largely offset by what the shorts receive. The strategy survives its own
  most-suspected hidden cost.

| momentum-14d, maker | no funding | with funding |
|---|---|---|
| annual return | +34.5% | **+32.3%** |
| Sharpe | 0.99 | **0.94** |
| total (3.7y) | +152% | +137% |

- **The real remaining test** is no longer a backtest: it is a live
  daily-rebalance paper run (like §12's, but at the daily close) to confirm
  the maker fills and to catch anything the historical funding series smooths
  over. That, plus widening the universe, is where this edge goes next.

This is the payoff of not stopping at §14: the same "4 signals a day" product,
rebuilt as a relative bet, goes from breakeven to a stable ~1.0-Sharpe,
cost-clearing strategy — the first in the log to do so.

Run on your own data:
```bash
python scripts/cross_sectional_report.py \
  --data binance_ETH_USDT_1h.csv binance_SOL_USDT_1h.csv binance_LINK_USDT_1h.csv \
         binance_AVAX_USDT_1h.csv binance_ADA_USDT_1h.csv binance_XRP_USDT_1h.csv \
         binance_DOGE_USDT_1h.csv binance_LTC_USDT_1h.csv --m-per-side 2

# Today's long/short book (the product)
python scripts/cross_sectional_report.py --data <same> --today --signal momentum --lookback 14
```

### 16. Improving the §15 book — what actually helped (diversification) and what didn't (carry, vol-targeting)

§15 left a real but bumpy edge: maker Sharpe 0.94 on 8 assets, but a −36%
max drawdown. This section tests three standard, *unfitted* portfolio-
construction levers as an ablation, so each one's real contribution is
measured rather than assumed — and reports the two that did **not** hold up
as honestly as the one that did.

**Method (`src/cross_sectional_daily.py`, `scripts/cross_sectional_v2.py`):**
same daily cross-sectional momentum, now with (1) a wider 20-asset universe,
(2) optional 1/vol position sizing (risk parity within each leg), and (3) an
optional funding-**carry** factor (long low/negative-funding names, short
high-funding ones) z-scored and blended with momentum. Everything graded at
maker AND taker cost with real funding P&L, bootstrap significance, and an
out-of-sample half-split.

**Lever 1 — wider universe, but breadth must scale with it.** The naive move
(keep M=2 longs/shorts, just add assets) *hurt*: top-2 of 20 is the extreme
±10% tail, which is noisier than the ±25% quartile that M=2 selected out of
8. Sharpe fell 0.94 → 0.79. Scaling breadth to hold the quartile (M=5 of 20,
= 10 signals/day) is the fix, and it is the one robust win in this section:

| Book | signals/day | maker Sharpe | **max DD** | OOS halves (Sharpe) | defl_p |
|---|---|---|---|---|---|
| §15: 8 assets, M=2, momentum | 4 | 0.94 | **−36%** | 0.95 / 1.03 | 0.13 |
| **§16: 20 assets, M=5, momentum** | 10 | **1.07** | **−18%** | **0.98 / 1.14** | **0.093** |

Same ~1.0 Sharpe, but the drawdown is **halved** (−36% → −18%) and identical
in both halves — the honest payoff of diversification is a *smoother* book,
not a higher headline number. Full-sample bootstrap p = 0.024 (deflated 0.093
for the handful of variants tried). At taker cost it is still ~flat — the §8
wall is unmoved.

**Lever 2 — vol-targeting: neutral-to-slightly-negative, not a free win.** On
the wide book it moved Sharpe 1.07 → 0.98 (and on 8 assets, 0.94 → 0.97). It
tidies risk contributions but does not reliably raise Sharpe here; reported
as roughly neutral rather than sold as an improvement.

**Lever 3 — carry: helps narrow books, hurts broad ones (so: not robust).**
Blending a funding-carry factor *helped* the narrow 8-asset book (Sharpe 0.94
→ 1.07, drawdown −36% → −28%) but *hurt* the wide 20-asset/M=5 book (1.07 →
0.56). An honest inconsistency: carry is not a dependable additive factor at
this horizon, and it also lifts turnover (0.41 → 0.59), which makes the taker
economics strictly worse. Kept in the code, but **not** part of the
recommended configuration.

**Verdict.** The defensible improvement over §15 is diversification alone: a
20-asset, quartile-breadth (M=5), equal-weight momentum book — **10 signals a
day, Sharpe ~1.07, −18% max drawdown, stable out of sample, at maker cost.**
The two cleverer ideas (vol-targeting, carry) did not survive as reliable
wins across universe sizes, and saying so is the point — §7's lesson was that
the ideas that look additive on one slice often are not. This does not beat
the §8 cost wall (taker still ~flat); it makes the maker-cost book you would
actually run materially smoother, which is the honest, incremental kind of
progress this log is for.

Run on your own data:
```bash
python scripts/cross_sectional_v2.py \
  --data binance_ETH_USDT_1h.csv binance_SOL_USDT_1h.csv binance_LINK_USDT_1h.csv \
         binance_AVAX_USDT_1h.csv binance_ADA_USDT_1h.csv binance_XRP_USDT_1h.csv \
         binance_DOGE_USDT_1h.csv binance_LTC_USDT_1h.csv binance_BNB_USDT_1h.csv \
         binance_DOT_USDT_1h.csv binance_ATOM_USDT_1h.csv binance_UNI_USDT_1h.csv \
         binance_AAVE_USDT_1h.csv binance_ETC_USDT_1h.csv binance_XLM_USDT_1h.csv \
         binance_FIL_USDT_1h.csv binance_TRX_USDT_1h.csv binance_BCH_USDT_1h.csv \
         binance_NEAR_USDT_1h.csv binance_ALGO_USDT_1h.csv \
  --m-per-side 5 --mom-lb 14 --carry-weight 0.3 --ablation
```

### 17. Attacking the taker wall directly — turnover, and the first taker-cost-positive book

Every result from §8 onward shares one verdict at realistic **taker** cost:
flat or losing. §14-16 all clear only the optimistic **maker** floor. This
section attacks the wall head-on. The wall is cost, and cost = per-trade
friction × turnover. §16 could not lower the friction; this section lowers
the **turnover**.

**The lever.** A 14-day momentum signal does not change much day to day, so
rebalancing the book *daily* pays cost far more often than the signal
justifies. Rebalancing every N days cuts turnover ~N-fold. The signal is
14 days old either way — the question is purely whether the cost saved
exceeds the signal decay from holding staler ranks.

**The honesty trap I walked into, and out of.** Rebalancing the whole book
every N days on one fixed schedule looked great — taker Sharpe **0.89** at
N=10. But sweeping the rebalance *phase* (which day of the 10 you start on)
exposed it: across the 10 possible offsets, taker Sharpe ran **0.17 to 0.91**
(mean 0.51). The 0.89 was a lucky phase — exactly the kind of single-number
mirage §7 exists to catch. The fix is the canonical **overlapping
(Jegadeesh-Titman laddered) portfolio**: run all N phases at once and hold
their average, i.e. refresh 1/N of the book each day. That keeps turnover low
AND is phase-independent by construction — no offset is privileged.

**Result** (20 assets, M=5, momentum, overlapping rebalance, real funding,
`overlap_rebalance` in `cross_sectional_daily.py`):

| Rebalance | turnover/day | maker Sharpe | **taker Sharpe** | taker ann. | max DD |
|---|---|---|---|---|---|
| daily (§16) | 0.41 | 1.08 | **0.10** (the §8 wall) | +2.5% | −18% |
| **overlapping 14d** | **0.10** | **1.06** | **0.68** | **+13.1%** | −18% |

**Turnover reduction is the first thing in this entire log to move the taker
number.** Cutting rebalancing from daily to a phase-independent 14-day ladder
drops turnover 4x and lifts the taker Sharpe from ~0 to **~0.68** (+13%/yr),
while barely touching the maker Sharpe (~1.0) — the maker book was never
turnover-limited, the taker book always was. This is the honest mechanism §8
pointed at from the start: "longer holding — cost is fixed, so a larger
per-trade edge covers it."

**The limits, stated plainly:**

- **Time-concentrated.** Split into halves, the taker edge lives in the
  second (2024-25): H1 Sharpe ≈ 0.0, H2 ≈ 1.1. Full-sample taker
  significance is therefore only borderline (bootstrap p ≈ 0.13, deflated
  ~0.33 for the rebalance intervals tried). Maker is significant (p=0.035);
  taker is *positive but not yet proven*. Cross-sectional dispersion was
  simply richer in the back half of the sample.
- **Still an optimistic cost.** 0.40% taker is the honest retail number, but
  it assumes clean fills on 20 names every fortnight; a live test is the only
  way to confirm the +13% survives real execution.

**Where this leaves the project.** §8 framed the taker wall as the thing no
amount of modelling could beat. It turns out one thing beats it — not a
better signal, but trading the same signal *less often*. The result is a
20-asset, quartile-breadth, 14-day-laddered momentum book that is **positive
after full retail taker cost (Sharpe ~0.7, +13%/yr, −18% drawdown)** and
significant at maker cost — the first in this log to clear taker at all. It
is a modest, regime-dependent edge, not a windfall, and its taker
significance is still borderline — but the wall that stood from §8 to §16 now
has a real crack in it, made by turnover, not cleverness.

Run on your own data (add `--rebalance-days 14` to the §16 command):
```bash
python scripts/cross_sectional_v2.py --data <20 1h csvs> \
  --m-per-side 5 --mom-lb 14 --carry-weight 0.0 --rebalance-days 14 --ablation
```

### 18. The aggressive configuration — variable-in-time leverage (volatility targeting)

Every section so far reported an *unlevered* book. This one answers the
question that keeps coming back — "can it make 50%?" — honestly, by writing
down exactly what leverage buys and what it costs. The headline: 50%+ is
reachable on the backtest, but it is bought *entirely* with leverage, and
leverage buys the drawdown in the same proportion.

**The one genuinely-useful refinement: variable leverage in time.** Equal
leverage on every day is wasteful — the book's own volatility swings a lot,
so a fixed multiplier over-risks the wild stretches and under-risks the calm
ones. Volatility targeting (`volatility_scale`) sets each day's exposure from
*trailing* realised vol to hold risk roughly constant — lever up when calm,
down when wild. This is not new edge; it spends the existing edge more evenly.
The effect is real and measured: on the §17 book it lifts Sharpe **0.99 →
~1.30** and, crucially, improves the return-per-unit-drawdown, which is what
actually matters when you then scale up. (Tested and rejected the alternative
reading of "variable size": weighting positions by signal *conviction* |z|
*lowered* Sharpe to 0.84 — bigger bets on stronger signals were just riskier,
not better. Variable in **time** helped; variable by **conviction** did not.)

**What each leverage level delivers** (§17 book, maker execution, vol-targeted):

| Configuration | avg/yr | Sharpe | max DD | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| §17 base, 1x | 20% | 0.99 | −14% | +8% | +29% | +23% | +6% |
| vol-targeted (~1.5x avg) | 33% | **1.34** | −21% | +18% | +60% | +33% | +16% |
| vol-targeted + 1.5x extra (~2.1x avg) | **55%** | 1.28 | **−29%** | +18% | +91% | +47% | +22% |

At **taker** cost the same overlay gives **+37%/yr** (Sharpe 0.91, −36% DD) —
still positive every calendar year, including the 2022 bear, because the book
is market-neutral. That "positive in the bear too" is the one thing that is
structural rather than lucky: a long/short book profits from *dispersion*, not
direction.

**The four caveats, stated as loudly as the returns — this is the aggressive
end of the repo, not its recommendation:**

1. **Maker-dependent.** The 55% needs passive limit fills. At taker cost the
   honest number is ~37% (and to force 50% at taker you need ~4x leverage and
   a ~−55% drawdown). §9's adverse-selection risk is exactly the thing that
   decides which column you actually live in.
2. **Drawdown is the price.** −29% in-sample typically means ~−40% live. A
   50%-return configuration is a ~−35% to −40% drawdown configuration; the two
   are the same dial.
3. **Liquidation risk.** ~2x leverage into a −30% drawdown sits near margin
   limits; a sharp gap can liquidate the book before it recovers — the exact
   mechanism that ended 3AC, Alameda, and most levered crypto funds.
4. **Regime- and tuning-dependent.** The +91% of 2023 was an exceptional
   dispersion year that will not repeat on demand, and the vol-target
   parameters were lightly tuned in-sample. The honest forward expectation is
   *below* the backtest — plausibly ~25-40% in a normal regime with maker
   fills, not 50%.

**Bottom line for the whole §14-18 arc.** Starting from "give me 3-4
profitable signals a day," the honest end state is: a 20-asset, market-neutral,
daily cross-sectional momentum book (10 signals/day), rebalanced as a low-
turnover overlapping ladder (§17) and run at variable, vol-targeted leverage
(§18). Unlevered and honest it is ~15-20%/yr at maker cost; pushed to the
aggressive end it *backtests* near 50% — but that number is leverage, not
alpha, and it carries a matching ~35-40% drawdown and real ruin risk. There is
no configuration in this repo that reaches 50%+ *without* that leverage and
that risk, and the project's whole point is to say so plainly rather than sell
the big number on its own.

Run on your own data:
```bash
python scripts/leveraged_book.py --data <20 1h csvs> --extra-leverage 1.5
```

## Project structure

```
trading-bot/
├── scripts/
│   ├── download_data.py               # Download OHLCV data via CCXT
│   ├── download_klines_vision.py      # Download deep OHLCV history from Binance's archive (20x Kraken)
│   ├── merge_klines.py                # Safely merge a fresh CSV into existing data (checks seam consistency)
│   ├── sweep_barrier_geometry.py      # Sweep pt/sl geometry: where a high win rate is actually profitable
│   ├── download_funding_vision.py     # Download funding-rate history from Binance's public archive
│   ├── download_l2_obi.py             # Download order-book imbalance (futures bookTicker) — the OBI signal, §11
│   ├── find_pairs.py                  # Discover cointegrated pairs (Cointegration Test)
│   ├── backtest_pairs.py              # Statistical-arbitrage backtest
│   ├── train_ml.py                    # Extract advanced features with pandas-ta and train XGBoost
│   ├── backtest_ml.py                 # Backtest the XGBoost strategy (direct direction prediction)
│   ├── train_meta_ml.py               # Pooled meta-labeling model training (Triple-Barrier)
│   ├── backtest_meta_ml.py            # Single-asset meta-labeling backtest
│   ├── backtest_meta_ml_walkforward.py# Walk-forward validation of the meta-labeling model
│   ├── backtest_cross_sectional.py    # Cross-sectional ranking strategy across assets
│   ├── backtest_maker_fill.py         # Maker order-fill simulation (§9) and daily-timeframe experiment (§10)
│   ├── paper_test_live.py             # Live shadow maker-fill simulation against real quotes, zero risk (§12)
│   ├── backtest_market_making.py      # Market-making simulator ablation (§13)
│   └── backtest_grid.py               # Grid Trading bot backtest
├── src/
│   ├── config.py               # Project settings and paths
│   ├── metrics.py              # Standard win-rate calculation (per closed trade)
│   ├── labeling.py             # Rule-based primary signals and Triple-Barrier labeling
│   ├── execution.py            # Maker order-queue simulation: optimistic OHLC-based fills (§9)
│   ├── paper_trading.py        # Live shadow-simulation state machine (§12)
│   ├── market_making.py        # Two-sided quote/fill/inventory simulator with skew (§13)
│   ├── regime.py               # Idea 1: market-regime context and the btc_alignment feature
│   ├── calibration.py          # Idea 3: precision-based threshold calibration (instead of F1)
│   ├── novelty.py              # Idea 4: OOD detection via leaf-frequency (no second model)
│   ├── gating.py               # Idea 2: decision layer with a dynamic threshold
│   ├── significance.py         # Significance testing: exact CI + permutation test + multiple-testing correction
│   └── strategies/
│       ├── pairs_trading.py   # Pairs Trading signal logic (Z-Score)
│       ├── ml_strategy.py     # ML model signal logic
│       └── grid_trading.py    # Grid trading and step-trade logic
├── data/                      # Downloaded data and saved models
├── config.yaml                # Configuration file
└── requirements.txt
```
