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
