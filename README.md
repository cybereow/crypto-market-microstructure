# Crypto Quant Research Framework

A quantitative research log for crypto trading strategies — not a "profitable
bot." Every strategy here is documented with its real, walk-forward,
cost-adjusted result, including the ones that failed and the one
significant-looking result that got **retracted** once a stricter statistical
test was applied. If you're evaluating this as engineering/research work
rather than an investment product, that discipline is the point.

**Stack:** Python, pandas/numpy, XGBoost, scikit-learn, ccxt, statsmodels/scipy.
Open to contract/freelance data-engineering and quant-research work — reach
out via GitHub.

**Process note:** built through AI pair-programming with Claude Code — from
the initial architecture through the tick-level data engineering and
statistical testing documented in this log. Said plainly, in the same spirit
of not hiding an inconvenient result that runs through the rest of this
project.

## What's actually interesting here

- **Statistical rigor over a good-looking number.** Bootstrap and permutation
  significance testing, deflated p-values for multiple-testing correction,
  purged/embargoed walk-forward cross-validation — built from scratch in
  [`src/significance.py`](src/significance.py) and used to reject the
  project's own best-looking result once it didn't survive scrutiny (Log §7).
- **Tick-level market data engineering.** A chunked, memory-safe processor
  for Binance's raw order-book tick archive (~20M events/day/asset) that
  derives both an order-flow-imbalance signal *and* an OHLC price series
  directly from the same tick stream, with a unit test that locks in
  correctness across chunk/day boundaries
  ([`scripts/download_l2_obi.py`](scripts/download_l2_obi.py)).
- **Meta-labeling ML pipeline** (Lopez de Prado style): triple-barrier
  labeling, an XGBoost classifier trained to answer "will this specific
  trade setup work" rather than "which way will price move," walk-forward
  validated and cost-adjusted
  ([`src/labeling.py`](src/labeling.py), [`scripts/train_meta_ml.py`](scripts/train_meta_ml.py)).
- **Execution-realism modeling**, not just backtest P&L: a maker-order queue
  fill simulator ([`src/execution.py`](src/execution.py)), a two-sided
  market-making simulator with inventory/signal-driven quote skew
  ([`src/market_making.py`](src/market_making.py)), and a live shadow
  paper-trader that checks a backtest's assumptions against real exchange
  quotes without ever risking capital
  ([`src/paper_trading.py`](src/paper_trading.py)).
- **96 passing unit tests** covering the statistical and execution logic, not
  just the happy path.

## Key findings

The full experimental detail — real numbers, tables, exact commands to
reproduce every one of these — lives in **[`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md)**
(§-numbers below refer to it). This is the condensed version.

| # | Finding | Verdict |
|---|---|---|
| §6 | Win rate is set by barrier geometry (`breakeven = 1/(1+pt/sl)`), not model quality — a 91% win rate can still lose money | Structural insight |
| §7 | Four ML "improvement" ideas raised win rate 48.8%→62.4% — but the improvement **failed significance testing** on a larger sample and 8-fold split | **Retracted** |
| §8 | Root cause isolated: transaction cost, not the model — every signal's raw edge (~0.15%) was smaller than the assumed taker cost (0.4%) | Structural insight |
| §8 | At maker-level cost (0.08%) the raw signal becomes significant on its own (p=0.0001) — the first edge that doesn't depend on any ML layer | Real, but cost-fragile |
| §9 | Simulated maker-order queue fills against historical OHLC: edge survives at one a-priori config (p=0.03), but fails after correcting for a small robustness sweep (p=0.115) | Borderline |
| §10 | Hypothesis: daily timeframe has a bigger, cost-resistant per-trade edge | **Rejected** — smaller edge, far fewer candidates |
| §11 | Order-flow-imbalance (OBI) signal at 5-minute resolution, as a genuinely different (non-price) data source | **Null** in both directions |
| §11-b | Same OBI signal re-tested at tick level (5-second bars): strongly significant (p=0.0003), robust to a direction-flip check and a time-split check | Real edge — but ~10x too small to clear even optimistic maker fees |
| §12 | Live shadow paper-test of the maker-fill hypothesis against real exchange quotes, zero capital at risk | Ongoing (signal is rare; needs weeks of data) |
| §13 | Market-making simulator: real top-of-book spread is thinner than the fee itself; volatility-scaled quoting still loses to adverse selection from once-per-bar (stale) requoting | **Rejected** — diagnosed root cause, not a bug |
| §14 | "3-4 signals/day" digest: a fixed, unfitted conviction score ranks a wide candidate pool and spends a daily budget on the best. The ranking has real power (permutation p=0.0002) — but only at a selectivity the daily quota forbids; forcing 4/day dilutes it to a loss | Frequency solved, **profit not** — breakeven at maker cost, loses at taker |
| §15 | Daily **cross-sectional momentum** (long strongest / short weakest, dollar-neutral, 4 signals/day): +32.3%/yr and Sharpe 0.94 at maker cost **after real funding cost**, stable in both out-of-sample halves (0.95 / 1.03); reversal loses everywhere (clean direction check). Pre-registered p=0.026, deflated to 0.24 | **Best result** — real, stable, cost- and funding-clearing edge; strict significance still borderline |
| §16 | Improving §15: a **20-asset, quartile-breadth (M=5, 10 signals/day)** momentum book keeps Sharpe ~1.07 but **halves max drawdown (−36%→−18%)**, stable OOS (0.98/1.14). Vol-targeting was ~neutral; a funding-**carry** factor helped narrow books but hurt broad ones (not robust) | Diversification is the real win (smoother book); the clever factors didn't hold up — and taker is still flat |
| §17 | Attacking the **taker wall** via turnover: an overlapping (Jegadeesh-Titman laddered) **14-day rebalance** cuts turnover 4x (0.41→0.10) and lifts taker Sharpe from ~0.10 to **~0.68 (+13%/yr, −18% DD)** — the first taker-cost-positive book in the log; maker significant (p=0.035). Caught and rejected a lucky single-phase Sharpe 0.89 first | **A real crack in the §8 wall** — positive after full retail cost, though taker significance still borderline (edge concentrated in 2024-25) |
| §18 | The **aggressive** config: variable-in-time leverage (volatility targeting) lifts the §17 book's Sharpe 0.99→**1.34** and, scaled ~2.1x, *backtests* **~55%/yr maker (−29% DD), +37%/yr taker**, positive every year incl. the 2022 bear. Conviction-|z| sizing was tested and **rejected** (lowered Sharpe) | 50%+ is reachable but it's **leverage, not alpha** — matching ~35-40% drawdown, liquidation & maker-fill risk; forward expectation well below the backtest |

## Project layout

```
.
├── scripts/
│   ├── download_data.py                # OHLCV via ccxt (live exchange API)
│   ├── download_klines_vision.py       # Deep OHLCV history from Binance's public archive (spot or futures)
│   ├── download_l2_obi.py              # Order-book imbalance + tick-level OHLC from bookTicker archive
│   ├── download_funding_vision.py      # Funding-rate history (alt-data feature)
│   ├── merge_klines.py                 # Safely merges a fresh CSV into existing history
│   ├── sweep_barrier_geometry.py       # Sweeps pt/sl geometry: where is a high win rate actually profitable
│   ├── train_ml.py / backtest_ml.py    # Direct-direction XGBoost baseline (the ~50% accuracy negative result)
│   ├── train_meta_ml.py                # Pooled meta-labeling model training (triple-barrier)
│   ├── backtest_meta_ml.py             # Single-asset meta-labeling backtest
│   ├── backtest_meta_ml_walkforward.py # The real validator: pooled, purged, walk-forward
│   ├── backtest_cross_sectional.py     # Cross-sectional ranking strategy
│   ├── backtest_maker_fill.py          # Maker-fill queue simulation + significance testing (§9-11)
│   ├── backtest_market_making.py       # Market-making simulator ablation (§13)
│   ├── paper_test_live.py              # Live shadow paper-trader, zero capital risk (§12)
│   ├── find_pairs.py / backtest_pairs.py  # Statistical-arbitrage pairs trading
│   ├── backtest_grid.py                # Grid trading backtest
│   ├── daily_signal_report.py         # "3-4 signals/day" digest: rank a wide pool, spend a daily budget (§14)
│   ├── cross_sectional_report.py      # Daily cross-sectional long/short momentum — the §15 result
│   ├── cross_sectional_v2.py          # §16-17: wider universe, carry ablation, low-turnover overlapping rebalance
│   └── leveraged_book.py              # §18: the aggressive config — variable-in-time (vol-targeted) leverage
├── src/
│   ├── labeling.py       # Rule-based primary signals + triple-barrier labeling
│   ├── execution.py      # Maker-order queue fill simulation (§9)
│   ├── market_making.py  # Two-sided quote/fill/inventory simulator with skew (§13)
│   ├── paper_trading.py  # Live shadow paper-trading state machine (§12)
│   ├── significance.py   # Bootstrap/permutation testing, deflated p-values
│   ├── regime.py / gating.py / calibration.py / novelty.py  # The four ablated ML ideas (§7)
│   ├── daily_signals.py  # Daily-digest engine: candidate pool + fixed conviction ranking + daily budget (§14)
│   ├── cross_sectional_daily.py  # Daily cross-sectional long/short book + turnover-aware backtest (§15)
│   ├── metrics.py        # Trade-level win rate / profit factor
│   └── strategies/       # Pairs trading, direct-ML, and grid strategy signal logic
├── tests/                # 96 unit tests
├── data/                 # Downloaded data and saved models (gitignored)
└── docs/RESEARCH_LOG.md  # Full experimental log, §1-13
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick start

**1. Get deep history** (the public Binance archive gives ~14,400 4h candles
per asset vs. ~720 from a live exchange API — the single highest-leverage
step for anything ML-based):

```bash
for s in BTCUSDT ETHUSDT SOLUSDT LINKUSDT AVAXUSDT DOTUSDT ADAUSDT XRPUSDT DOGEUSDT LTCUSDT ATOMUSDT; do
  python scripts/download_klines_vision.py --symbol $s --timeframe 4h --start-date 2020-01-01
done
```

**2. Train and validate the meta-labeling model** (the pooled, purged,
walk-forward validator is the one to trust — a single-asset backtest is not):

```bash
python scripts/train_meta_ml.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv binance_LINK_USDT_4h.csv \
         binance_AVAX_USDT_4h.csv binance_DOT_USDT_4h.csv binance_ADA_USDT_4h.csv \
         binance_XRP_USDT_4h.csv binance_DOGE_USDT_4h.csv binance_LTC_USDT_4h.csv \
         binance_ATOM_USDT_4h.csv \
  --btc-regime-file binance_BTC_USDT_4h.csv \
  --signal reversion --pt-mult 2.0 --sl-mult 2.0 --target-precision 0.60

python scripts/backtest_meta_ml_walkforward.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv ... \
  --btc-regime-file binance_BTC_USDT_4h.csv \
  --signal reversion --pt-mult 2.0 --sl-mult 2.0 --target-precision 0.60
```

**3. Check whether a high win rate is actually profitable for your geometry:**

```bash
python scripts/sweep_barrier_geometry.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv ... \
  --btc-regime-file binance_BTC_USDT_4h.csv
```

**4. Simulate maker-order fills instead of assuming instant execution:**

```bash
python scripts/backtest_maker_fill.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv ... \
  --signal vol_breakout --pt-mult 2.0 --sl-mult 1.0
```

Other strategies (grid trading, statistical-arbitrage pairs, cross-sectional
ranking, order-flow imbalance, market making, live paper-testing) each have
their own script under `scripts/` — see the project layout above, or
`docs/RESEARCH_LOG.md` for the exact command used in each experiment.

## Testing

```bash
pytest
```

96 tests covering signal generation, triple-barrier labeling, significance
testing, maker-fill simulation, market-making mechanics, and the live
paper-trading state machine.
