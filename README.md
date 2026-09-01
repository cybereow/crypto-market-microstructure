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
│   ├── paper_test_llm.py               # Same, but an LLM (Claude) must approve each candidate first
│   ├── backtest_llm_gate.py            # LLM gate backtested against already-downloaded history
│   ├── find_pairs.py / backtest_pairs.py  # Statistical-arbitrage pairs trading
│   └── backtest_grid.py                # Grid trading backtest
├── src/
│   ├── labeling.py       # Rule-based primary signals + triple-barrier labeling
│   ├── execution.py      # Maker-order queue fill simulation (§9)
│   ├── market_making.py  # Two-sided quote/fill/inventory simulator with skew (§13)
│   ├── paper_trading.py  # Live shadow paper-trading state machine (§12)
│   ├── llm_decision.py   # LLM (Claude) approve/reject gate for paper_test_llm.py
│   ├── significance.py   # Bootstrap/permutation testing, deflated p-values
│   ├── regime.py / gating.py / calibration.py / novelty.py  # The four ablated ML ideas (§7)
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

**5. LLM-gated shadow paper-trade (experimental, no capital at risk):**
same `vol_breakout` candidate detection as step 4's live paper-trader, but
each candidate must also be approved by Claude (via the Messages API)
before it's logged as a paper trade — every candidate is recorded either
way, so the gate's effect can be compared against the ungated run later.
This is explicitly *not* held to this repo's usual statistical bar (see
[`src/llm_decision.py`](src/llm_decision.py) for why) — it's a probe, not
a validated strategy:

```bash
export ANTHROPIC_API_KEY=...
python scripts/paper_test_llm.py --assets BTC/USDT ETH/USDT SOL/USDT
```

Waiting weeks for that live signal to fire enough times to say anything is
slow. `scripts/backtest_llm_gate.py` replays the same gate against
already-downloaded history in one run instead — one real API call per
historical candidate (costs money and minutes, not weeks), cached to disk
so a re-run never re-pays for a candidate it already decided, and scored
against the ungated pool with the same permutation/bootstrap significance
tests the rest of this repo uses (README section 7's bar, not a lesser
one):

```bash
python scripts/download_klines_vision.py --symbol BTCUSDT --timeframe 4h --start-date 2023-01-01
python scripts/download_klines_vision.py --symbol ETHUSDT --timeframe 4h --start-date 2023-01-01
python scripts/download_klines_vision.py --symbol SOLUSDT --timeframe 4h --start-date 2023-01-01
export ANTHROPIC_API_KEY=...
python scripts/backtest_llm_gate.py \
  --data binance_BTC_USDT_4h.csv binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv \
  --limit 300
```

## Testing

```bash
pytest
```

96 tests covering signal generation, triple-barrier labeling, significance
testing, maker-fill simulation, market-making mechanics, and the live
paper-trading state machine.
