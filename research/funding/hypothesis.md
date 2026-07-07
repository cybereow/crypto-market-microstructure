# Hypothesis: Funding Rate Arbitrage

## Claim
Long spot + short perpetual on BTC/ETH collects positive funding rate.
Delta-neutral, no rebalancing needed (1x hedge does not drift with price).

## Data Needed
- BTC/USDT and ETH/USDT perpetual funding-rate history (data.binance.vision
  archive, ~2020-present) — NOT the live fapi.binance.com API, which is
  geo-blocked in some environments.
- Binance spot + futures maker/taker fee schedule.
- Daily spot klines, to ground any leverage/liquidation safety check in
  real historical price moves.

## Backtest Parameters
- Capital: $10,000
- Entry: hold the whole period once opened (no threshold-based rebalancing —
  see README for why v0.1's threshold toggle model was replaced)
- Fees: real per-leg bps (see config.yaml), entry + exit only
- Leverage: 1x baseline; test 2x/3x on the short-perp leg only

## Success Criteria
- net_apy_pct (taker fees) > 5% annually
- pct_windows_negative (90d rolling) < 25%
- No leverage level flagged UNSAFE against real historical worst-move data
