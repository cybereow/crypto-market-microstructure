# Hypothesis: Cross-Exchange IRT Spot Arbitrage (Tabdeal vs Nobitex)

## Claim
Iran's crypto exchanges quote independent USDT/IRT (and other IRT) prices
because capital controls and thin liquidity prevent efficient cross-exchange
arbitrage by large players. Buying USDT/asset cheap on one exchange and
selling it on the other, net of both exchanges' taker fees, should have
a persistent positive edge.

## Why not funding-rate arbitrage (the original plan)
Confirmed from the official Tabdeal API docs: the `fapi` (leverage) product
has no `fundingRate`/`premiumIndex` endpoint and no `FUNDING_FEE` income
type -- its cost mechanism is margin borrow interest
(`GET /api/v1/margin/interestHistory`, types `ON_BORROW`/`PERIODIC`), not a
perpetual funding-rate exchange between longs and shorts. There is nothing
to "collect" by holding a hedged position there. This strategy avoids that
mechanism entirely -- it only needs spot order books, which are public and
confirmed real on both exchanges.

## Data Needed
- Tabdeal spot order book: `GET https://api1.tabdeal.org/r/api/v1/depth?symbol=USDTIRT&limit=N`
- Nobitex spot order book: `GET https://apiv2.nobitex.ir/v3/orderbook/USDTIRT`
- Real taker fee schedule for both exchanges (Tabdeal's 10bps is confirmed
  from docs; Nobitex's in config.yaml is a PLACEHOLDER -- replace with the
  real value before trusting any net-edge number)
- No public historical archive exists for this pair (unlike Binance funding
  rate) -- history must be self-collected via `scripts/scan_irt_arb.py`

## Backtest Parameters
- Min net edge to flag: 50 bps (config.yaml `scan.min_net_edge_bps`) --
  arbitrary starting threshold, not yet validated against real collected data
- Excludes: withdrawal fee, transfer time, transfer-asset price risk
  (e.g. TRX) between the two exchanges, and the two-account operational
  overhead

## Known unknowns (must resolve before risking capital)
- Real Nobitex taker fee (currently a placeholder)
- Actual transfer time + cost between the two exchanges for whatever asset
  is used to move value (TRX suggested in prior research, unverified)
- Whether the sandbox that built this could even reach these APIs is
  unknown -- every attempt from the build environment returned HTTP 503 for
  both api1.tabdeal.org and apiv2.nobitex.ir. This must be re-tested from
  wherever the bot will actually run.

## Success Criteria
- pct_time_viable (from `scripts/analyze_irt_arb.py`, after a few days of
  collection) > some meaningful share of the time, not just rare spikes
- mean_net_edge_bps clearly larger than the untracked costs above (transfer
  fee, transfer time slippage)
