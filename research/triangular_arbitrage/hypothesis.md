# Hypothesis: Triangular Arbitrage within Tabdeal

## Claim
Tabdeal quotes X/USDT, X/IRT, and USDT/IRT markets independently for many
assets. Because IRT liquidity is thin and there's no automated market-maker
forcing these three prices to stay consistent, the "synthetic" X/IRT price
implied by X/USDT * USDT/IRT will sometimes diverge from the real quoted
X/IRT price by more than the round-trip cost of exploiting it (3 taker
legs).

## Why this, not funding-rate arb or cross-exchange IRT arb
- Funding-rate arb: rejected -- Tabdeal's `fapi` has no
  `fundingRate`/`premiumIndex` endpoint and no `FUNDING_FEE` income type
  (confirmed from docs.tabdeal.org); its cost mechanism is margin borrow
  interest, not a perpetual funding exchange.
- Cross-exchange IRT arb (Tabdeal vs Nobitex): rejected by the user --
  no second exchange account wanted.
- This strategy needs only Tabdeal, only public (no-auth) endpoints, and
  doesn't bet on any specific asset (e.g. PAXG) actually being listed --
  eligible assets are discovered live via `exchangeInfo`, since PAXG is
  never mentioned anywhere in Tabdeal's real API docs and its existence
  there is unverified.

## Data Needed
- `GET /api/v1/exchangeInfo` -- discover every base asset with both a
  TRADING USDT market and a TRADING IRT market
- `GET /api/v1/depth?symbol=...` for X/USDT, X/IRT, and USDT/IRT
- Real Tabdeal taker fee (10bps, confirmed from docs) -- 3 legs per round trip
- No public historical archive exists for this -- history must be
  self-collected via `scripts/scan_triangular.py`

## Backtest Parameters
- `max_symbols`: 30 per cycle by default (config.yaml) -- avoids hammering
  the API with hundreds of depth calls before the real rate limit is known
- `min_net_edge_bps`: 50 -- arbitrary starting threshold, not yet validated

## Known unknowns (must resolve before risking capital)
- Whether the build sandbox's inability to reach api1.tabdeal.org (HTTP
  503 on every attempt) is specific to that sandbox or a broader issue --
  must be re-tested from the actual deployment machine
- Real Tabdeal API rate limits (undocumented in what was provided)
- Whether opportunities, if real, last long enough to act on manually or
  need automated execution from the start
- Actual liquidity at each leg -- an edge that exists on a 5-row order book
  snapshot may not survive placing a real order big enough to matter

## Target
User wants at least $20/month starting from < $500 capital (~4-6%/month).
This is a high bar for a low-risk arbitrage strategy -- scanning many
assets simultaneously (rather than betting on one pair) is the main lever
to get there, but this is unproven until real data is collected.

## Success Criteria
- `pct_time_viable` and `mean_net_edge_bps` (from `analyze_triangular.py`,
  after a few days of collection) imply enough executable opportunities
  per month, at realistic size, to clear $20/month after all 3 legs' fees
