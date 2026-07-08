# Results: Triangular Arbitrage within Tabdeal

## Collection Period: 2026-07-07 19:15 to 2026-07-08 08:43 (~13.5 hours)

## Results
| Metric | Value |
|--------|-------|
| n_snapshots | 125,782 |
| n_assets_scanned | 100 |
| n_viable_opportunities | 4 |
| pct_time_viable | 0.0% (0.003%) |
| mean_net_edge_bps | -134.01 |
| max_net_edge_bps | 124.01 (ACA) |
| top_assets_by_max_edge | ACA 124.01, AGLD 57.54, BABYSHARK 14.43, BERT 9.55, ARIA 7.45 (rest below the 50bps viable threshold) |

## Verdict
**NOT VIABLE** (as currently scoped: 100 assets, manual/near-real-time reaction)

## Notes
- All 4 viable events happened within the first 6 hours of collection;
  zero new ones appeared in the following 7.5 hours despite the same 100
  assets being scanned at the same frequency -- this isn't just "not
  enough data yet," it's a consistent pattern of extreme rarity.
- Inspected the ACA event in detail: net edge went from -16bps to
  +124bps to -444bps across three consecutive ~40s polls -- the
  opportunity existed for roughly one polling cycle, then reversed hard.
  Not a multi-minute window a human (or even a naively-automated 3-leg
  sequential executor) could reliably act on.
- mean_net_edge_bps is stable around -134bps across both the 6h and
  13.5h checkpoints -- the "crossing 3 bid-ask spreads + fees" cost floor
  is consistent, as expected in a reasonably-behaved market most of the
  time.
- Does this clear $20/month on <$500 capital? No, not as scoped. 4 events
  in 13.5h across 100 assets, each lasting ~1 poll cycle and averaging
  ~60-120bps net when they occur, is not enough occurrence + reaction
  time to build a reliable income stream, especially before accounting
  for real execution risk (partial fill on one of 3 legs).
- Not fully closing the door: scanning the full ~500+ asset universe
  (currently capped at max_symbols=100) might raise the event count
  proportionally, and is cheap to test since the scanner already exists.
  But the "each event lasts ~1 cycle" problem doesn't go away by scanning
  more assets -- it would still require near-instant automated execution,
  which carries real capital risk (an unhedged leg on partial fill) that
  hasn't been built or evaluated.
- Compare to LBank funding-rate arb (research/ -- not yet written up),
  which showed real, moderately-elevated (not extreme) funding rates on
  genuinely liquid symbols (confirmed real order books, e.g. KIOXIAUSDT,
  MUUUSDT) -- a steadier, lower-maintenance candidate than chasing these
  rare triangular blips.
