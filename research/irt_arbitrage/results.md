# Results: Cross-Exchange IRT Spot Arbitrage (Tabdeal vs Nobitex)

## Collection Period: YYYY-MM-DD to YYYY-MM-DD

Run `python scripts/scan_irt_arb.py` for at least a few days, then:
```bash
python scripts/analyze_irt_arb.py
```

## Results
| Metric | Value |
|--------|-------|
| n_snapshots | |
| n_viable_opportunities | |
| pct_time_viable | |
| mean_net_edge_bps | |
| max_net_edge_bps | |
| min_net_edge_bps | |

## Verdict
VIABLE / NOT VIABLE

## Notes
- Confirm the real Nobitex taker fee before trusting net_edge_bps
- Confirm actual transfer cost/time for the asset used to move value between
  exchanges, and re-run the analysis subtracting it
- (anything else learned while the scanner ran)
