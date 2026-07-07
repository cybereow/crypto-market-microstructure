# Results: Triangular Arbitrage within Tabdeal

## Collection Period: YYYY-MM-DD to YYYY-MM-DD

Run `python scripts/scan_triangular.py` for at least a few days, then:
```bash
python scripts/analyze_triangular.py
```

## Results
| Metric | Value |
|--------|-------|
| n_snapshots | |
| n_assets_scanned | |
| n_viable_opportunities | |
| pct_time_viable | |
| mean_net_edge_bps | |
| max_net_edge_bps | |
| top_assets_by_max_edge | |

## Verdict
VIABLE / NOT VIABLE

## Notes
- Does this realistically clear $20/month given starting capital?
- How long do viable windows last -- long enough to act on manually?
- Any assets consistently at the top of `top_assets_by_max_edge` worth
  investigating for a structural (not just noise) reason?
