"""
Live triangular-arbitrage scanner + (optional) executor across Tabdeal's
own markets: for every asset with both a USDT and an IRT market
(discovered from exchangeInfo, capped at config.yaml's
triangular.max_symbols), compares X/USDT x USDT/IRT against the real
X/IRT price.

DRY-RUN BY DEFAULT: logs every snapshot, prints "VIABLE" when net edge
clears the threshold, and shows what orders WOULD be placed -- it does
NOT touch real money unless you pass --live explicitly, in which case it
also needs TABDEAL_API_KEY / TABDEAL_API_SECRET set as environment
variables. Each opportunity historically lasted about one poll cycle
before reversing (see research/triangular_arbitrage/results.md) -- there
is no human-reaction-time window, execution has to fire immediately on
detection within the same cycle.

Real-money risk if --live is used: if leg 2 or 3 of a 3-leg trade fails
after leg 1 succeeds, the position is left unhedged. This script does not
attempt automatic unwinding -- it stops and prints the failure loudly.

Usage:
    python scripts/scan_triangular.py                          # dry-run, watch only
    python scripts/scan_triangular.py --once                   # one cycle, dry-run
    python scripts/scan_triangular.py --live --capital-irt 2000000 --max-trades-per-day 3
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.exchanges.tabdeal import TabdealClient
from src.storage import Storage
from src.strategies.triangular_arbitrage import TriangularArbitrage
from src.strategies.triangular_executor import TriangularExecutor


def run_cycle(strat: TriangularArbitrage, client: TabdealClient, storage: Storage,
              assets: list[str], executor: TriangularExecutor | None,
              capital_irt: float, trade_count: dict):
    usdt_irt_bid, usdt_irt_ask = client.best_bid_ask(config.ANCHOR_SYMBOL)

    snapshots = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {
            pool.submit(strat.fetch_snapshot, asset, usdt_irt_bid, usdt_irt_ask): asset
            for asset in assets
        }
        for future in as_completed(futures):
            asset = futures[future]
            try:
                snapshots.append(future.result())
            except Exception as e:
                print(f"[{asset}] ERROR: {e}")

    for snapshot in snapshots:
        storage.save_snapshot(snapshot)

    viable = [s for s in snapshots if s["net_edge_bps"] >= config.MIN_NET_EDGE_BPS]
    for s in sorted(viable, key=lambda s: -s["net_edge_bps"]):
        print(f"  VIABLE [{s['base_asset']}] net={s['net_edge_bps']:+.1f}bps  dir={s['direction']}")

        if executor is not None:
            if trade_count["today"] >= trade_count["max_per_day"]:
                print(f"    SKIPPED -- daily trade limit ({trade_count['max_per_day']}) reached")
                continue
            print(f"    executing (live={executor.live})...")
            result = executor.execute(
                s["base_asset"], s["direction"], capital_irt,
                s["x_irt_ask"], s["x_usdt_bid"], s["x_usdt_ask"], s["usdt_irt_ask"],
            )
            trade_count["today"] += 1
            if result.get("error"):
                print(f"    !!! EXECUTION ERROR: {result['error']}")
                if result.get("UNHEDGED_RISK"):
                    print(f"    !!! UNHEDGED POSITION -- only {len(result.get('legs', []))}/3 legs completed. "
                          f"Check your Tabdeal balances manually.")
            else:
                print(f"    done: 3/3 legs completed")

    if snapshots:
        best = max(snapshots, key=lambda s: s["net_edge_bps"])
        print(f"cycle: {len(snapshots)} assets scanned, {len(viable)} viable, "
              f"best={best['base_asset']} ({best['net_edge_bps']:+.1f}bps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one scan cycle and exit")
    ap.add_argument("--live", action="store_true",
                     help="place REAL orders with REAL money. Requires TABDEAL_API_KEY/TABDEAL_API_SECRET env vars.")
    ap.add_argument("--capital-irt", type=float, default=0,
                     help="IRT capital to risk per triggered trade. Required with --live.")
    ap.add_argument("--max-trades-per-day", type=int, default=3,
                     help="circuit breaker: stop executing after this many trades in a run")
    args = ap.parse_args()

    if args.live and args.capital_irt <= 0:
        print("ERROR: --live requires --capital-irt > 0")
        sys.exit(1)

    client = TabdealClient()
    strat = TriangularArbitrage()
    storage = Storage()

    print("Discovering eligible assets and building symbol filter map...")
    all_symbols = client.get_exchange_info()
    symbols_info = {s["symbol"]: s for s in all_symbols}
    assets = strat.eligible_assets()
    print(f"Scanning {len(assets)} assets (capped at max_symbols={config.MAX_SYMBOLS}, "
          f"{config.MAX_WORKERS} parallel requests)")
    print(f"min_net_edge_bps={config.MIN_NET_EDGE_BPS}, poll every {config.POLL_INTERVAL_SECONDS}s")

    executor = None
    if args.live:
        print("\n" + "=" * 60)
        print("!!! LIVE MODE -- THIS WILL PLACE REAL ORDERS WITH REAL MONEY !!!")
        print(f"!!! capital per trade: {args.capital_irt:,.0f} IRT, "
              f"max {args.max_trades_per_day} trades this run !!!")
        print("=" * 60 + "\n")
        executor = TriangularExecutor(symbols_info, live=True)
    else:
        print("Dry-run mode (default) -- no real orders will be placed.\n")
        executor = TriangularExecutor(symbols_info, live=False)

    trade_count = {"today": 0, "max_per_day": args.max_trades_per_day}

    while True:
        t0 = time.time()
        run_cycle(strat, client, storage, assets, executor, args.capital_irt, trade_count)
        print(f"(cycle took {time.time() - t0:.1f}s)\n")
        if args.once:
            break
        time.sleep(config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
