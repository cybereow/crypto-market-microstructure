"""
Cross-exchange spot arbitrage: Tabdeal vs Nobitex, USDT/IRT (and any other
shared pair). Pure spot order-book comparison -- no dependency on Tabdeal's
disputed "funding rate" mechanism, since it doesn't have one.

Two accounts (one per exchange) and a way to move value between them (e.g.
transferring USDT via TRX) are required to actually realize the edge; this
module only detects and sizes the opportunity, it does not execute trades.
"""
from __future__ import annotations

import time

from .. import config
from ..exchanges.nobitex import NobitexClient
from ..exchanges.tabdeal import TabdealClient
from ..strategy import Strategy


def compute_spread(tabdeal_bid, tabdeal_ask, nobitex_bid, nobitex_ask) -> dict:
    """Two directions are possible; report whichever is more profitable
    gross, then net it against both exchanges' taker fees."""
    buy_tabdeal_sell_nobitex = (nobitex_bid - tabdeal_ask) / tabdeal_ask * 1e4  # bps
    buy_nobitex_sell_tabdeal = (tabdeal_bid - nobitex_ask) / nobitex_ask * 1e4  # bps

    fee_bps = config.TABDEAL_TAKER_FEE_BPS + config.NOBITEX_TAKER_FEE_BPS

    if buy_tabdeal_sell_nobitex >= buy_nobitex_sell_tabdeal:
        direction = "buy_tabdeal_sell_nobitex"
        gross_edge_bps = buy_tabdeal_sell_nobitex
    else:
        direction = "buy_nobitex_sell_tabdeal"
        gross_edge_bps = buy_nobitex_sell_tabdeal

    return {
        "direction": direction,
        "gross_edge_bps": gross_edge_bps,
        "net_edge_bps": gross_edge_bps - fee_bps,
    }


class IRTArbitrage(Strategy):
    def __init__(self, tabdeal_symbol="USDTIRT", nobitex_symbol="USDTIRT"):
        self.tabdeal_symbol = tabdeal_symbol
        self.nobitex_symbol = nobitex_symbol
        self.tabdeal = TabdealClient()
        self.nobitex = NobitexClient()

    def fetch_snapshot(self) -> dict:
        """One live read from both exchanges. Note: NOT simultaneous --
        two sequential HTTP calls, so there's a small timing gap between
        the two order books. Fine for detection/logging, not for sizing
        a real fill."""
        tabdeal_bid, tabdeal_ask = self.tabdeal.best_bid_ask(self.tabdeal_symbol)
        nobitex_bid, nobitex_ask = self.nobitex.best_bid_ask(self.nobitex_symbol)
        spread = compute_spread(tabdeal_bid, tabdeal_ask, nobitex_bid, nobitex_ask)
        return {
            "pair": self.tabdeal_symbol,
            "timestamp": int(time.time() * 1000),
            "tabdeal_bid": tabdeal_bid,
            "tabdeal_ask": tabdeal_ask,
            "nobitex_bid": nobitex_bid,
            "nobitex_ask": nobitex_ask,
            **spread,
        }

    def generate_signal(self, data):
        """data = a single fetch_snapshot() dict (live use), not historical."""
        snapshot = data.get("snapshot") or self.fetch_snapshot()
        if snapshot["net_edge_bps"] >= config.MIN_NET_EDGE_BPS:
            return [{
                "timestamp": snapshot["timestamp"],
                "action": snapshot["direction"],
                "net_edge_bps": snapshot["net_edge_bps"],
            }]
        return []

    def backtest(self, data, capital=None, fees=None):
        """data = dict with 'snapshots' = a DataFrame from
        Storage.get_snapshots(pair), i.e. your OWN collected history --
        there is no public archive for this pair the way there is for
        Binance funding rates. Run scripts/scan_irt_arb.py for a while
        first to build one."""
        snapshots = data.get("snapshots") if data else None
        if snapshots is None or len(snapshots) == 0:
            return {"error": "no collected snapshots -- run scripts/scan_irt_arb.py first"}

        viable = snapshots[snapshots["net_edge_bps"] >= config.MIN_NET_EDGE_BPS]
        return {
            "n_snapshots": len(snapshots),
            "n_viable_opportunities": len(viable),
            "pct_time_viable": round(len(viable) / len(snapshots) * 100, 2),
            "mean_net_edge_bps": round(float(snapshots["net_edge_bps"].mean()), 2),
            "max_net_edge_bps": round(float(snapshots["net_edge_bps"].max()), 2),
            "min_net_edge_bps": round(float(snapshots["net_edge_bps"].min()), 2),
        }
