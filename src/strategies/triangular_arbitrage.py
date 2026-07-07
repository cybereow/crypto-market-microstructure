"""
Triangular arbitrage within Tabdeal: X/USDT x USDT/IRT vs X/IRT, for every
asset X that has both a USDT and an IRT market. Single exchange, three
legs, no cross-exchange transfer and no dependency on any specific asset
(e.g. PAXG) actually existing -- the set of eligible assets is discovered
live from exchangeInfo, not assumed from unverified third-party claims.

Two round-trip directions starting and ending in IRT:

  Path A (IRT -> X -> USDT -> IRT):
    buy X with IRT (pay X_IRT ask), sell X for USDT (get X_USDT bid),
    sell USDT for IRT (get USDT_IRT bid)
    gross_multiplier = X_USDT_bid * USDT_IRT_bid / X_IRT_ask

  Path B (IRT -> USDT -> X -> IRT):
    buy USDT with IRT (pay USDT_IRT ask), buy X with USDT (pay X_USDT ask),
    sell X for IRT (get X_IRT bid)
    gross_multiplier = X_IRT_bid / (USDT_IRT_ask * X_USDT_ask)

Whichever path has the higher gross multiplier is reported; net multiplier
applies 3 legs of taker fees.
"""
from __future__ import annotations

import time

from .. import config
from ..exchanges.tabdeal import TabdealClient
from ..strategy import Strategy


def discover_assets(symbols: list[dict]) -> list[str]:
    """Base assets with both a TRADING USDT market and a TRADING IRT
    market, excluding USDT/IRT themselves."""
    usdt_bases = set()
    irt_bases = set()
    for s in symbols:
        if s.get("status") != "TRADING":
            continue
        base, quote = s["baseAsset"], s["quoteAsset"]
        if base in ("USDT", "IRT"):
            continue
        if quote == "USDT":
            usdt_bases.add(base)
        elif quote == "IRT":
            irt_bases.add(base)
    return sorted(usdt_bases & irt_bases)


def compute_edge(x_usdt_bid, x_usdt_ask, x_irt_bid, x_irt_ask, usdt_irt_bid, usdt_irt_ask) -> dict:
    gross_a = x_usdt_bid * usdt_irt_bid / x_irt_ask
    gross_b = x_irt_bid / (usdt_irt_ask * x_usdt_ask)

    if gross_a >= gross_b:
        direction, gross_multiplier = "irt_to_x_to_usdt_to_irt", gross_a
    else:
        direction, gross_multiplier = "irt_to_usdt_to_x_to_irt", gross_b

    fee = config.TABDEAL_TAKER_FEE_BPS / 1e4
    net_multiplier = gross_multiplier * (1 - fee) ** 3

    return {
        "direction": direction,
        "gross_edge_bps": (gross_multiplier - 1) * 1e4,
        "net_edge_bps": (net_multiplier - 1) * 1e4,
    }


class TriangularArbitrage(Strategy):
    def __init__(self):
        self.client = TabdealClient()

    def eligible_assets(self) -> list[str]:
        symbols = self.client.get_exchange_info()
        return discover_assets(symbols)[: config.MAX_SYMBOLS]

    def fetch_snapshot(self, base_asset: str, usdt_irt_bid: float, usdt_irt_ask: float) -> dict:
        x_usdt_bid, x_usdt_ask = self.client.best_bid_ask(f"{base_asset}USDT")
        x_irt_bid, x_irt_ask = self.client.best_bid_ask(f"{base_asset}IRT")
        edge = compute_edge(x_usdt_bid, x_usdt_ask, x_irt_bid, x_irt_ask, usdt_irt_bid, usdt_irt_ask)
        return {
            "base_asset": base_asset,
            "timestamp": int(time.time() * 1000),
            "x_usdt_bid": x_usdt_bid, "x_usdt_ask": x_usdt_ask,
            "x_irt_bid": x_irt_bid, "x_irt_ask": x_irt_ask,
            "usdt_irt_bid": usdt_irt_bid, "usdt_irt_ask": usdt_irt_ask,
            **edge,
        }

    def generate_signal(self, data):
        """data = a single fetch_snapshot() dict (live use)."""
        snapshot = data["snapshot"]
        if snapshot["net_edge_bps"] >= config.MIN_NET_EDGE_BPS:
            return [{
                "timestamp": snapshot["timestamp"],
                "base_asset": snapshot["base_asset"],
                "action": snapshot["direction"],
                "net_edge_bps": snapshot["net_edge_bps"],
            }]
        return []

    def backtest(self, data, capital=None, fees=None):
        """data = dict with 'snapshots' = a DataFrame from
        Storage.get_snapshots(), i.e. your own collected history --
        there is no public archive for this. Run scripts/scan_triangular.py
        for a while first to build one."""
        snapshots = data.get("snapshots") if data else None
        if snapshots is None or len(snapshots) == 0:
            return {"error": "no collected snapshots -- run scripts/scan_triangular.py first"}

        viable = snapshots[snapshots["net_edge_bps"] >= config.MIN_NET_EDGE_BPS]
        return {
            "n_snapshots": len(snapshots),
            "n_assets_scanned": snapshots["base_asset"].nunique(),
            "n_viable_opportunities": len(viable),
            "pct_time_viable": round(len(viable) / len(snapshots) * 100, 2),
            "mean_net_edge_bps": round(float(snapshots["net_edge_bps"].mean()), 2),
            "max_net_edge_bps": round(float(snapshots["net_edge_bps"].max()), 2),
            "top_assets_by_max_edge": (
                snapshots.groupby("base_asset")["net_edge_bps"].max()
                .sort_values(ascending=False).head(10).round(2).to_dict()
            ),
        }
