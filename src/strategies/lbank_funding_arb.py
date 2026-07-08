"""
Long Spot + Short Perpetual on LBank, collecting positive funding rate.
Same delta-neutral idea as the original (Binance-based) research, but on
an exchange that's actually usable and confirmed to have a real funding
mechanism (unlike Tabdeal).

No historical funding-rate archive is publicly documented for LBank, so
this evaluates a self-collected snapshot dataset (scripts/scan_lbank_funding.py)
instead of a bulk download -- the naive_annualized_pct figures are a
time-sampled approximation (mean observed rate x periods/year for that
symbol's own funding interval), not a reconstruction of actual realized
payments.
"""
from __future__ import annotations

import time

import pandas as pd

from .. import config
from ..exchanges.lbank import LBankClient
from ..strategy import Strategy


class LBankFundingArb(Strategy):
    def __init__(self):
        self.client = LBankClient(config.LBANK_PRODUCT_GROUP)

    def fetch_snapshot(self) -> list[dict]:
        """One live read of every perpetual symbol's current funding rate."""
        data = self.client.get_perp_market_data()
        ts = int(time.time() * 1000)
        rows = []
        for d in data:
            if "fundingRate" not in d:
                continue
            try:
                rows.append({
                    "symbol": d["symbol"],
                    "timestamp": ts,
                    "funding_rate": float(d["fundingRate"]),
                    "funding_interval_seconds": int(d.get("positionFeeTime") or 28800),
                    "marked_price": float(d.get("markedPrice") or 0),
                })
            except (TypeError, ValueError):
                continue
        return rows

    def generate_signal(self, data):
        """Not used -- this is a cash-flow carry strategy, evaluated
        directly in backtest(), not via discrete buy/sell signals."""
        return []

    def backtest(self, data, capital=None, fees=None):
        """data = dict with 'snapshots' = a DataFrame from
        LBankStorage.get_snapshots() -- your own collected history."""
        snapshots = data.get("snapshots") if data else None
        if snapshots is None or len(snapshots) == 0:
            return {"error": "no collected snapshots -- run scripts/scan_lbank_funding.py first"}

        fee_bps = config.LBANK_SPOT_TAKER_FEE_BPS + config.LBANK_PERP_TAKER_FEE_BPS
        per_symbol = {}
        for symbol, g in snapshots.groupby("symbol"):
            interval_s = int(g["funding_interval_seconds"].mode().iloc[0])
            payments_per_year = (365 * 24 * 3600) / interval_s
            mean_rate = float(g["funding_rate"].mean())
            days = (g["timestamp"].max() - g["timestamp"].min()) / 1000 / 86400
            per_symbol[symbol] = {
                "n_snapshots": len(g),
                "days_collected": round(days, 2),
                "mean_funding_rate_pct": round(mean_rate * 100, 5),
                "funding_interval_hours": round(interval_s / 3600, 1),
                "naive_annualized_pct": round(mean_rate * payments_per_year * 100, 2),
            }

        ranked = dict(sorted(per_symbol.items(), key=lambda kv: kv[1]["naive_annualized_pct"], reverse=True))
        return {
            "fee_bps_per_round_trip_UNCONFIRMED": fee_bps,
            "n_symbols": len(ranked),
            "per_symbol": ranked,
        }
