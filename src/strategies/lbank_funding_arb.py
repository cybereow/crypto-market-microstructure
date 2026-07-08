"""
Long Spot + Short Perpetual on LBank, collecting positive funding rate.
Same economics as the original Binance-based research (per-leg fees,
leverage-adjusted sizing, liquidation safety check against real historical
price moves, rolling realized-yield range) -- ported to an exchange that's
actually usable and confirmed to have a real funding mechanism (unlike
Tabdeal, which does not).

Two things are approximations here that weren't on Binance:
  1. No historical funding-rate archive is publicly documented for LBank,
     so funding history comes from scripts/scan_lbank_funding.py's own
     self-collected snapshots, not a multi-year bulk download. Annualized
     figures are a time-sampled approximation (mean observed rate x real
     number of settlement periods implied by elapsed time), not a
     reconstruction of actual realized payments.
  2. Trading fees (config.yaml lbank.*_fee_bps) are UNCONFIRMED placeholders
     -- LBank's public docs gave an inconsistent spot fee example and don't
     document perp fees at all.

Daily spot klines ARE real and confirmed working via LBank's own API
(unlike Tabdeal), so the liquidation safety check uses real historical
price data, same as the original Binance version.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..exchanges.lbank import LBankClient
from ..strategy import Strategy


def fee_cost_bps() -> float:
    return config.LBANK_SPOT_TAKER_FEE_BPS + config.LBANK_PERP_TAKER_FEE_BPS


def worst_move_stats(klines: list[list], window_days: int) -> dict:
    """Worst / p95 upside move (entry close -> peak high) over the
    `window_days` FOLLOWING entry, excluding entry day itself. klines =
    [[ts, open, high, low, close, volume], ...] from LBankClient.get_daily_klines."""
    if len(klines) < window_days + 1:
        return {}
    df = pd.DataFrame(klines, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.sort_values("ts").reset_index(drop=True)
    fwd_high = df["high"].shift(-1)[::-1].rolling(window_days, min_periods=window_days).max()[::-1]
    move_pct = ((fwd_high / df["close"] - 1) * 100).dropna()
    if not len(move_pct):
        return {}
    return {
        "window_days": window_days,
        "worst_pct": round(float(move_pct.max()), 1),
        "p95_pct": round(float(move_pct.quantile(0.95)), 1),
        "median_pct": round(float(move_pct.median()), 1),
    }


class LBankFundingArb(Strategy):
    def __init__(self, symbol: str = "BTCUSDT", leverage: float = 1.0):
        self.symbol = symbol
        self.leverage = leverage
        self.client = LBankClient(config.LBANK_PRODUCT_GROUP)

    def fetch_snapshot(self) -> list[dict]:
        """One live read of every perpetual symbol's current funding rate."""
        import time
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
                    "turnover_24h": float(d.get("turnover") or 0),
                })
            except (TypeError, ValueError):
                continue
        return rows

    def generate_signal(self, data):
        """Not used -- this is a cash-flow carry strategy, evaluated
        directly in backtest()/plan_entry(), not via discrete signals."""
        return []

    def backtest(self, data, capital=None, fees=None, min_turnover_24h=20000):
        """data = dict with 'snapshots' = a DataFrame from
        LBankStorage.get_snapshots() -- your own collected history, not a
        bulk archive. Symbols whose mean 24h turnover never clears
        min_turnover_24h are dropped -- LBank lists hundreds of dead/zero-
        volume contracts (empty order book) whose stale funding-rate field
        is meaningless and would otherwise dominate a naive ranking."""
        snapshots = data.get("snapshots") if data else None
        if snapshots is None or len(snapshots) == 0:
            return {"error": "no collected snapshots -- run scripts/scan_lbank_funding.py first"}

        capital = capital or 10000
        per_symbol = {}
        n_dropped_illiquid = 0
        for symbol, g in snapshots.groupby("symbol"):
            if "turnover_24h" in g.columns and g["turnover_24h"].mean() < min_turnover_24h:
                n_dropped_illiquid += 1
                continue
            interval_s = int(g["funding_interval_seconds"].mode().iloc[0])
            payments_per_year = (365 * 24 * 3600) / interval_s
            mean_rate = float(g["funding_rate"].mean())
            days = (g["timestamp"].max() - g["timestamp"].min()) / 1000 / 86400

            notional = capital * self.leverage / (self.leverage + 1)
            gross_annual_income = mean_rate * payments_per_year * notional
            entry_fee = notional * 2 * (fee_cost_bps() / 1e4)
            total_fee = entry_fee * 2  # entry + exit
            net_annual_income = gross_annual_income - total_fee

            per_symbol[symbol] = {
                "n_snapshots": len(g),
                "days_collected": round(days, 2),
                "mean_turnover_24h": round(float(g["turnover_24h"].mean()), 0) if "turnover_24h" in g.columns else None,
                "mean_funding_rate_pct": round(mean_rate * 100, 5),
                "funding_interval_hours": round(interval_s / 3600, 1),
                "naive_annualized_pct": round(mean_rate * payments_per_year * 100, 2),
                "leverage": self.leverage,
                "gross_apy_pct": round(gross_annual_income / capital * 100, 2),
                "net_apy_pct_UNCONFIRMED_fees": round(net_annual_income / capital * 100, 2),
            }

        ranked = dict(sorted(per_symbol.items(), key=lambda kv: kv[1]["naive_annualized_pct"], reverse=True))
        return {
            "fee_bps_per_round_trip_UNCONFIRMED": fee_cost_bps(),
            "n_symbols_liquid": len(ranked),
            "n_symbols_dropped_illiquid": n_dropped_illiquid,
            "per_symbol": ranked,
        }

    def plan_entry(self, capital: float, leverage: float = None, days_history: int = 60):
        """Pre-entry calculator: real liquidation price (from real
        historical daily klines) and a go/no-go safety check. Does not
        place any orders."""
        L = leverage if leverage is not None else self.leverage
        klines = self.client.get_daily_klines(self.symbol.replace("USDT", "_usdt").lower(), days=days_history)
        if not klines:
            return {"error": f"no kline data for {self.symbol}"}

        last_close = float(klines[-1][4])
        notional = capital * L / (L + 1)
        liq_buffer_pct = 100 / L if L > 1 else None
        liq_price = last_close * (1 + 1 / L) if L > 1 else None

        safety = {}
        unsafe_any = False
        if L > 1:
            for window in (1, 7, 30):
                mv = worst_move_stats(klines, window)
                if not mv:
                    continue
                unsafe = mv["worst_pct"] >= liq_buffer_pct
                unsafe_any = unsafe_any or unsafe
                safety[window] = {**mv, "unsafe": unsafe}

        return {
            "symbol": self.symbol,
            "capital": capital,
            "leverage": L,
            "reference_price": last_close,
            "notional_per_leg": round(notional, 2),
            "liquidation_buffer_pct": round(liq_buffer_pct, 1) if liq_buffer_pct else None,
            "liquidation_price": round(liq_price, 2) if liq_price else None,
            "safety_check": safety,
            "unsafe_historically": unsafe_any,
        }
