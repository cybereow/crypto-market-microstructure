"""
Long Spot + Short Perpetual funding-rate carry.

Sign convention: always short the perpetual (long the equivalent spot), so
a POSITIVE funding_rate means we RECEIVE payment (longs pay shorts); a
NEGATIVE rate means we PAY. A 1x spot-vs-perp hedge does not drift with
price, so a matched pair needs no periodic rebalancing -- it opens once,
holds, and closes once.

All historical numbers here come from the real Binance Vision archives
(binance_funding.py, binance_klines.py), not estimates.
"""
from __future__ import annotations

import pandas as pd

from .. import binance_funding as bf
from .. import binance_klines as bk
from .. import config
from ..strategy import Strategy


def fee_cost_bps(fee_mode: str) -> float:
    spot = config.SPOT_TAKER_BPS if fee_mode == "taker" else config.SPOT_MAKER_BPS
    fut = config.FUT_TAKER_BPS if fee_mode == "taker" else config.FUT_MAKER_BPS
    return spot + fut  # one side (entry OR exit), both legs combined


class FundingArb(Strategy):
    def __init__(self, symbol="BTCUSDT", leverage=None):
        self.symbol = symbol
        self.leverage = leverage if leverage is not None else config.LEVERAGE

    def fetch_history(self, start=None, end=None) -> pd.DataFrame:
        return bf.load_symbol(self.symbol, start or config.FUNDING_START, end or config.FUNDING_END)

    def generate_signal(self, data):
        """Threshold-based signal for the future paper/live phases. Not
        used by backtest(), which models the strategy as a single
        open-hold-close cash-flow position instead of discrete trades."""
        funding = data.get("funding_history")
        if funding is None or len(funding) == 0:
            return []

        signals = []
        for _, row in funding.iterrows():
            if row["funding_rate"] > config.ENTRY_THRESHOLD:
                signals.append({"timestamp": row["ts"], "action": "open", "funding_rate": row["funding_rate"]})
            elif row["funding_rate"] < config.EXIT_THRESHOLD:
                signals.append({"timestamp": row["ts"], "action": "close", "funding_rate": row["funding_rate"]})
        return signals

    def backtest(self, data, capital=None, fees=None):
        """Real economics of holding the hedge for the whole period:
        notional sizing under leverage, per-leg entry+exit fees, realized
        distribution stats, and a rolling realistic-holding-period range."""
        funding = data.get("funding_history") if data else None
        if funding is None:
            funding = self.fetch_history()
        if funding is None or len(funding) == 0:
            return {"error": "no funding data"}

        capital = capital or config.CAPITAL
        fee_mode = (fees or {}).get("mode", "taker")
        L = self.leverage

        stats = self._distribution_stats(funding)
        apy = self._net_apy(funding, capital, fee_mode, L)
        result = {"symbol": self.symbol, "leverage": L, **stats, "apy": apy}

        if L > 1:
            liq_buffer_pct = 100 / L
            result["liquidation_buffer_pct"] = round(liq_buffer_pct, 1)
            try:
                kdf = bk.load_daily(self.symbol, config.FUNDING_START, config.FUNDING_END)
                result["worst_move"] = {
                    w: bk.worst_move_stats(kdf, w) for w in (1, 7, 30)
                }
            except Exception as e:
                result["worst_move_error"] = str(e)

        roll = self._rolling_annualized(funding, config.ROLLING_WINDOW_DAYS)
        if len(roll):
            result["rolling"] = {
                "window_days": config.ROLLING_WINDOW_DAYS,
                "min_pct": round(float(roll.min()), 2),
                "p25_pct": round(float(roll.quantile(0.25)), 2),
                "median_pct": round(float(roll.median()), 2),
                "p75_pct": round(float(roll.quantile(0.75)), 2),
                "max_pct": round(float(roll.max()), 2),
                "pct_windows_negative": round(float((roll < 0).mean() * 100), 1),
            }
        return result

    def plan_entry(self, capital, leverage=None, recent_days=90):
        """Pre-entry calculator: exact per-leg dollar amounts, liquidation
        price, expected income (full-history vs recent regime), and a
        go/no-go safety check against real historical worst-case moves.
        Does not place any orders."""
        L = leverage if leverage is not None else self.leverage
        fdf = self.fetch_history()
        kdf = bk.load_daily(self.symbol, config.FUNDING_START, config.FUNDING_END)

        last_price = float(kdf["close"].iloc[-1])
        last_price_date = str(kdf["ts"].iloc[-1].date())

        notional = capital * L / (L + 1)
        spot_cost = notional
        perp_margin = notional / L
        liq_buffer_pct = 100 / L
        liq_price = last_price * (1 + 1 / L)
        unused = max(capital - spot_cost - perp_margin, 0.0)

        entry_fee = notional * 2 * (fee_cost_bps("taker") / 1e4)
        total_fee = entry_fee * 2  # entry + symmetric exit

        full_mean = float(fdf["funding_rate"].mean())
        recent = fdf[fdf["ts"] >= fdf["ts"].max() - pd.Timedelta(days=recent_days)]
        recent_mean = float(recent["funding_rate"].mean()) if len(recent) else float("nan")
        payments_per_year = 365 * 24 / int(fdf["funding_interval_hours"].mode().iloc[0])

        income = {}
        for label, rate in (("full_history", full_mean), (f"recent_{recent_days}d", recent_mean)):
            monthly = rate * notional * (payments_per_year / 12)
            income[label] = {
                "rate_pct_per_8h": round(rate * 100, 4),
                "gross_monthly": round(monthly, 2),
                "gross_annual": round(monthly * 12, 2),
                "gross_annual_pct_of_capital": round(100 * monthly * 12 / capital, 2),
                "net_first_month": round(monthly - total_fee, 2),
            }

        safety = {}
        unsafe_any = False
        for window in (1, 7, 30):
            mv = bk.worst_move_stats(kdf, window)
            if not mv:
                continue
            unsafe = mv["worst_pct"] >= liq_buffer_pct
            unsafe_any = unsafe_any or unsafe
            safety[window] = {**mv, "unsafe": unsafe}

        return {
            "symbol": self.symbol,
            "capital": capital,
            "leverage": L,
            "reference_price": last_price,
            "reference_price_date": last_price_date,
            "sizing": {
                "notional_per_leg": round(notional, 2),
                "spot_leg_cost": round(spot_cost, 2),
                "perp_leg_margin": round(perp_margin, 2),
                "unused_capital": round(unused, 2),
            },
            "liquidation": {
                "buffer_pct": round(liq_buffer_pct, 1),
                "approx_price": round(liq_price, 2),
            },
            "entry_exit_fee": round(total_fee, 2),
            "income": income,
            "safety_check": safety,
            "unsafe_historically": unsafe_any,
        }

    @staticmethod
    def _distribution_stats(df: pd.DataFrame) -> dict:
        r = df["funding_rate"].to_numpy()
        negative = r < 0
        longest_neg = 0
        cur = 0
        for neg in negative:
            cur = cur + 1 if neg else 0
            longest_neg = max(longest_neg, cur)
        interval_h = int(df["funding_interval_hours"].mode().iloc[0]) if len(df) else 8
        payments_per_year = (365 * 24) / interval_h
        return {
            "n_payments": len(df),
            "mean_rate_pct": round(float(r.mean()) * 100, 5),
            "median_rate_pct": round(float(pd.Series(r).median()) * 100, 5),
            "pct_negative": round(float(negative.mean()) * 100, 1),
            "longest_negative_streak_hours": longest_neg * interval_h,
            "naive_annualized_pct": round(float(r.mean()) * payments_per_year * 100, 3),
        }

    @staticmethod
    def _net_apy(df: pd.DataFrame, capital: float, fee_mode: str, leverage: float) -> dict:
        # Spot leg is always unlevered (full notional in cash). Perp leg
        # margin = notional / leverage. Solving notional + notional/L =
        # capital gives notional = capital * L/(L+1). At L=1 this is
        # capital/2. As L -> inf, notional -> capital: a hard ceiling of
        # ~2x the L=1 income, since the spot leg always ties up unlevered
        # cash.
        notional = capital * leverage / (leverage + 1)
        gross_income = float(df["funding_rate"].sum()) * notional
        entry_fee = notional * 2 * (fee_cost_bps(fee_mode) / 1e4)
        total_fee = entry_fee * 2
        net_income = gross_income - total_fee

        days = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400
        years = days / 365
        return {
            "fee_mode": fee_mode,
            "days": round(days, 1),
            "gross_income": round(gross_income, 2),
            "total_fees": round(total_fee, 2),
            "net_income": round(net_income, 2),
            "gross_apy_pct": round(100 * (gross_income / capital) / years, 3) if years > 0 else float("nan"),
            "net_apy_pct": round(100 * (net_income / capital) / years, 3) if years > 0 else float("nan"),
        }

    @staticmethod
    def _rolling_annualized(df: pd.DataFrame, window_days: int) -> pd.Series:
        """Rolling realized annualized funding yield -- shows how much the
        real number moves over time, instead of hiding behind one static
        average."""
        s = df.set_index("ts")["funding_rate"]
        interval_h = int(df["funding_interval_hours"].mode().iloc[0]) if len(df) else 8
        payments_per_window = int(window_days * 24 / interval_h)
        roll_sum = s.rolling(payments_per_window, min_periods=payments_per_window).sum()
        return (roll_sum * (365 / window_days) * 100).dropna()
