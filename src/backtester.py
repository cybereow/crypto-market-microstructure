"""
Generic trade-signal backtester (entry/exit price pairs -> PnL stats).

Funding-arb does not use this: it is a cash-flow carry strategy (holds one
hedge and collects periodic payments), not a sequence of round-trip trades,
so its economics are computed directly in strategies/funding.py instead.
This backtester is here for future signal-based strategies (basis, grid).
"""
import numpy as np


class Backtester:
    def __init__(self, capital=10000, maker_fee=0.0002, taker_fee=0.0005):
        self.capital = capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    def run(self, strategy, data):
        signals = strategy.generate_signal(data)
        trades = self._simulate(signals)
        return self._analyze(trades)

    def _simulate(self, signals):
        trades = []
        for signal in signals:
            if signal["action"] not in ("buy", "sell"):
                continue
            fee = self.taker_fee
            cost = signal["qty"] * signal["price"] * fee
            pnl = signal.get("exit_price", signal["price"]) * signal["qty"] - signal["price"] * signal["qty"]
            net_pnl = pnl - cost

            trades.append({
                "timestamp": signal["timestamp"],
                "action": signal["action"],
                "price": signal["price"],
                "qty": signal["qty"],
                "fee": cost,
                "gross_pnl": pnl,
                "net_pnl": net_pnl
            })
        return trades

    def _analyze(self, trades):
        if not trades:
            return {"error": "no trades"}

        pnls = [t["net_pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "total_pnl": sum(pnls),
            "total_return_pct": sum(pnls) / self.capital * 100,
            "max_drawdown_pct": self._max_drawdown(pnls),
            "profit_factor": abs(sum(wins) / sum(losses)) if losses else float("inf"),
            "total_fees": sum(t["fee"] for t in trades),
            "fee_drag_pct": sum(t["fee"] for t in trades) / self.capital * 100
        }

    def _max_drawdown(self, pnls):
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (self.capital + peak) * 100
        return max(drawdown) if len(drawdown) > 0 else 0
