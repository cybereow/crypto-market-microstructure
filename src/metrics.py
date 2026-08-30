"""Shared backtest metrics.

The standard trading definition of "win rate" is the share of *closed
trades* that were net profitable, not the share of individual bars with a
positive return. A bar-level win rate is dominated by noise from bars spent
sitting inside an open (already-decided) position and understates a
strategy that wins a few big moves and loses many small chops within the
same trade, so it is not comparable to how traders/vendors quote win rate.
"""
import numpy as np
import pandas as pd


def trade_level_stats(position: pd.Series, strat_ret: pd.Series) -> dict:
    """Group a bar-level position/return series into discrete round-trip
    trades (contiguous bars holding a position with the same sign) and
    compute trade-level win rate, average win/loss and profit factor.
    """
    pos = position.fillna(0).to_numpy()
    ret = strat_ret.fillna(0).to_numpy()
    sign = np.sign(pos)

    trade_ids = np.zeros(len(sign), dtype=int)
    current_id = 0
    prev_sign = 0
    for i in range(len(sign)):
        if sign[i] == 0:
            prev_sign = 0
            continue
        if sign[i] != prev_sign:
            current_id += 1
        trade_ids[i] = current_id
        prev_sign = sign[i]

    if current_id == 0:
        return {"num_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0}

    trade_pnls = pd.Series(ret).groupby(trade_ids).sum()
    trade_pnls = trade_pnls[trade_pnls.index != 0]

    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]

    if len(losses) > 0 and losses.sum() != 0:
        profit_factor = wins.sum() / abs(losses.sum())
    else:
        profit_factor = float("inf") if len(wins) > 0 else 0.0

    return {
        "num_trades": int(len(trade_pnls)),
        "win_rate": float(len(wins) / len(trade_pnls)) if len(trade_pnls) > 0 else 0.0,
        "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
        "profit_factor": profit_factor,
    }
