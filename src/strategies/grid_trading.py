import pandas as pd
import numpy as np
from collections import deque


class GridTradingStrategy:
    def __init__(self, num_grids=10, grid_range_pct=0.2, initial_capital=10000, fee_pct=0.001, slippage_pct=0.001, grid_type="arithmetic", adaptive_atr_period=None, atr_multiplier=2.0, recenter_cooldown=20, min_spacing_pct=0.005):
        self.num_grids = num_grids
        self.grid_range_pct = grid_range_pct
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.grid_type = grid_type
        self.adaptive_atr_period = adaptive_atr_period
        self.atr_multiplier = atr_multiplier
        self.recenter_cooldown = recenter_cooldown
        self.min_spacing_pct = min_spacing_pct

    def backtest(self, df: pd.DataFrame, regime_mask: pd.Series = None) -> dict:
        """
        Simulates a simple Grid Trading bot.
        Sets up a grid around the starting price and buys when price crosses down a grid line,
        sells when it crosses up.

        regime_mask: optional boolean Series aligned to df.index. Where False,
        the grid does not execute fills or re-center for that bar (e.g. a
        strongly trending regime where grid/mean-reversion has no edge).
        This must gate execution itself rather than being applied to the
        equity curve after the fact — zeroing an already-realized return in
        post-processing doesn't undo the fills that produced it.
        """
        if df.empty:
            return {}

        start_price = df['close'].iloc[0]

        if regime_mask is not None:
            regime_mask = regime_mask.reindex(df.index).fillna(True).to_numpy()
        else:
            regime_mask = np.ones(len(df), dtype=bool)

        # Pre-calculate ATR if adaptive to avoid NameError
        if self.adaptive_atr_period and self.adaptive_atr_period > 0:
            # Calculate ATR using pure Pandas
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(self.adaptive_atr_period).mean().ffill().fillna(true_range.expanding().mean())

            # Start ATR is the ATR at the beginning of the backtest
            start_atr = atr.iloc[0]
            # Grid range is based on ATR
            range_abs = start_atr * self.atr_multiplier
            lower_bound = start_price - (range_abs / 2)
            upper_bound = start_price + (range_abs / 2)
            # Ensure lower bound is positive
            lower_bound = max(1e-9, lower_bound)
        else:
            atr = None
            lower_bound = start_price * (1 - self.grid_range_pct/2)
            upper_bound = start_price * (1 + self.grid_range_pct/2)

        # Adjust num_grids based on min_spacing_pct
        current_num_grids = self.num_grids
        while current_num_grids > 2:
            spacing = (upper_bound - lower_bound) / (current_num_grids - 1)
            if spacing / start_price >= self.min_spacing_pct:
                break
            current_num_grids -= 1

        # Create grid levels
        if self.grid_type == "geometric":
            grid_levels = np.geomspace(lower_bound, upper_bound, current_num_grids)
        else:
            grid_levels = np.linspace(lower_bound, upper_bound, current_num_grids)

        # Assume we allocate 50% capital to quote asset (cash), 50% to base asset (inventory)
        cash = self.initial_capital / 2.0
        inventory = cash / start_price

        buy_grids = len([g for g in grid_levels if g < start_price])
        sell_grids = len([g for g in grid_levels if g >= start_price])

        # Budget cash strictly across the buy grids
        trade_amount_quote = cash / buy_grids if buy_grids > 0 else 0

        # Budget inventory strictly across the sell grids
        trade_amount_base = inventory / sell_grids if sell_grids > 0 else 0

        trades = []
        equity_curve = []

        # Initialize grid tracking.
        # `grid_status[level]`: inventory seeded at initialization (or after
        # a re-center) for levels at/above the reference price — its cost
        # basis is the reference price itself, which sits below `level`, so
        # selling there captures real profit.
        # `open_position[idx]`: inventory bought dynamically when price
        # drops to grid_levels[idx] during normal operation. Its sell target
        # is grid_levels[idx + 1] — one grid step up — which is what
        # actually captures the grid spacing as profit; round-tripping back
        # to the same price (as a single `grid_status`-keyed bucket would)
        # nets to ~0 before costs and a guaranteed small loss after them.
        grid_status = {}
        open_position = {}
        for idx, level in enumerate(grid_levels):
            if level >= start_price:
                # We hold inventory to sell at this upper grid
                grid_status[level] = trade_amount_base
            else:
                # We hold cash, waiting for price to drop to this lower grid
                grid_status[level] = 0
                open_position[idx] = 0.0

        # Final sanity check to prevent negative balances
        total_inventory_allocated = sum(v for v in grid_status.values())
        if total_inventory_allocated > inventory + 1e-9:
            raise ValueError(f"Grid allocation error: Assigned {total_inventory_allocated} but only hold {inventory} base asset.")
        if cash < 0:
            raise ValueError("Grid allocation error: Starting cash is negative.")

        # Pre-compute ATR if adaptive
        if self.adaptive_atr_period and self.adaptive_atr_period > 0:
            atr_values = atr.values
        else:
            atr_values = np.zeros(len(df))

        # Optimize loop using Numpy arrays (vectorized extraction)
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        dates = df.index

        # We track whether we were above or below a level on the PREVIOUS bar.
        # This prevents the "buy triggers on wrong condition" bug, ensuring we only trigger
        # when crossing the line, not just because `low <= level` while sitting above it.
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = start_price

        last_recenter_idx = -self.recenter_cooldown - 1

        for i in range(len(closes)):
            h = highs[i]
            l = lows[i]
            c = closes[i]
            pc = prev_closes[i]

            if not regime_mask[i]:
                # Trending regime: the grid is paused (no fills, no
                # re-center) until a calmer regime resumes. Held cash/
                # inventory simply mark-to-market for this bar below.
                equity = cash + (inventory * c)
                equity_curve.append(equity)
                continue

            # Dynamic buys: price started ABOVE the level and dropped TO or
            # BELOW it. Excludes the top rung (grid_levels[-1]) since there
            # is no higher line to sell a fresh position into.
            for idx, level in enumerate(grid_levels[:-1]):
                if pc > level and l <= level and grid_status[level] == 0 and open_position.get(idx, 0.0) == 0:
                    exec_price = level * (1 + self.slippage_pct)
                    trade_value = trade_amount_quote
                    fee = trade_value * self.fee_pct

                    if cash >= (trade_value + fee):
                        amount_bought = trade_value / exec_price
                        cash -= (trade_value + fee)
                        inventory += amount_bought
                        open_position[idx] = amount_bought
                        trades.append({'time': dates[i], 'type': 'buy', 'price': exec_price, 'amount': amount_bought})

            # Dynamic sells: price reaches the NEXT grid line up from where
            # the position was opened, capturing the grid spacing.
            for idx in range(len(grid_levels) - 2, -1, -1):
                amount_open = open_position.get(idx, 0.0)
                if amount_open <= 0:
                    continue
                target = grid_levels[idx + 1]
                if pc < target and h >= target:
                    exec_price = target * (1 - self.slippage_pct)
                    if inventory >= amount_open * 0.999:
                        trade_value = amount_open * exec_price
                        fee = trade_value * self.fee_pct
                        cash += (trade_value - fee)
                        inventory -= amount_open
                        inventory = max(inventory, 0)
                        open_position[idx] = 0.0
                        trades.append({'time': dates[i], 'type': 'sell', 'price': exec_price, 'amount': amount_open})

            # Sells for the seeded upper-grid inventory (basis = the
            # reference price at seed time, genuinely below `level`).
            for level in reversed(grid_levels):
                if pc < level and h >= level and grid_status[level] > 0:
                    exec_price = level * (1 - self.slippage_pct)
                    amount_to_sell = grid_status[level]

                    if inventory >= amount_to_sell * 0.999:
                        trade_value = amount_to_sell * exec_price
                        fee = trade_value * self.fee_pct
                        cash += (trade_value - fee)
                        inventory -= amount_to_sell
                        inventory = max(inventory, 0)
                        grid_status[level] = 0
                        trades.append({'time': dates[i], 'type': 'sell', 'price': exec_price, 'amount': amount_to_sell})

            # Re-center logic AFTER grid trades to correctly capture intra-bar action
            if (c > upper_bound or c < lower_bound) and (i - last_recenter_idx) >= self.recenter_cooldown:
                last_recenter_idx = i
                # Rebalance portfolio to 50/50
                equity = cash + (inventory * c)
                # target cash and inventory based on total equity
                target_cash = equity / 2.0
                target_inventory = target_cash / c

                # Rebalance by buying/selling difference
                if inventory > target_inventory:
                    # Sell excess
                    amount_to_sell = inventory - target_inventory
                    exec_price = c * (1 - self.slippage_pct)
                    trade_value = amount_to_sell * exec_price
                    fee = trade_value * self.fee_pct
                    cash += (trade_value - fee)
                    inventory -= amount_to_sell
                    trades.append({'time': dates[i], 'type': 'sell', 'price': exec_price, 'amount': amount_to_sell, 'note': 'rebalance'})
                elif inventory < target_inventory:
                    # Buy missing
                    amount_to_buy = target_inventory - inventory
                    exec_price = c * (1 + self.slippage_pct)
                    fee_estimate = amount_to_buy * exec_price * self.fee_pct

                    # Ensure we do not skip due to insufficient cash; buy as much as possible
                    amount_to_buy = min(amount_to_buy, max(0, cash - fee_estimate) / exec_price)

                    if amount_to_buy > 0:
                        trade_value = amount_to_buy * exec_price
                        fee = trade_value * self.fee_pct
                        if cash >= (trade_value + fee):
                            cash -= (trade_value + fee)
                            inventory += amount_to_buy
                            trades.append({'time': dates[i], 'type': 'buy', 'price': exec_price, 'amount': amount_to_buy, 'note': 'rebalance'})

                # Recalculate grid
                if self.adaptive_atr_period and self.adaptive_atr_period > 0:
                    current_atr = atr_values[i]
                    range_abs = current_atr * self.atr_multiplier
                    lower_bound = c - (range_abs / 2)
                    upper_bound = c + (range_abs / 2)
                    lower_bound = max(1e-9, lower_bound)
                else:
                    lower_bound = c * (1 - self.grid_range_pct/2)
                    upper_bound = c * (1 + self.grid_range_pct/2)

                # Adjust num_grids based on min_spacing_pct
                current_num_grids = self.num_grids
                while current_num_grids > 2:
                    spacing = (upper_bound - lower_bound) / (current_num_grids - 1)
                    if spacing / c >= self.min_spacing_pct:
                        break
                    current_num_grids -= 1

                if self.grid_type == "geometric":
                    grid_levels = np.geomspace(lower_bound, upper_bound, current_num_grids)
                else:
                    grid_levels = np.linspace(lower_bound, upper_bound, current_num_grids)

                buy_grids = len([g for g in grid_levels if g < c])
                sell_grids = len([g for g in grid_levels if g >= c])

                trade_amount_quote = cash / buy_grids if buy_grids > 0 else 0
                trade_amount_base = inventory / sell_grids if sell_grids > 0 else 0

                grid_status = {}
                open_position = {}
                for idx, level in enumerate(grid_levels):
                    if level >= c:
                        grid_status[level] = trade_amount_base
                    else:
                        grid_status[level] = 0
                        open_position[idx] = 0.0

            equity = cash + (inventory * c)
            equity_curve.append(equity)

        df_result = pd.DataFrame(index=df.index)
        df_result['equity'] = equity_curve
        df_result['return'] = df_result['equity'].pct_change()

        total_return = (df_result['equity'].iloc[-1] / self.initial_capital) - 1
        buy_hold_return = (df['close'].iloc[-1] / start_price) - 1

        return {
            'final_equity': df_result['equity'].iloc[-1],
            'final_cash': cash,
            'final_inventory': inventory,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'num_trades': len(trades),
            'equity_curve': df_result,
            'trade_stats': self._round_trip_stats(trades),
        }

    def _round_trip_stats(self, trades: list) -> dict:
        """FIFO-match buys against sells to get realized P&L per closed
        round-trip (a grid "trade" in the win-rate sense: bought at one
        level, later sold at a higher level). Inventory still open at the
        end of the backtest is unrealized and excluded, same as the
        position-based trade_level_stats used for the ML/Pairs strategies.
        """
        buy_queue = deque()  # each item: [price, amount]
        round_trip_pnls = []

        for t in trades:
            if t['type'] == 'buy':
                buy_queue.append([t['price'], t['amount']])
                continue

            remaining = t['amount']
            sell_price = t['price']
            pnl = 0.0
            while remaining > 1e-12 and buy_queue:
                buy_price, buy_amt = buy_queue[0]
                matched = min(buy_amt, remaining)
                gross = (sell_price - buy_price) * matched
                fees = (buy_price + sell_price) * matched * self.fee_pct
                pnl += gross - fees
                buy_amt -= matched
                remaining -= matched
                if buy_amt <= 1e-12:
                    buy_queue.popleft()
                else:
                    buy_queue[0][1] = buy_amt
            if remaining <= 1e-12:
                round_trip_pnls.append(pnl)
            # else: sold against inventory that predates this backtest window
            # (shouldn't happen given how the grid seeds its starting
            # inventory) — skip rather than fabricate a cost basis.

        if not round_trip_pnls:
            return {"num_trades": 0, "win_rate": 0.0, "profit_factor": 0.0}

        pnls = np.array(round_trip_pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else (float("inf") if len(wins) else 0.0)

        return {
            "num_trades": len(pnls),
            "win_rate": len(wins) / len(pnls),
            "profit_factor": profit_factor,
        }
