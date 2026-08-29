import pandas as pd
import numpy as np

class GridTradingStrategy:
    def __init__(self, num_grids=10, grid_range_pct=0.2, initial_capital=10000, fee_pct=0.001, slippage_pct=0.001, grid_type="arithmetic", adaptive_atr_period=None, atr_multiplier=2.0):
        self.num_grids = num_grids
        self.grid_range_pct = grid_range_pct
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.grid_type = grid_type
        self.adaptive_atr_period = adaptive_atr_period
        self.atr_multiplier = atr_multiplier

    def backtest(self, df: pd.DataFrame) -> dict:
        """
        Simulates a simple Grid Trading bot.
        Sets up a grid around the starting price and buys when price crosses down a grid line,
        sells when it crosses up.
        """
        if df.empty:
            return {}

        start_price = df['close'].iloc[0]

        # Pre-calculate ATR if adaptive to avoid NameError
        if self.adaptive_atr_period and self.adaptive_atr_period > 0:
            # Calculate ATR using pure Pandas
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(self.adaptive_atr_period).mean().bfill()

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

        # Create grid levels
        if self.grid_type == "geometric":
            grid_levels = np.geomspace(lower_bound, upper_bound, self.num_grids)
        else:
            grid_levels = np.linspace(lower_bound, upper_bound, self.num_grids)

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

        # Initialize grid tracking
        grid_status = {}
        for level in grid_levels:
            if level >= start_price:
                # We hold inventory to sell at this upper grid
                grid_status[level] = trade_amount_base
            else:
                # We hold cash, waiting for price to drop to this lower grid
                grid_status[level] = 0

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

        for i in range(len(closes)):
            h = highs[i]
            l = lows[i]
            c = closes[i]
            pc = prev_closes[i]

            # Re-center logic
            if c > upper_bound or c < lower_bound:
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

                if self.grid_type == "geometric":
                    grid_levels = np.geomspace(lower_bound, upper_bound, self.num_grids)
                else:
                    grid_levels = np.linspace(lower_bound, upper_bound, self.num_grids)

                buy_grids = len([g for g in grid_levels if g < c])
                sell_grids = len([g for g in grid_levels if g >= c])

                trade_amount_quote = cash / buy_grids if buy_grids > 0 else 0
                trade_amount_base = inventory / sell_grids if sell_grids > 0 else 0

                grid_status = {}
                for level in grid_levels:
                    if level >= c:
                        grid_status[level] = trade_amount_base
                    else:
                        grid_status[level] = 0


            # Buys: The price must have started ABOVE the level and dropped TO or BELOW it.
            for level in grid_levels:
                if pc > level and l <= level and grid_status[level] == 0:
                    exec_price = level * (1 + self.slippage_pct)
                    trade_value = trade_amount_quote
                    fee = trade_value * self.fee_pct

                    if cash >= (trade_value + fee):
                        amount_bought = trade_value / exec_price
                        cash -= (trade_value + fee)
                        inventory += amount_bought
                        # We store the base amount bought so we sell exactly this amount later
                        grid_status[level] = amount_bought
                        trades.append({'time': dates[i], 'type': 'buy', 'price': exec_price, 'amount': amount_bought})

            # Sells: The price must have started BELOW the level and rose TO or ABOVE it.
            for level in reversed(grid_levels):
                if pc < level and h >= level and grid_status[level] > 0:
                    exec_price = level * (1 - self.slippage_pct)
                    amount_to_sell = grid_status[level]

                    if inventory >= amount_to_sell * 0.999:
                        trade_value = amount_to_sell * exec_price
                        fee = trade_value * self.fee_pct
                        cash += (trade_value - fee)
                        inventory -= amount_to_sell
                        grid_status[level] = 0
                        trades.append({'time': dates[i], 'type': 'sell', 'price': exec_price, 'amount': amount_to_sell})

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
            'equity_curve': df_result
        }
