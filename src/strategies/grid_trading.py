import pandas as pd
import numpy as np

class GridTradingStrategy:
    def __init__(self, num_grids=10, grid_range_pct=0.2, initial_capital=10000, fee_pct=0.001, slippage_pct=0.001):
        self.num_grids = num_grids
        self.grid_range_pct = grid_range_pct
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

    def backtest(self, df: pd.DataFrame) -> dict:
        """
        Simulates a simple Grid Trading bot.
        Sets up a grid around the starting price and buys when price crosses down a grid line,
        sells when it crosses up.
        """
        if df.empty:
            return {}

        start_price = df['close'].iloc[0]
        lower_bound = start_price * (1 - self.grid_range_pct/2)
        upper_bound = start_price * (1 + self.grid_range_pct/2)

        # Create grid levels
        grid_levels = np.linspace(lower_bound, upper_bound, self.num_grids)

        # Calculate size per grid trade
        # Assume we allocate 50% capital to base asset initially, 50% to quote asset
        capital_quote = self.initial_capital / 2
        capital_base = (self.initial_capital / 2) / start_price

        # Roughly allocate quote capital across the buy grids (lower half)
        buy_grids = len([g for g in grid_levels if g < start_price])
        trade_amount_quote = capital_quote / buy_grids if buy_grids > 0 else 0

        cash = capital_quote
        inventory = capital_base

        trades = []
        equity_curve = []

        # Phantom inventory fix:
        # Base inventory should only cover levels below start_price, so when price drops, we buy.
        # But wait, grid trading means we hold inventory for grids above the current price to sell.
        # If a grid is ABOVE start_price, we ALREADY hold inventory to sell.
        # If a grid is BELOW start_price, we hold CASH to buy.
        grid_status = {}
        for level in grid_levels:
            if level >= start_price:
                # We hold inventory to sell at this upper grid
                amount_held = trade_amount_quote / level
                grid_status[level] = amount_held
            else:
                # We hold cash, waiting for price to drop to this lower grid
                grid_status[level] = 0

        # Adjust starting inventory/cash safely: exactly 50% quote, 50% base
        cash = self.initial_capital / 2.0
        inventory = (self.initial_capital / 2.0) / start_price

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
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'num_trades': len(trades),
            'equity_curve': df_result
        }
