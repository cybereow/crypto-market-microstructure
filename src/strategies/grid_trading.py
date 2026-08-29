import pandas as pd
import numpy as np

class GridTradingStrategy:
    def __init__(self, num_grids=10, grid_range_pct=0.2, initial_capital=10000):
        self.num_grids = num_grids
        self.grid_range_pct = grid_range_pct
        self.initial_capital = initial_capital

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

        # Track which grid levels are active to prevent multiple triggers
        # 1 means we hold the inventory bought at this level, 0 means we have cash waiting to buy
        grid_status = {level: (1 if level >= start_price else 0) for level in grid_levels}

        for index, row in df.iterrows():
            current_price = row['close']

            # Check for buys (price dropped below a grid level we haven't bought yet)
            for level in grid_levels:
                if current_price < level and grid_status[level] == 0:
                    # Buy
                    if cash >= trade_amount_quote:
                        amount_bought = trade_amount_quote / current_price
                        cash -= trade_amount_quote
                        inventory += amount_bought
                        grid_status[level] = 1 # Mark as bought
                        trades.append({'time': index, 'type': 'buy', 'price': current_price, 'amount': amount_bought})

            # Check for sells (price rose above a grid level we are currently holding)
            for level in reversed(grid_levels):
                if current_price > level and grid_status[level] == 1:
                    # Sell
                    amount_to_sell = trade_amount_quote / level # Simplify: sell the nominal amount allocated
                    if inventory >= amount_to_sell:
                        cash += amount_to_sell * current_price
                        inventory -= amount_to_sell
                        grid_status[level] = 0 # Mark as sold (ready to buy again)
                        trades.append({'time': index, 'type': 'sell', 'price': current_price, 'amount': amount_to_sell})

            # Record daily equity
            equity = cash + (inventory * current_price)
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
