import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.alpha_deep_dive import run_strategy, evaluate_trades
from scripts.backtest_daily_alpha import download_or_load_data
from src.regime import build_btc_regime
import pandas as pd

TIER1_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'ADA/USDT', 'SUI/USDT', 'NEAR/USDT', 'INJ/USDT', 'FET/USDT', 'OP/USDT'
]

dfs = download_or_load_data(TIER1_SYMBOLS, timeframe='1h', limit=2500)
symbols = list(dfs.keys())
btc_regime = build_btc_regime(dfs['BTC_USDT'])
total_days = len(dfs['BTC_USDT']) / 24.0

trades = run_strategy(dfs, btc_regime, symbols, donchian_lb=48, pt_m=4.0, sl_m=1.2, vol_mult=1.05)
m_m = evaluate_trades(trades, total_days, cost=0.0006)
m_t = evaluate_trades(trades, total_days, cost=0.0016)

# Split 60% In-Sample / 40% Out-of-Sample
btc_df = dfs['BTC_USDT']
split_date = btc_df.index[int(len(btc_df) * 0.60)]
is_trades = trades[trades['entry_time'] < split_date]
oos_trades = trades[trades['entry_time'] >= split_date]
is_days = (split_date - btc_df.index[0]).total_seconds() / 86400.0
oos_days = (btc_df.index[-1] - split_date).total_seconds() / 86400.0

is_m = evaluate_trades(is_trades, is_days, cost=0.0006)
is_t = evaluate_trades(is_trades, is_days, cost=0.0016)
oos_m = evaluate_trades(oos_trades, oos_days, cost=0.0006)
oos_t = evaluate_trades(oos_trades, oos_days, cost=0.0016)

print("================== WALK-FORWARD VALIDATION ==================")
print(f"In-Sample ({is_days:.1f} days):   Trades={len(is_trades)} ({len(is_trades)/is_days:.2f}/d) | Maker Sharpe={is_m['sharpe']:.2f}, PF={is_m['pf']:.2f}, Ret={is_m['ret_pct']:+.2f}% | Taker Sharpe={is_t['sharpe']:.2f}, PF={is_t['pf']:.2f}, Ret={is_t['ret_pct']:+.2f}%")
print(f"Out-of-Sample ({oos_days:.1f} days): Trades={len(oos_trades)} ({len(oos_trades)/oos_days:.2f}/d) | Maker Sharpe={oos_m['sharpe']:.2f}, PF={oos_m['pf']:.2f}, Ret={oos_m['ret_pct']:+.2f}% | Taker Sharpe={oos_t['sharpe']:.2f}, PF={oos_t['pf']:.2f}, Ret={oos_t['ret_pct']:+.2f}%")

# Per asset
print("\n================== PER ASSET BREAKDOWN ==================")
asset_rows = []
for s in symbols:
    st = trades[trades['symbol'] == s]
    m = evaluate_trades(st, total_days, cost=0.0006)
    asset_rows.append({
        'symbol': s,
        'trades': len(st),
        'win_rate': m['win_rate'],
        'sharpe': m['sharpe'],
        'profit_factor': m['pf'],
        'cum_ret': m['ret_pct'],
        'max_dd': m['max_dd']
    })
adf = pd.DataFrame(asset_rows).sort_values('sharpe', ascending=False)
print(adf.to_string(index=False))
