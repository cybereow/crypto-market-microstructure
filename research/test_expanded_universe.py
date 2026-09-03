import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.alpha_deep_dive import run_strategy, evaluate_trades
from scripts.backtest_daily_alpha import download_or_load_data
from src.regime import build_btc_regime
import pandas as pd

EXPANDED_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'SUI/USDT', 'NEAR/USDT',
    'APT/USDT', 'INJ/USDT', 'RENDER/USDT', 'FET/USDT', 'OP/USDT', 'TIA/USDT'
]

dfs = download_or_load_data(EXPANDED_SYMBOLS, timeframe='1h', limit=2500)
symbols = list(dfs.keys())
btc_regime = build_btc_regime(dfs['BTC_USDT'])
total_days = len(dfs['BTC_USDT']) / 24.0

print(f"Loaded {len(symbols)} assets across {total_days:.1f} days.")

trades = run_strategy(dfs, btc_regime, symbols, donchian_lb=48, pt_m=4.0, sl_m=1.2, vol_mult=1.05)
m_maker = evaluate_trades(trades, total_days, cost=0.0006)
m_taker = evaluate_trades(trades, total_days, cost=0.0016)

print("\n================== FULL 17-ASSET UNIVERSE RESULTS ==================")
print(f"Total Trades: {len(trades)} ({len(trades)/total_days:.2f} trades/day)")
print(f"Maker Model (0.06% drag): Sharpe={m_maker['sharpe']:.2f} | PF={m_maker['pf']:.2f} | Return={m_maker['ret_pct']:+.2f}% | MaxDD={m_maker['max_dd']:.2f}% | Win%={m_maker['win_rate']:.2f}%")
print(f"Taker Model (0.16% drag): Sharpe={m_taker['sharpe']:.2f} | PF={m_taker['pf']:.2f} | Return={m_taker['ret_pct']:+.2f}% | MaxDD={m_taker['max_dd']:.2f}% | Win%={m_taker['win_rate']:.2f}%")

longs = trades[trades['side'] == 1]
shorts = trades[trades['side'] == -1]
print(f"Long Trades:  {len(longs)} ({len(longs)/len(trades)*100:.1f}%)")
print(f"Short Trades: {len(shorts)} ({len(shorts)/len(trades)*100:.1f}%)")

# Per-asset breakdown
print("\n--- PER-ASSET BREAKDOWN (MAKER MODEL) ---")
asset_rows = []
for s in symbols:
    st = trades[trades['symbol'] == s]
    if len(st) == 0:
        continue
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
