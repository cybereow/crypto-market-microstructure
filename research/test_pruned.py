import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.alpha_deep_dive import run_strategy, evaluate_trades
from scripts.backtest_daily_alpha import download_or_load_data, DEFAULT_SYMBOLS
from src.regime import build_btc_regime
import pandas as pd

dfs = download_or_load_data(DEFAULT_SYMBOLS, timeframe='1h', limit=2500)
# Filter to 8 high-conviction assets
high_alpha_syms = [s for s in ['BTC_USDT', 'SOL_USDT', 'BNB_USDT', 'XRP_USDT', 'ADA_USDT', 'SUI_USDT', 'NEAR_USDT', 'INJ_USDT'] if s in dfs]
dfs_sub = {s: dfs[s] for s in high_alpha_syms}
btc_regime = build_btc_regime(dfs['BTC_USDT'])
total_days = len(dfs['BTC_USDT']) / 24.0

trades = run_strategy(dfs_sub, btc_regime, high_alpha_syms, donchian_lb=48, pt_m=4.0, sl_m=1.2, vol_mult=1.05)
m_maker = evaluate_trades(trades, total_days, cost=0.0006)
m_taker = evaluate_trades(trades, total_days, cost=0.0016)

print('--- HIGH ALPHA UNIVERSE (8 ASSETS) ---')
print(f'Trades: {len(trades)} ({len(trades)/total_days:.2f}/day)')
print(f"Maker: Sharpe={m_maker['sharpe']:.2f}, PF={m_maker['pf']:.2f}, Win%={m_maker['win_rate']:.2f}%, Ret={m_maker['ret_pct']:+.2f}%, MaxDD={m_maker['max_dd']:.2f}%")
print(f"Taker: Sharpe={m_taker['sharpe']:.2f}, PF={m_taker['pf']:.2f}, Win%={m_taker['win_rate']:.2f}%, Ret={m_taker['ret_pct']:+.2f}%, MaxDD={m_taker['max_dd']:.2f}%")
