"""Daily signal digest: generate ~N tradeable signals per day and report,
honestly, whether that digest makes or loses money.

This is the CLI for `src/daily_signals.py`. It does three things:

  1. Pools every rule-based primary signal across a universe of assets into
     one candidate set, with each candidate's triple-barrier outcome.
  2. Ranks candidates by the fixed, unfitted conviction score and spends a
     fixed daily budget of N slots on the best ones (the digest product).
  3. Grades that digest with the same cost model and significance tests the
     rest of the repo uses — at taker cost AND at maker cost, because §8 of
     the research log showed cost, not the model, is the deciding variable.

Two selection methods are reported side by side:
  - top-N-per-day  : the literal "3-4 signals/day" digest (bounded same-day
                     ranking lookahead — this is the *presentation*);
  - threshold      : fully causal, frequency-calibrated (this is the *honest
                     economics headline* — no future information at all).

Run the backtest:
  python scripts/daily_signal_report.py \
      --data binance_ETH_USDT_1h.csv binance_SOL_USDT_1h.csv ... \
      --btc-regime-file binance_BTC_USDT_1h.csv \
      --signals reversion vol_breakout trend_pullback \
      --pt-mult 2.0 --sl-mult 2.0 --per-day 4

Emit today's digest (the latest calendar day in the data):
  python scripts/daily_signal_report.py --data ... --btc-regime-file ... --today
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features
from src.daily_signals import (
    SIGNAL_FUNCS, build_candidates, attach_scores, btc_trend_strength,
    select_top_n_per_day, select_by_threshold, threshold_for_frequency,
)
from src.metrics import net_pf_expectancy
from src.significance import (
    binomial_ci, breakeven_win_rate, permutation_test, bootstrap_mean_pvalue,
    deflated_pvalue,
)


def load_features(path: str) -> tuple:
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    feats = create_features(df)
    raw_atr = feats['ATR_14'] * feats['close']
    return feats, raw_atr


def economics_row(name: str, sel: pd.DataFrame, pool_labels: np.ndarray,
                  pool_mask: np.ndarray, cost: float, payoff_ratio: float,
                  span_days: int, n_cost_configs: int) -> dict:
    """Compute the full economics + significance verdict for one selection."""
    rets = sel['ret'].to_numpy()
    labels = sel['label'].to_numpy()
    n = len(sel)
    wr = float((labels == 1).mean()) if n else float('nan')
    pf, exp = net_pf_expectancy(rets, cost)
    lo, hi = binomial_ci(int((labels == 1).sum()), n) if n else (float('nan'),) * 2
    be = breakeven_win_rate(payoff_ratio, cost)
    # Permutation: does picking THESE trades beat picking the same number at
    # random from the pool? (tests the ranking, independent of base rate)
    perm = permutation_test(pool_labels, pool_mask, n_iter=5000, random_state=42)
    # Bootstrap: is net (after-cost) per-trade expectancy > 0?
    boot_p = bootstrap_mean_pvalue(rets - cost, n_iter=5000)
    boot_p_def = deflated_pvalue(boot_p, n_cost_configs)
    return {
        'name': name, 'n': n, 'per_day': n / max(span_days, 1),
        'win_rate': wr, 'ci_low': lo, 'ci_high': hi, 'breakeven': be,
        'pf': pf, 'exp': exp, 'perm_p': perm['p_value'],
        'boot_p': boot_p, 'boot_p_def': boot_p_def,
    }


def print_rows(title: str, rows: list):
    print(f"\n{'=' * 92}")
    print(f"  {title}")
    print(f"{'=' * 92}")
    print(f"  {'selection':<26}{'n':>6}{'/day':>7}{'win%':>7}{'break%':>8}"
          f"{'PF':>7}{'exp/trade':>12}{'perm_p':>9}{'boot_p*':>9}")
    print(f"  {'-' * 88}")
    for r in rows:
        print(f"  {r['name']:<26}{r['n']:>6}{r['per_day']:>7.2f}"
              f"{r['win_rate'] * 100:>7.1f}{r['breakeven'] * 100:>8.1f}"
              f"{r['pf']:>7.2f}{r['exp']:>+12.3%}{r['perm_p']:>9.4f}{r['boot_p_def']:>9.4f}")


def verdict_line(maker_rows: list):
    """One honest sentence about the maker-cost headline (threshold method)."""
    thr = next((r for r in maker_rows if r['name'].startswith('threshold')), None)
    if thr is None:
        return
    profitable = thr['pf'] > 1.0 and thr['exp'] > 0
    sig = thr['boot_p_def'] < 0.05 and thr['ci_low'] > thr['breakeven']
    print(f"\n  Verdict (maker cost, causal threshold selection):")
    if profitable and sig:
        print(f"    Positive expectancy ({thr['exp']:+.3%}/trade, PF {thr['pf']:.2f}) AND "
              f"significant (deflated p={thr['boot_p_def']:.4f}, CI low "
              f"{thr['ci_low']:.1%} > breakeven {thr['breakeven']:.1%}).")
    elif profitable:
        print(f"    Positive expectancy ({thr['exp']:+.3%}/trade, PF {thr['pf']:.2f}) but NOT "
              f"significant after correction (deflated p={thr['boot_p_def']:.4f}). "
              f"Consistent with a small, cost-fragile edge — treat as unproven.")
    else:
        print(f"    Negative expectancy ({thr['exp']:+.3%}/trade, PF {thr['pf']:.2f}). The §8 "
              f"cost wall stands: ranking helps quality but does not by itself clear cost here.")


def format_digest(day_df: pd.DataFrame) -> str:
    """Human-readable 'today's signals' table."""
    if day_df.empty:
        return "  (no candidate signals fired on this day)"
    lines = []
    for ts, row in day_df.sort_values('score', ascending=False).iterrows():
        direction = 'LONG ' if row['side'] > 0 else 'SHORT'
        asset = str(row['asset']).replace('binance_', '').replace('_1h.csv', '').replace('.csv', '')
        lines.append(f"    {ts:%Y-%m-%d %H:%M}  {direction}  {asset:<10} "
                     f"[{row['signal']:<14}]  conviction={row['score']:.3f}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs='+', required=True, help="Universe OHLCV CSVs.")
    p.add_argument("--btc-regime-file", required=True,
                   help="BTC OHLCV CSV, used only as the regime context (not traded).")
    p.add_argument("--signals", nargs='+', default=['reversion', 'vol_breakout', 'trend_pullback'],
                   choices=list(SIGNAL_FUNCS.keys()))
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--pt-mult", type=float, default=2.0)
    p.add_argument("--sl-mult", type=float, default=2.0)
    p.add_argument("--max-holding", type=int, default=18)
    p.add_argument("--per-day", type=int, default=4, help="Daily signal budget N.")
    p.add_argument("--taker-cost", type=float, default=0.004,
                   help="Round-trip taker cost (0.1%% fee + 0.1%% slippage per side).")
    p.add_argument("--maker-cost", type=float, default=0.0008,
                   help="Round-trip maker cost (~0.02%% fee + 0.02%% slippage per side).")
    p.add_argument("--today", action='store_true',
                   help="Print the latest calendar day's top-N digest instead of the backtest.")
    args = p.parse_args()

    payoff_ratio = args.pt_mult / args.sl_mult

    # BTC regime strength (point-in-time).
    btc_path = os.path.join(OUTPUT_DIR, args.btc_regime_file)
    btc_feats, _ = load_features(btc_path)
    btc_strength = btc_trend_strength(btc_feats)

    # Pool candidates across the universe.
    pool = []
    for f in args.data:
        path = os.path.join(OUTPUT_DIR, f)
        if not os.path.exists(path):
            print(f"  {f}: not found, skipping.")
            continue
        feats, raw_atr = load_features(path)
        cand = build_candidates(feats, raw_atr, asset=f, signals=args.signals,
                                lookback=args.lookback, pt_mult=args.pt_mult,
                                sl_mult=args.sl_mult, max_holding=args.max_holding)
        if not cand.empty:
            pool.append(cand)
            print(f"  {f}: {len(cand)} candidates")
    if not pool:
        print("No candidates. Did you download the data?")
        return

    scored = attach_scores(pd.concat(pool).sort_index(), btc_strength)
    span_days = max((scored.index.max() - scored.index.min()).days, 1)

    if args.today:
        last_day = pd.Index(scored.index).normalize().max()
        digest = select_top_n_per_day(scored[pd.Index(scored.index).normalize() == last_day],
                                      n=args.per_day)
        print(f"\n{'=' * 70}\n  DAILY SIGNAL DIGEST — {last_day:%Y-%m-%d} (top {args.per_day})\n{'=' * 70}")
        print(format_digest(digest))
        print(f"\n  Ranking = fixed conviction score (BTC regime + volatility state + "
              f"trend agreement).\n  Not investment advice; see the backtest for cost-adjusted "
              f"expectancy.")
        return

    base_wr = float((scored['label'] == 1).mean())
    print(f"\n  Pool: {len(scored)} candidates over {span_days} days "
          f"({scored.index.min():%Y-%m-%d} -> {scored.index.max():%Y-%m-%d})")
    print(f"  Signals: {', '.join(args.signals)}   geometry pt/sl={args.pt_mult}/{args.sl_mult} "
          f"(payoff {payoff_ratio:.2f}, breakeven {1/(1+payoff_ratio):.1%})")
    print(f"  Base pool win rate (all candidates, no cost): {base_wr:.1%}")

    # Two selections.
    topn = select_top_n_per_day(scored, n=args.per_day)
    thr = threshold_for_frequency(scored, target_per_day=args.per_day)
    thr_sel = select_by_threshold(scored, thr)

    pool_labels = scored['label'].to_numpy()
    # Robust membership mask by (asset, timestamp, signal, score) tuple —
    # timestamps repeat across assets/signals, so no single column is a key.
    def mk_mask(sel):
        key_pool = list(zip(scored['asset'], scored.index, scored['signal'], scored['score'].round(6)))
        key_sel = set(zip(sel['asset'], sel.index, sel['signal'], sel['score'].round(6)))
        return np.array([k in key_sel for k in key_pool])
    topn_mask = mk_mask(topn)
    thr_mask = mk_mask(thr_sel)

    # n_cost_configs=2: we report the same selection at taker AND maker cost,
    # so the bootstrap p is deflated for having looked at two cost regimes.
    for cost, label in [(args.taker_cost, 'TAKER'), (args.maker_cost, 'MAKER')]:
        rows = [
            economics_row(f'top-{args.per_day}/day', topn, pool_labels, topn_mask,
                          cost, payoff_ratio, span_days, 2),
            economics_row(f'threshold>={thr:.3f}', thr_sel, pool_labels, thr_mask,
                          cost, payoff_ratio, span_days, 2),
        ]
        print_rows(f"{label} cost = {cost:.2%} round-trip", rows)
        if label == 'MAKER':
            maker_rows = rows

    verdict_line(maker_rows)
    print(f"\n  * boot_p is the bootstrap p-value for net expectancy>0, deflated for the two "
          f"cost regimes examined.\n    perm_p asks whether the ranking beats a random pick of "
          f"the same size from the pool.")
    print(f"  Reminder: maker economics assume passive limit orders actually fill. "
          f"scripts/backtest_maker_fill.py\n  stress-tests that assumption (adverse selection); "
          f"RESEARCH_LOG §9 shows it is the real risk to this edge.")


if __name__ == "__main__":
    main()
