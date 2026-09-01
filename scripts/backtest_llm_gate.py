"""Historical backtest of the LLM (Claude) approval gate on `vol_breakout`
candidates, using already-downloaded 4h OHLCV history instead of waiting
weeks for `scripts/paper_test_llm.py`'s live shadow trader to accumulate a
sample (this signal is rare -- see README section 12).

For every historical candidate, the SAME `src.llm_decision.get_llm_decision`
call used live is made against the indicator snapshot AT THAT SIGNAL BAR
(no lookahead -- `create_features` is rolling/shift-based, so nothing after
the signal bar leaks in). The gate's economics are then compared against
the ungated candidate pool -- and held to the SAME statistical bar as
every other selection in this repo, not a lesser one:
`src.significance.permutation_test` (is the LLM's selection distinguishable
from a random same-size draw?) and `bootstrap_mean_pvalue` (is the
approved subset's net expectancy distinguishable from zero?) -- the exact
tools README section 7 used to retract the traditional-ML meta-labeling
gate once a stricter test was applied. An LLM gate gets no free pass here.

Unlike every other backtest script in this repo, this one costs real money
and real wall-clock time: one live API call per candidate, not a vectorized
replay. Decisions are cached to `data/<--cache>` (default
llm_gate_backtest_cache.csv), keyed by (asset, signal_time), and flushed
every `--flush-every` calls, so an interrupted run or a re-run never
re-pays for an already-decided candidate. Use `--limit` to cap spend on a
first run (start small: a few hundred candidates for BTC/ETH/SOL on a
year or two of 4h data is a reasonable first sample).

Needs ANTHROPIC_API_KEY in the environment.
"""
import argparse
import os
import sys

import anthropic
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR
from scripts.train_ml import create_features
from src.labeling import volatility_breakout_entries, triple_barrier_labels
from src.metrics import net_pf_expectancy
from src.significance import permutation_test, bootstrap_mean_pvalue
from src.paper_trading import LOOKBACK, PT_MULT, SL_MULT, MAX_HOLDING_BARS
from src.llm_decision import FEATURE_KEYS, get_llm_decision

MODEL = 'claude-sonnet-5'
CACHE_COLUMNS = ['asset', 'signal_time', 'decision', 'confidence', 'reason']


def load_cache(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=CACHE_COLUMNS)
    df = pd.read_csv(path)
    df['signal_time'] = pd.to_datetime(df['signal_time'])
    return df


def save_cache(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def gather_candidates(data_files: list) -> tuple:
    """Detect vol_breakout candidates on each asset's already-downloaded
    history and label them with the SAME triple-barrier geometry
    scripts/paper_test_llm.py uses live (src.paper_trading's PT_MULT/
    SL_MULT/MAX_HOLDING_BARS), so a backtest result here is directly
    comparable to what the live gate would have logged.

    Returns (market_df, features_by_key): a pooled DataFrame indexed by
    signal_time with an 'asset' column, side/label/ret/signal_price/atr;
    and a dict mapping (asset, signal_time) -> the FEATURE_KEYS snapshot
    handed to the LLM.
    """
    frames = []
    features_by_key = {}
    for data_file in data_files:
        path = os.path.join(OUTPUT_DIR, data_file)
        if not os.path.exists(path):
            print(f"  {data_file}: not found, skipping.")
            continue
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df_features = create_features(df)
        raw_atr = df_features['ATR_14'] * df_features['close']
        entries = volatility_breakout_entries(df_features, lookback=LOOKBACK)

        market = triple_barrier_labels(df_features, entries, raw_atr, pt_mult=PT_MULT,
                                        sl_mult=SL_MULT, max_holding=MAX_HOLDING_BARS)
        if market.empty:
            print(f"  {data_file}: no candidates, skipping.")
            continue

        market['asset'] = data_file
        market['signal_price'] = df_features.loc[market.index, 'close'].to_numpy()
        market['atr'] = raw_atr.loc[market.index].to_numpy()
        frames.append(market)

        for ts in market.index:
            row = df_features.loc[ts]
            features_by_key[(data_file, ts)] = {
                key: (float(row[key]) if key in row.index and pd.notna(row[key]) else None)
                for key in FEATURE_KEYS
            }
        print(f"  {data_file}: {len(market)} candidates")

    if not frames:
        return pd.DataFrame(), {}
    return pd.concat(frames), features_by_key


def main():
    parser = argparse.ArgumentParser(
        description="Historical backtest of the Claude approval gate on vol_breakout candidates.")
    parser.add_argument("--data", type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap the number of NEW (uncached) LLM calls this run makes. "
                              "Each call costs real money and real time -- start small.")
    parser.add_argument("--cache", type=str, default="llm_gate_backtest_cache.csv",
                         help="Filename under data/ where decisions are cached, keyed by "
                              "(asset, signal_time), so a re-run never re-pays for an "
                              "already-decided candidate.")
    parser.add_argument("--flush-every", type=int, default=20)
    args = parser.parse_args()

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. This script calls the real Claude API once per "
            "candidate trade.")
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Gathering vol_breakout candidates from {len(args.data)} file(s)...")
    market, features_by_key = gather_candidates(args.data)
    if market.empty:
        print("No candidates found.")
        return
    print(f"\n{len(market)} total candidates across all assets.\n")

    cache_path = os.path.join(OUTPUT_DIR, args.cache)
    cache = load_cache(cache_path)
    cached_keys = set(zip(cache['asset'], cache['signal_time'])) if len(cache) else set()

    new_rows = []
    calls_made = 0
    for ts, row in market.iterrows():
        key = (row['asset'], ts)
        if key in cached_keys:
            continue
        if args.limit is not None and calls_made >= args.limit:
            print(f"  --limit {args.limit} reached; remaining candidates stay undecided this run.")
            break

        decision = get_llm_decision(client, args.model, row['asset'], int(row['side']),
                                     float(row['signal_price']), float(row['atr']),
                                     features_by_key[key])
        new_rows.append({'asset': row['asset'], 'signal_time': ts,
                          'decision': decision['decision'], 'confidence': decision['confidence'],
                          'reason': decision['reason']})
        calls_made += 1
        if calls_made % 10 == 0:
            print(f"  ...{calls_made} new decisions made")
        if len(new_rows) >= args.flush_every:
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            save_cache(cache, cache_path)
            new_rows = []

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        save_cache(cache, cache_path)

    print(f"\n{calls_made} new LLM calls made this run; {len(cache)} decisions cached in total.")

    decision_map = dict(zip(zip(cache['asset'], cache['signal_time']), cache['decision']))
    market = market.copy()
    market['key'] = list(zip(market['asset'], market.index))
    market['llm_decision'] = market['key'].map(decision_map)

    decided = market[market['llm_decision'].notna()]
    if decided.empty:
        print("No decisions available yet -- increase --limit or re-run.")
        return

    approved_mask = (decided['llm_decision'] == 'approve').to_numpy()
    labels = decided['label'].to_numpy()
    rets = decided['ret'].to_numpy()
    n, n_approved = len(decided), int(approved_mask.sum())

    print(f"\n{'=' * 78}")
    print(f"  LLM gate results on {n} decided candidates "
          f"({n_approved} approved, {n - n_approved} rejected)")
    print(f"{'=' * 78}")

    net_rets_maker = rets - 0.0008
    for cost, label in [(0.004, 'TAKER'), (0.0008, 'MAKER')]:
        all_pf, all_exp = net_pf_expectancy(rets, cost)
        all_wr = float(labels.mean())
        if n_approved > 0:
            appr_pf, appr_exp = net_pf_expectancy(rets[approved_mask], cost)
            appr_wr = float(labels[approved_mask].mean())
        else:
            appr_pf, appr_exp, appr_wr = float('nan'), float('nan'), float('nan')
        print(f"  {label} cost {cost:.2%} -- ALL candidates: n={n:4d}  win rate {all_wr:.1%}  "
              f"PF {all_pf:.2f}  exp/trade {all_exp:+.4%}")
        print(f"  {label} cost {cost:.2%} -- LLM-approved  : n={n_approved:4d}  "
              f"win rate {appr_wr:.1%}  PF {appr_pf:.2f}  exp/trade {appr_exp:+.4%}")

    if 0 < n_approved < n:
        perm_wr = permutation_test(labels, approved_mask, statistic='win_rate')
        perm_ret = permutation_test(labels, approved_mask, statistic='mean_return',
                                    returns=net_rets_maker)
        print(f"\n  Permutation test (does the LLM's selection beat a random same-size draw?):")
        print(f"    win_rate   : gate {perm_wr['statistic']:.1%} vs random-draw mean "
              f"{perm_wr['null_mean']:.1%}  p={perm_wr['p_value']:.4f}")
        print(f"    mean_return (maker basis): gate {perm_ret['statistic']:+.4%} vs "
              f"random-draw mean {perm_ret['null_mean']:+.4%}  p={perm_ret['p_value']:.4f}")
        p_boot = bootstrap_mean_pvalue(net_rets_maker[approved_mask])
        print(f"    bootstrap p(mean net maker-cost return > 0 | approved only): {p_boot:.4f}")
    else:
        print("\n  Degenerate selection (gate approved everything or nothing this run) -- "
              "permutation test skipped.")

    out_path = os.path.join(OUTPUT_DIR, 'llm_gate_backtest_results.csv')
    decided.drop(columns=['key']).to_csv(out_path)
    print(f"\n  Full per-candidate detail saved to {out_path}")


if __name__ == "__main__":
    main()
