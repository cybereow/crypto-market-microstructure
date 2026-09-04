"""Historical backtest of the LLM (Claude) approval gate on ANY primary
signal's candidates (`--signal`, from `SIGNAL_BUILDERS` in
scripts/train_meta_ml.py -- vol_breakout, funding_reversion, etc.), using
already-downloaded 4h OHLCV history instead of waiting weeks for
`scripts/paper_test_llm.py`'s live shadow trader to accumulate a sample.

For every historical candidate, the SAME `src.llm_decision.get_llm_decision`
call used live is made against the indicator snapshot AT THAT SIGNAL BAR
(no lookahead -- `create_features` is rolling/shift-based, so nothing after
the signal bar leaks in), told which signal it's gating so the prompt
states that signal's own premise rather than a generic breakout framing
(see `src.llm_decision.SIGNAL_DESCRIPTIONS`). The gate's economics are
then compared against the ungated candidate pool -- and held to the SAME
statistical bar as every other selection in this repo, not a lesser one:
`src.significance.permutation_test` (is the LLM's selection distinguishable
from a random same-size draw?) and `bootstrap_mean_pvalue` (is the
approved subset's net expectancy distinguishable from zero?) -- the exact
tools README section 7 used to retract the traditional-ML meta-labeling
gate once a stricter test was applied. An LLM gate gets no free pass here.

Two levers this script exposes for actually improving the gate's economics,
both reusing methodology this repo already validated rather than inventing
a new one to chase a number:

  - `--min-confidence`: the LLM returns a 0-1 confidence with every
    decision (`src/llm_decision.py`); requiring a higher one before
    counting a candidate as approved trades fewer, hopefully better,
    setups -- the same precision/threshold tradeoff `src/calibration.py`
    already applies to the traditional-ML gate.
  - Realistic maker-fill execution on the approved subset
    (`src.execution.simulate_maker_fills` / `triple_barrier_from_fill`,
    same OPTIMISTIC-upper-bound OHLC queue simulation as README section 9):
    the instant-fill "MAKER cost" figure below only swaps the cost
    assumption on trades entered at the signal close, which overstates
    what's actually reachable. This is the same lever that took the raw
    signal from net-negative to net-significant in section 8->9 -- so it's
    applied here too, on however many candidates the gate actually approves.

`--pt-mult`/`--sl-mult`/`--lookback`/`--max-holding` default to
vol_breakout's own tuned geometry (2.0/1.0/20/18, matching
src/paper_trading.py's live constants). A different signal generally
wants different geometry -- e.g. funding_reversion is a symmetric
reversion play tested at 2.0/2.0 with a 90-bar lookback in
scripts/backtest_funding_reversion.py (README section 14); pass the same
values here with `--signal funding_reversion --funding-data ...` for a
directly comparable gated run.

Unlike every other backtest script in this repo, this one costs real money
and real wall-clock time: one live API call per candidate, not a vectorized
replay. Decisions are cached to `data/<--cache>` (default
llm_gate_backtest_cache.csv), keyed by (asset, signal_time), and flushed
every `--flush-every` calls, so an interrupted run or a re-run never
re-pays for an already-decided candidate. Use `--limit` to cap spend on a
first run.

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
from scripts.train_meta_ml import SIGNAL_BUILDERS
from src.labeling import triple_barrier_labels
from src.execution import simulate_maker_fills, triple_barrier_from_fill
from src.metrics import net_pf_expectancy
from src.significance import permutation_test, bootstrap_mean_pvalue
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


def gather_candidates(data_files: list, signal: str = 'vol_breakout', lookback: int = 20,
                       pt_mult: float = 2.0, sl_mult: float = 1.0, max_holding: int = 18,
                       funding_data_files: list = None) -> tuple:
    """Detect `signal`'s candidates on each asset's already-downloaded
    history and label them with the given triple-barrier geometry.

    `funding_data_files`, if given, must be the same length as
    `data_files` and in the same order (one funding-rate file per asset,
    like scripts/backtest_funding_reversion.py) -- required when
    `signal='funding_reversion'`, which needs a 'funding_rate' column
    joined in before feature-building.

    Returns (market_df, features_by_key, per_asset):
      - market_df: a pooled DataFrame indexed by signal_time with an
        'asset' column, side/label/ret/signal_price/atr (instant-fill
        basis)
      - features_by_key: dict mapping (asset, signal_time) -> the
        FEATURE_KEYS snapshot handed to the LLM
      - per_asset: dict mapping asset -> {df_features, raw_atr, entries},
        kept around so the approved subset can later be re-run through
        `src.execution.simulate_maker_fills` for a realistic (not
        instant-fill) execution check
    """
    if funding_data_files is not None and len(funding_data_files) != len(data_files):
        raise ValueError(
            f"--funding-data has {len(funding_data_files)} file(s) but --data has "
            f"{len(data_files)} -- need exactly one funding file per asset, in the same order.")

    frames = []
    features_by_key = {}
    per_asset = {}
    for i, data_file in enumerate(data_files):
        path = os.path.join(OUTPUT_DIR, data_file)
        if not os.path.exists(path):
            print(f"  {data_file}: not found, skipping.")
            continue
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)

        if funding_data_files is not None:
            funding_path = os.path.join(OUTPUT_DIR, funding_data_files[i])
            if not os.path.exists(funding_path):
                print(f"  {funding_data_files[i]}: not found, skipping {data_file}.")
                continue
            funding_df = pd.read_csv(funding_path, index_col='timestamp', parse_dates=True)
            df = df.join(funding_df[['funding_rate']], how='left')
            df['funding_rate'] = df['funding_rate'].ffill()

        df_features = create_features(df)
        raw_atr = df_features['ATR_14'] * df_features['close']
        entries = SIGNAL_BUILDERS[signal](df_features, lookback)

        market = triple_barrier_labels(df_features, entries, raw_atr, pt_mult=pt_mult,
                                        sl_mult=sl_mult, max_holding=max_holding)
        if market.empty:
            print(f"  {data_file}: no candidates, skipping.")
            continue

        market['asset'] = data_file
        market['signal_price'] = df_features.loc[market.index, 'close'].to_numpy()
        market['atr'] = raw_atr.loc[market.index].to_numpy()
        frames.append(market)
        per_asset[data_file] = {'df_features': df_features, 'raw_atr': raw_atr, 'entries': entries}

        for ts in market.index:
            row = df_features.loc[ts]
            features_by_key[(data_file, ts)] = {
                key: (float(row[key]) if key in row.index and pd.notna(row[key]) else None)
                for key in FEATURE_KEYS
            }
        print(f"  {data_file}: {len(market)} candidates")

    if not frames:
        return pd.DataFrame(), {}, {}
    return pd.concat(frames), features_by_key, per_asset


def maker_fill_economics(approved: pd.DataFrame, per_asset: dict, maker_cost: float,
                          offset_mult: float = 0.15, queue_timeout: int = 3,
                          pt_mult: float = 2.0, sl_mult: float = 1.0,
                          max_holding: int = 18) -> dict:
    """Re-simulate REALISTIC maker-order execution (src.execution's
    OHLC-only queue simulation, README section 9) restricted to just the
    candidates the LLM gate approved, instead of the instant-fill-at-close
    assumption `triple_barrier_labels` uses. Per asset: zero out every
    non-approved candle's entry so `simulate_maker_fills` only ever prices
    a resting limit at the gate's own approved signals, then re-label only
    the ones that actually filled from their real fill bar/price.

    Returns a dict with fill_rate, n_filled, win_rate, pf, expectancy,
    and a bootstrap p-value on the filled subset's net returns -- or
    n_filled=0 / NaNs if nothing in `approved` filled (or nothing was
    approved at all).
    """
    fills_frames, maker_frames = [], []
    for asset, group in approved.groupby('asset'):
        data = per_asset.get(asset)
        if data is None:
            continue
        approved_times = set(group.index)
        gated_entries = pd.Series(0, index=data['entries'].index, dtype=data['entries'].dtype)
        mask = gated_entries.index.isin(approved_times)
        gated_entries.loc[mask] = data['entries'].loc[mask]

        fills = simulate_maker_fills(data['df_features'], gated_entries, data['raw_atr'],
                                     offset_mult=offset_mult, queue_timeout=queue_timeout)
        maker = triple_barrier_from_fill(data['df_features'], fills, data['raw_atr'],
                                         pt_mult=pt_mult, sl_mult=sl_mult,
                                         max_holding=max_holding)
        if len(fills):
            fills_frames.append(fills)
        if len(maker):
            maker_frames.append(maker)

    n_candidates = len(approved)
    if not fills_frames:
        return {'n_candidates': n_candidates, 'n_filled': 0, 'fill_rate': float('nan'),
               'win_rate': float('nan'), 'pf': float('nan'), 'expectancy': float('nan'),
               'p_value': float('nan')}

    fills_df = pd.concat(fills_frames)
    fill_rate = float(fills_df['filled'].mean()) if len(fills_df) else float('nan')

    if not maker_frames:
        return {'n_candidates': n_candidates, 'n_filled': 0, 'fill_rate': fill_rate,
               'win_rate': float('nan'), 'pf': float('nan'), 'expectancy': float('nan'),
               'p_value': float('nan')}

    maker_df = pd.concat(maker_frames)
    rets = maker_df['ret'].to_numpy()
    pf, expectancy = net_pf_expectancy(rets, maker_cost)
    win_rate = float((maker_df['label'] == 1).mean())
    p_value = bootstrap_mean_pvalue(rets - maker_cost)

    return {'n_candidates': n_candidates, 'n_filled': len(maker_df), 'fill_rate': fill_rate,
            'win_rate': win_rate, 'pf': pf, 'expectancy': expectancy, 'p_value': p_value}


def main():
    parser = argparse.ArgumentParser(
        description="Historical backtest of the Claude approval gate on a primary signal's candidates.")
    parser.add_argument("--data", type=str, nargs='+', required=True)
    parser.add_argument("--signal", type=str, default="vol_breakout",
                        choices=list(SIGNAL_BUILDERS.keys()),
                        help="Which src/labeling.py primary signal to gate. "
                             "'funding_reversion' requires --funding-data.")
    parser.add_argument("--funding-data", type=str, nargs='+', default=None,
                        help="One funding-rate file per --data file, in the SAME order "
                             "(download_funding_vision.py). Required for "
                             "--signal funding_reversion.")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--pt-mult", type=float, default=2.0)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--max-holding", type=int, default=18)
    parser.add_argument("--offset-mult", type=float, default=0.15,
                        help="How much better than the signal price the resting maker limit "
                             "is priced, in ATR multiples (used by the realistic maker-fill "
                             "check).")
    parser.add_argument("--queue-timeout", type=int, default=3,
                        help="Bars the resting limit order stays before being cancelled.")
    parser.add_argument("--model", type=str, default=os.environ.get('ANTHROPIC_MODEL', MODEL),
                        help="Claude model ID to call, e.g. claude-sonnet-5, "
                             "claude-opus-5, claude-haiku-4-5-20251001. Defaults to the "
                             "ANTHROPIC_MODEL env var if set, else claude-sonnet-5.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Override the Anthropic API endpoint (default: the "
                             "ANTHROPIC_BASE_URL env var if set, else the real "
                             "https://api.anthropic.com). Only needed if you're routing "
                             "through a different Anthropic-compatible gateway/proxy.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of NEW (uncached) LLM calls this run makes. "
                             "Each call costs real money and real time -- start small.")
    parser.add_argument("--cache", type=str, default="llm_gate_backtest_cache.csv",
                        help="Filename under data/ where decisions are cached, keyed by "
                             "(asset, signal_time), so a re-run never re-pays for an "
                             "already-decided candidate.")
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Only count an 'approve' decision as approved if the LLM's "
                             "reported confidence is >= this (0-1). Raising it trades fewer, "
                             "hopefully higher-quality, setups -- same precision/threshold "
                             "tradeoff as src/calibration.py's traditional-ML gate.")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max output tokens per decision call. Raise this if a "
                             "thinking-capable model/gateway is burning its whole budget "
                             "on reasoning and never emitting the JSON answer (shows up as "
                             "reason='no text in response ... stop_reason=max_tokens' in "
                             "the cache).")
    args = parser.parse_args()

    if args.signal == 'funding_reversion' and not args.funding_data:
        raise SystemExit(
            "--signal funding_reversion requires --funding-data (one funding-rate file per "
            "--data asset, in the same order).")

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. This script calls the real Claude API once per "
            "candidate trade.")
    client = anthropic.Anthropic(api_key=api_key, base_url=args.base_url)

    print(f"Gathering {args.signal} candidates from {len(args.data)} file(s)...")
    market, features_by_key, per_asset = gather_candidates(
        args.data, signal=args.signal, lookback=args.lookback, pt_mult=args.pt_mult,
        sl_mult=args.sl_mult, max_holding=args.max_holding, funding_data_files=args.funding_data)
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
                                     features_by_key[key], max_tokens=args.max_tokens,
                                     signal_name=args.signal)
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
    confidence_map = dict(zip(zip(cache['asset'], cache['signal_time']), cache['confidence']))
    market = market.copy()
    market['key'] = list(zip(market['asset'], market.index))
    market['llm_decision'] = market['key'].map(decision_map)
    market['llm_confidence'] = market['key'].map(confidence_map)

    decided = market[market['llm_decision'].notna()]
    if decided.empty:
        print("No decisions available yet -- increase --limit or re-run.")
        return

    approved_mask = ((decided['llm_decision'] == 'approve') &
                     (decided['llm_confidence'] >= args.min_confidence)).to_numpy()
    labels = decided['label'].to_numpy()
    rets = decided['ret'].to_numpy()
    n, n_approved = len(decided), int(approved_mask.sum())

    print(f"\n{'=' * 78}")
    print(f"  LLM gate results ({args.signal}) on {n} decided candidates "
          f"({n_approved} approved at confidence >= {args.min_confidence:.2f}, "
          f"{n - n_approved} rejected)")
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

    print(f"\n{'=' * 78}")
    print("  Realistic maker-fill execution on the APPROVED subset only (README section 9's "
          "OHLC-only queue simulation, an OPTIMISTIC upper bound -- not the instant-fill-at-"
          "close basis the table above uses)")
    print(f"{'=' * 78}")
    if n_approved > 0:
        mf = maker_fill_economics(decided[approved_mask], per_asset, maker_cost=0.0008,
                                  offset_mult=args.offset_mult, queue_timeout=args.queue_timeout,
                                  pt_mult=args.pt_mult, sl_mult=args.sl_mult,
                                  max_holding=args.max_holding)
        if mf['n_filled'] > 0:
            print(f"  fill rate: {mf['fill_rate']:.1%}  ({mf['n_filled']}/{mf['n_candidates']} "
                  f"approved candidates filled within {args.queue_timeout} bars)")
            print(f"  MAKER cost 0.08% -- FILLED: n={mf['n_filled']:4d}  win rate "
                  f"{mf['win_rate']:.1%}  PF {mf['pf']:.2f}  exp/trade {mf['expectancy']:+.4%}  "
                  f"bootstrap p={mf['p_value']:.4f}")
        else:
            print("  none of the approved candidates would have filled within "
                  f"{args.queue_timeout} bars in this simulation.")
    else:
        print("  no approved candidates to simulate.")

    out_path = os.path.join(OUTPUT_DIR, 'llm_gate_backtest_results.csv')
    decided.drop(columns=['key']).to_csv(out_path)
    print(f"\n  Full per-candidate detail saved to {out_path}")


if __name__ == "__main__":
    main()
