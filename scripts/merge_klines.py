"""Merge externally-supplied OHLCV CSVs into the existing data/ files.

Why this exists: data.binance.vision only publishes a *monthly* archive, so
the most recent (incomplete) month is never available there. download_klines_vision.py
therefore always stops at the end of the last completed month, leaving a gap
of up to ~31 days between the archive and the present. That gap is precisely
the most recent, most relevant out-of-sample data.

A live API pull (or a manual export from the Binance UI) can cover the gap.
This script splices such a file onto the archive safely.

The important word is *safely*. Naively concatenating two price series is a
classic way to poison a dataset: the two sources may quote different symbols
(spot vs perp), different quote assets, adjusted vs unadjusted prices, or a
different timezone convention. Any of those creates an artificial jump at the
seam, and a breakout/reversion strategy will happily "discover" that jump as
a tradable edge. So before writing anything, this script requires that the
two series *agree on the bars they share*, and aborts if they do not.

Usage:
    python scripts/merge_klines.py --new-dir /home/user/uploaded_files
    python scripts/merge_klines.py --new-dir ./incoming --dry-run
"""
import argparse
import glob
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR

REQUIRED_COLS = ['open', 'high', 'low', 'close', 'volume']


def load_ohlcv(path):
    """Read an OHLCV CSV in this repo's schema: timestamp index + OHLCV."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing columns: {missing}")
    df = df.set_index('timestamp').sort_index()
    # Duplicate timestamps within a single source are a data error, not a
    # merge concern; resolve them here so the seam check sees clean input.
    df = df[~df.index.duplicated(keep='last')]
    return df[REQUIRED_COLS]


def compare_overlap(old, new, tolerance):
    """Quantify agreement on shared timestamps.

    Returns (n_overlap, max_rel_diff, worst_timestamp). A tolerance is allowed
    rather than demanding bit-equality because a live API and an archive can
    legitimately differ in float formatting or final-decimal rounding.
    """
    shared = old.index.intersection(new.index)
    if len(shared) == 0:
        return 0, None, None

    o = old.loc[shared, 'close'].astype(float)
    n = new.loc[shared, 'close'].astype(float)
    # Relative difference, guarded against division by zero on any zero prints.
    denom = o.abs().where(o.abs() > 1e-12, 1.0)
    rel = (n - o).abs() / denom
    return len(shared), float(rel.max()), rel.idxmax()


def merge_one(old_path, new_path, tolerance, dry_run, backup):
    """Splice `new_path` onto `old_path`, refusing to write if the seam is bad."""
    name = os.path.basename(old_path)
    new = load_ohlcv(new_path)

    if not os.path.exists(old_path):
        # Nothing to reconcile: the incoming file simply becomes the dataset.
        print(f"{name:32s} NEW FILE   {len(new):6d} bars "
              f"({new.index[0]} -> {new.index[-1]})")
        if not dry_run:
            new.to_csv(old_path)
        return {'status': 'created', 'added': len(new)}

    old = load_ohlcv(old_path)
    n_overlap, max_diff, worst = compare_overlap(old, new, tolerance)

    # Bars strictly newer than anything we already hold.
    fresh = new[new.index > old.index.max()]
    # Bars older than our history start (back-fill; rare but harmless).
    prefix = new[new.index < old.index.min()]

    if n_overlap == 0:
        # No shared bars means no way to verify the two series are the same
        # instrument. Refusing is the conservative choice: a silent splice of
        # a mismatched series is far more expensive than a manual override.
        print(f"{name:32s} REFUSED    no overlapping bars to verify against "
              f"(old ends {old.index.max()}, new starts {new.index.min()})")
        return {'status': 'refused_no_overlap', 'added': 0}

    if max_diff > tolerance:
        print(f"{name:32s} REFUSED    overlap disagrees: max rel diff "
              f"{max_diff:.6%} at {worst} (tolerance {tolerance:.4%}) "
              f"-- likely a different symbol/market, not a top-up")
        return {'status': 'refused_mismatch', 'added': 0}

    if len(fresh) == 0 and len(prefix) == 0:
        print(f"{name:32s} NO-OP      {n_overlap} bars verified identical, "
              f"nothing new (already covers {new.index[-1]})")
        return {'status': 'noop', 'added': 0}

    merged = pd.concat([prefix, old, fresh]).sort_index()
    merged = merged[~merged.index.duplicated(keep='last')]

    print(f"{name:32s} MERGED     +{len(fresh)} new"
          f"{f' +{len(prefix)} backfill' if len(prefix) else ''}"
          f"  (verified {n_overlap} shared bars, max diff {max_diff:.6%})  "
          f"-> {len(merged)} bars, ends {merged.index[-1]}")

    if not dry_run:
        if backup and not os.path.exists(old_path + '.bak'):
            shutil.copy2(old_path, old_path + '.bak')
        merged.to_csv(old_path)

    return {'status': 'merged', 'added': len(fresh) + len(prefix)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new-dir", type=str, required=True,
                        help="Directory containing incoming OHLCV CSVs to splice in.")
    parser.add_argument("--pattern", type=str, default="binance_*_4h.csv",
                        help="Glob for incoming files (default: binance_*_4h.csv).")
    parser.add_argument("--data-dir", type=str, default=OUTPUT_DIR,
                        help="Existing dataset directory (default: config OUTPUT_DIR).")
    parser.add_argument("--tolerance", type=float, default=0.001,
                        help="Max allowed relative close diff on shared bars "
                             "before refusing to merge (default 0.001 = 0.1%%).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip writing a .bak copy of each modified file.")
    args = parser.parse_args()

    incoming = sorted(glob.glob(os.path.join(args.new_dir, args.pattern)))
    if not incoming:
        print(f"No files matched {args.pattern} in {args.new_dir}")
        sys.exit(1)

    print(f"Merging {len(incoming)} file(s) into {args.data_dir}"
          f"{'  [DRY RUN]' if args.dry_run else ''}\n")

    results = []
    for new_path in incoming:
        old_path = os.path.join(args.data_dir, os.path.basename(new_path))
        try:
            results.append(merge_one(old_path, new_path, args.tolerance,
                                     args.dry_run, not args.no_backup))
        except Exception as exc:
            print(f"{os.path.basename(new_path):32s} ERROR      {exc}")
            results.append({'status': 'error', 'added': 0})

    total_added = sum(r['added'] for r in results)
    refused = [r for r in results if r['status'].startswith('refused') or r['status'] == 'error']
    print(f"\nTotal bars added: {total_added}")
    if refused:
        print(f"WARNING: {len(refused)} file(s) were refused or errored -- see above.")
        sys.exit(2)


if __name__ == "__main__":
    main()
