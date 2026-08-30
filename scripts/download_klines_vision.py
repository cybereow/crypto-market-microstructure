"""Download deep OHLCV history from Binance's public data archive
(data.binance.vision) instead of a live exchange REST API.

Why this exists: the live-API downloader (download_data.py) is capped by
whatever each exchange serves. Kraken in particular returns only ~720 4h
bars regardless of the requested `--limit`/`--since` (about 4 months), and
binance.com's REST API answers HTTP 451 from many regions. Meta-labeling
needs *trades*, not bars: 720 bars produce on the order of 40-60 Donchian
breakout candidates per asset, which is far too few to fit a model on, and
is exactly why single-asset meta-label backtests in this repo swing wildly
between assets.

data.binance.vision is a static CDN of monthly kline archives going back to
each symbol's listing, is not geo-restricted the way the trading API is, and
needs no API key. Pulling 2020-present at 4h yields ~10k bars per asset
(~14x Kraken), which is the single highest-leverage change available for
model quality — no amount of feature engineering fixes a 60-sample fit.

Output schema matches download_data.py exactly (timestamp index + open,
high, low, close, volume), so every downstream script consumes it unchanged.
"""
import argparse
import io
import os
import sys
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR

BASE_URLS = {
    'spot': "https://data.binance.vision/data/spot/monthly/klines",
    # USD-M perpetual futures. Needed to match instruments with the L2 OBI
    # downloader (download_l2_obi.py), which only has a futures bookTicker
    # archive to draw from -- pairing it with spot klines would compare a
    # futures order-flow signal against a different instrument's price.
    'futures': "https://data.binance.vision/data/futures/um/monthly/klines",
}

# The archive's CSVs are headerless; these are the 12 documented columns.
KLINE_COLS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore',
]


def month_range(start: datetime, end: datetime):
    cur = start.replace(day=1)
    while cur <= end:
        yield cur
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)


def download_month(symbol: str, timeframe: str, month: datetime, market: str = 'spot') -> pd.DataFrame:
    """Fetch one monthly archive. Returns an empty frame for months before
    the symbol listed (the CDN 404s), which the caller treats as 'skip'.
    """
    month_str = month.strftime("%Y-%m")
    base_url = BASE_URLS[market]
    url = f"{base_url}/{symbol}/{timeframe}/{symbol}-{timeframe}-{month_str}.zip"
    try:
        resp = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        print(f"  {month_str}: request failed ({exc}), skipping.")
        return pd.DataFrame()
    if resp.status_code != 200:
        return pd.DataFrame()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLS)

    # Some months ship WITH a header row (inconsistent across the archive),
    # which read_csv turns into a non-numeric first data row. Drop those
    # BEFORE any numeric comparison on open_time -- a string first row
    # (e.g. futures archives, which hit this every month) breaks a
    # magnitude check done first.
    df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
    df = df[df['open_time'].notna()]

    # Binance switched open_time from milliseconds to microseconds in the
    # 2025 archives. Detect by magnitude rather than by date so this keeps
    # working for whatever they do next: ms timestamps for any plausible
    # trading date are ~1e12, us timestamps are ~1e15.
    if not df.empty and df['open_time'].iloc[0] > 1e14:
        df['open_time'] = df['open_time'] // 1000

    return df


def fetch_klines(symbol: str, timeframe: str, start: datetime, end: datetime, market: str = 'spot') -> pd.DataFrame:
    frames = []
    for month in month_range(start, end):
        df = download_month(symbol, timeframe, month, market=market)
        if df.empty:
            continue
        frames.append(df)
        print(f"  {month.strftime('%Y-%m')}: {len(df)} bars")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out['timestamp'] = pd.to_datetime(out['open_time'].astype('int64'), unit='ms')
    out = out[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(
        {'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
    )
    out = out.drop_duplicates(subset='timestamp').sort_values('timestamp')
    out = out.set_index('timestamp')
    return out[(out.index >= start) & (out.index <= end)]


def main():
    parser = argparse.ArgumentParser(
        description="Download deep OHLCV history from data.binance.vision.")
    parser.add_argument("--symbol", type=str, required=True,
                        help="Archive symbol without a slash, e.g. BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="4h", help="e.g. 4h, 1h, 1d")
    parser.add_argument("--market", type=str, default="spot", choices=list(BASE_URLS.keys()),
                        help="'futures' pulls USD-M perpetual klines instead of spot -- use "
                             "this to pair prices with download_l2_obi.py, which only has a "
                             "futures order-book archive.")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None,
                        help="YYYY-MM-DD (default: today)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV filename. Default mirrors download_data.py's "
                             "naming so downstream scripts need no changes.")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.utcnow()

    print(f"Downloading {args.market} {args.symbol} {args.timeframe} from {start.date()} to {end.date()}")
    df = fetch_klines(args.symbol, args.timeframe, start, end, market=args.market)

    if df.empty:
        print("Error: no data downloaded (check the symbol spelling and date range).")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.out:
        filename = args.out
    else:
        # BTCUSDT -> binance_BTC_USDT_4h.csv (or binance_futures_BTC_USDT_4h.csv)
        base = args.symbol[:-4] if args.symbol.endswith("USDT") else args.symbol
        prefix = "binance_futures" if args.market == "futures" else "binance"
        filename = f"{prefix}_{base}_USDT_{args.timeframe}.csv"

    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath)
    print(f"\nSaved {len(df)} bars ({df.index[0]} to {df.index[-1]}) to {filepath}")


if __name__ == "__main__":
    main()
