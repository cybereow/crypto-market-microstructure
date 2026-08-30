import argparse
import os
import sys
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR

import tempfile

RULE_MAP = {'1d': 'D', '1h': 'h', '15m': '15min', '5m': '5min', '1m': '1min',
            '30s': '30s', '10s': '10s', '5s': '5s', '1s': 's'}


def process_bookticker_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Extract bid/ask quantities AND the top-of-book mid price for a chunk
    of bookTicker data. The mid price matters at sub-minute granularity: no
    kline archive exists below 1-minute resolution, so a tick-level OBI
    test needs its own price series, and bookTicker already carries one
    (best_bid_price/best_ask_price on every update) -- reusing it avoids a
    second, separately-timestamped download.
    """
    if len(chunk.columns) >= 7:
        if 'best_bid_qty' not in chunk.columns:
            # Assume no header, map manually
            chunk.columns = ['update_id', 'best_bid_price', 'best_bid_qty', 'best_ask_price', 'best_ask_qty', 'transaction_time', 'event_time'] + list(chunk.columns[7:])

    bid_price = chunk['best_bid_price'].astype(float)
    ask_price = chunk['best_ask_price'].astype(float)
    chunk['bid_qty'] = chunk['best_bid_qty'].astype(float)
    chunk['ask_qty'] = chunk['best_ask_qty'].astype(float)
    chunk['mid'] = (bid_price + ask_price) / 2.0

    time_col = 'transaction_time' if 'transaction_time' in chunk.columns else chunk.columns[5]
    chunk['timestamp'] = pd.to_datetime(chunk[time_col], unit='ms')

    return chunk[['timestamp', 'bid_qty', 'ask_qty', 'mid']]


def _resample_chunk(processed: pd.DataFrame, rule: str) -> pd.DataFrame:
    """One chunk's worth of ticks -> one bucket per bar, quantities summed
    (OBI's mean-of-means fix) and the mid price turned into an OHLC-style
    bar (first/max/min/last). Both kinds of aggregate are associative
    across chunks/days when chunks are processed in chronological order
    (guaranteed by pd.read_csv(chunksize=...) reading the archive
    sequentially): re-summing per-chunk sums reproduces the true sum,
    and taking first-of-firsts / max-of-maxes / min-of-mins / last-of-lasts
    over chronologically ordered chunks reproduces the true first/high/
    low/last -- so the same combine step in `main()` works for both.
    """
    agg = processed.resample(rule).agg({
        'bid_qty': ['sum'], 'ask_qty': ['sum'],
        'mid': ['first', 'max', 'min', 'last'],
    })
    agg.columns = ['_'.join(c) for c in agg.columns]
    return agg


def download_and_process_l2(symbol: str, date: str, timeframe: str = '1d') -> pd.DataFrame:
    """Download daily bookTicker ZIP from Binance Vision, process in chunks, and return aggregated OBI + OHLC."""
    base_url = "https://data.binance.vision/data/futures/um/daily/bookTicker"
    file_name = f"{symbol}-bookTicker-{date}"
    url = f"{base_url}/{symbol}/{file_name}.zip"

    print(f"Fetching L2 Data: {url}")
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f"Failed to download {file_name}. Status: {response.status_code}")
        return pd.DataFrame()

    print("Download successful. Processing ZIP stream in chunks to save RAM...")

    # Save stream to temporary file to prevent loading full ZIP into RAM (OOM fix)
    temp_zip_fd, temp_zip_path = tempfile.mkstemp(suffix='.zip')
    with os.fdopen(temp_zip_fd, 'wb') as f:
        for zip_chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
            if zip_chunk:
                f.write(zip_chunk)

    aggregated_chunks = []
    rule = RULE_MAP.get(timeframe, 'D')

    try:
        with zipfile.ZipFile(temp_zip_path) as z:
            csv_filename = f"{file_name}.csv"
            if csv_filename not in z.namelist():
                csv_filename = z.namelist()[0]

            with z.open(csv_filename) as f:
                chunk_iter = pd.read_csv(f, chunksize=100000, low_memory=False)
                for i, chunk in enumerate(chunk_iter):
                    processed = process_bookticker_chunk(chunk)
                    processed.set_index('timestamp', inplace=True)
                    aggregated_chunks.append(_resample_chunk(processed, rule))

                    if i % 10 == 0 and i > 0:
                        print(f"  Processed {i * 100000} tick events...")
    finally:
        os.remove(temp_zip_path)

    print("Finalizing aggregation...")
    if not aggregated_chunks:
        return pd.DataFrame()

    combined = pd.concat(aggregated_chunks)
    final = combined.groupby(combined.index).agg({
        'bid_qty_sum': 'sum', 'ask_qty_sum': 'sum',
        'mid_first': 'first', 'mid_max': 'max', 'mid_min': 'min', 'mid_last': 'last',
    })
    final = final.rename(columns={
        'bid_qty_sum': 'bid_qty', 'ask_qty_sum': 'ask_qty',
        'mid_first': 'open', 'mid_max': 'high', 'mid_min': 'low', 'mid_last': 'close',
    })
    return final[['bid_qty', 'ask_qty', 'open', 'high', 'low', 'close']]

def main():
    parser = argparse.ArgumentParser(description="Download and process Binance L2 BookTicker Data for OBI.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol (e.g. BTCUSDT without slash for Binance Vision)")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1d",
                         help=f"Aggregation timeframe: {', '.join(RULE_MAP.keys())}")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    current = start
    all_obi = []

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        daily_obi = download_and_process_l2(args.symbol, date_str, args.timeframe)
        if not daily_obi.empty:
            all_obi.append(daily_obi)
        current += timedelta(days=1)

    if not all_obi:
        print("No data was successfully downloaded and processed.")
        sys.exit(1)

    # Combine all days. A bar can straddle a day boundary (e.g. a 5s bucket
    # spanning 23:59:57-00:00:02, split across two daily archives), so this
    # needs the SAME per-column combine as the within-day chunk merge: sum
    # for quantities, first/max/min/last for the OHLC mid-price columns --
    # summing price columns here would silently corrupt every boundary bar.
    result_df = pd.concat(all_obi).sort_index()
    result_df = result_df.groupby(result_df.index).agg({
        'bid_qty': 'sum', 'ask_qty': 'sum',
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    })

    # Calculate the true mathematical OBI for the final dataset
    result_df['obi'] = result_df['bid_qty'] / (result_df['bid_qty'] + result_df['ask_qty'] + 1e-9)

    # Keep OHLC alongside obi: at sub-minute granularity no separate kline
    # archive exists to pair with it, so this file doubles as the price
    # series too (see backtest_maker_fill.py --data / --obi-data, both of
    # which can point at this same file for a sub-minute run).
    final_feature_df = result_df[['open', 'high', 'low', 'close', 'obi']]

    safe_symbol = args.symbol
    filename = f"binance_l2obi_{safe_symbol}_{args.timeframe}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    final_feature_df.to_csv(filepath)
    print(f"OBI Feature data saved to {filepath} ({len(final_feature_df)} rows)")

if __name__ == "__main__":
    main()
