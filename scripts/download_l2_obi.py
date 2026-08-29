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

def process_bookticker_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Extract bid/ask quantities for a chunk of bookTicker data."""
    if len(chunk.columns) >= 7:
        if 'best_bid_qty' not in chunk.columns:
            # Assume no header, map manually
            chunk.columns = ['update_id', 'best_bid_price', 'best_bid_qty', 'best_ask_price', 'best_ask_qty', 'transaction_time', 'event_time'] + list(chunk.columns[7:])

    bid_qty = chunk['best_bid_qty'].astype(float)
    ask_qty = chunk['best_ask_qty'].astype(float)

    chunk['bid_qty'] = bid_qty
    chunk['ask_qty'] = ask_qty

    time_col = 'transaction_time' if 'transaction_time' in chunk.columns else chunk.columns[5]
    chunk['timestamp'] = pd.to_datetime(chunk[time_col], unit='ms')

    return chunk[['timestamp', 'bid_qty', 'ask_qty']]

def download_and_process_l2(symbol: str, date: str, timeframe: str = '1d') -> pd.DataFrame:
    """Download daily bookTicker ZIP from Binance Vision, process in chunks, and return aggregated OBI."""
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

                    rule_map = {'1d': 'D', '1h': 'h', '15m': '15min', '5m': '5min', '1m': '1min'}
                    rule = rule_map.get(timeframe, 'D')

                    # Sum quantities instead of averaging OBI to prevent Mean-of-Means bias
                    resampled = processed.resample(rule).sum()
                    aggregated_chunks.append(resampled)

                    if i % 10 == 0 and i > 0:
                        print(f"  Processed {i * 100000} tick events...")
    finally:
        os.remove(temp_zip_path)

    print("Finalizing aggregation...")
    if not aggregated_chunks:
        return pd.DataFrame()

    # Combine all chunk sums and add them together by index
    final_sums = pd.concat(aggregated_chunks)
    final_sums = final_sums.groupby(final_sums.index).sum()

    # Calculate the true mathematical OBI for the final timeframe
    final_sums['obi'] = final_sums['bid_qty'] / (final_sums['bid_qty'] + final_sums['ask_qty'] + 1e-9)

    return final_sums[['obi']]

def main():
    parser = argparse.ArgumentParser(description="Download and process Binance L2 BookTicker Data for OBI.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol (e.g. BTCUSDT without slash for Binance Vision)")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1d", help="Aggregation timeframe (e.g. 1d, 1h, 5m)")
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

    result_df = pd.concat(all_obi)
    # Ensure no duplicates in case of overlapping chunk boundaries
    result_df = result_df.groupby(result_df.index).mean()

    safe_symbol = args.symbol
    filename = f"binance_l2obi_{safe_symbol}_{args.timeframe}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    result_df.to_csv(filepath)
    print(f"OBI Feature data saved to {filepath} ({len(result_df)} rows)")

if __name__ == "__main__":
    main()
