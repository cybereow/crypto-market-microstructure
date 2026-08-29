import argparse
import os
import sys
import requests
import zipfile
import tempfile
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR

def process_bookticker_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Calculate Order Book Imbalance (OBI) for a chunk of bookTicker data."""
    # Binance bookTicker format typically: update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time, event_time
    # Since headers might vary or be missing, we ensure we have transaction_time and quantities

    # Try to map columns if standard headers exist, otherwise use indices
    # UM Futures bookTicker CSV standard:
    # update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time, event_time
    if len(chunk.columns) >= 7:
        if 'best_bid_qty' not in chunk.columns:
            # Assume no header, map manually
            chunk.columns = ['update_id', 'best_bid_price', 'best_bid_qty', 'best_ask_price', 'best_ask_qty', 'transaction_time', 'event_time'] + list(chunk.columns[7:])

    # Calculate OBI: Bid Qty / (Bid Qty + Ask Qty)
    # 0.5 is balanced. > 0.5 is bid-heavy (bullish). < 0.5 is ask-heavy (bearish).
    bid_qty = chunk['best_bid_qty'].astype(float)
    ask_qty = chunk['best_ask_qty'].astype(float)

    chunk['obi'] = bid_qty / (bid_qty + ask_qty + 1e-9)

    # Convert time to datetime
    time_col = 'transaction_time' if 'transaction_time' in chunk.columns else chunk.columns[5]
    chunk['timestamp'] = pd.to_datetime(chunk[time_col], unit='ms')

    return chunk[['timestamp', 'obi']]

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

    print("Download successful. Saving to temporary file and processing in chunks to save RAM...")

    # Save to a temporary file instead of loading into RAM
    fd, temp_zip_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)

    with open(temp_zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    aggregated_chunks = []
    try:
        with zipfile.ZipFile(temp_zip_path) as z:
            csv_filename = f"{file_name}.csv"
            if csv_filename not in z.namelist():
                csv_filename = z.namelist()[0] # Fallback to first file

            with z.open(csv_filename) as f:
                # Process in chunks of 100,000 rows
                chunk_iter = pd.read_csv(f, chunksize=100000, low_memory=False)
                for i, chunk in enumerate(chunk_iter):
                    processed = process_bookticker_chunk(chunk)

                # Resample chunk to the target timeframe to compress memory immediately
                processed.set_index('timestamp', inplace=True)

                # Pandas resample rule mapping
                rule_map = {'1d': 'D', '1h': 'h', '15m': '15min', '5m': '5min', '1m': '1min'}
                rule = rule_map.get(timeframe, 'D')

                # Calculate sum and count for each chunk to avoid mean-of-means bias later
                resampled_sum = processed.resample(rule)['obi'].sum().rename('obi_sum')
                resampled_count = processed.resample(rule)['obi'].count().rename('obi_count')
                resampled = pd.concat([resampled_sum, resampled_count], axis=1)

                aggregated_chunks.append(resampled)

                if i % 10 == 0 and i > 0:
                    print(f"  Processed {i * 100000} tick events...")
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

    print("Finalizing aggregation...")
    # Combine all resampled chunks and group by index to get the true mean per timeframe
    final_df = pd.concat(aggregated_chunks)
    final_agg = final_df.groupby(final_df.index).sum()

    # Calculate the mathematically correct mean
    result_df = pd.DataFrame(index=final_agg.index)
    result_df['obi'] = final_agg['obi_sum'] / final_agg['obi_count']

    return result_df

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
