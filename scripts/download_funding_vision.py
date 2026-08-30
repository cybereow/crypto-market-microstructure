"""Download historical funding rate from Binance's public data archive
(data.binance.vision) — a static CDN, not the live trading API, so it isn't
subject to the same geo-restrictions and is far cheaper to pull in bulk.

Funding rate is an alternative-data feature: persistently extreme positive
funding (longs paying shorts heavily) has historically preceded local tops
in crypto perpetuals, and vice versa — a mean-reversion signal that isn't
derived from price/volume alone, so it isn't already fully priced into the
technical indicators every other participant is also computing.
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

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate"


def month_range(start: datetime, end: datetime):
    cur = start.replace(day=1)
    while cur <= end:
        yield cur
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)


def download_month(symbol: str, month: datetime) -> pd.DataFrame:
    month_str = month.strftime("%Y-%m")
    url = f"{BASE_URL}/{symbol}/{symbol}-fundingRate-{month_str}.zip"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return pd.DataFrame()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    return df


def main():
    parser = argparse.ArgumentParser(description="Download historical funding rate from data.binance.vision.")
    parser.add_argument("--symbol", type=str, required=True, help="e.g. BTCUSDT (no slash)")
    parser.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", type=str, default=None,
                         help="Output filename; defaults to binance_funding_<SYMBOL>.csv")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    frames = []
    for month in month_range(start, end):
        print(f"Fetching {args.symbol} funding rate for {month.strftime('%Y-%m')}...")
        df = download_month(args.symbol, month)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("No funding rate data downloaded.")
        sys.exit(1)

    result = pd.concat(frames, ignore_index=True)
    ts_col = 'calc_time' if 'calc_time' in result.columns else result.columns[0]
    rate_col = 'last_funding_rate' if 'last_funding_rate' in result.columns else result.columns[-1]

    result['timestamp'] = pd.to_datetime(result[ts_col], unit='ms')
    result = result[['timestamp', rate_col]].rename(columns={rate_col: 'funding_rate'})
    result = result.drop_duplicates(subset='timestamp').set_index('timestamp').sort_index()

    # Funding is usually published every 8h; forward-fill to hourly so it can
    # be reindexed onto any OHLCV timeframe downstream without gaps.
    result = result.resample('h').ffill()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = args.out or f"binance_funding_{args.symbol}.csv"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    result.to_csv(out_path)
    print(f"Saved {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
