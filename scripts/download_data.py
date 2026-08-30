import argparse
import os
import sys

import ccxt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR

def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int = 1000, since: str = None) -> pd.DataFrame:
    """Fetch OHLCV data from an exchange using ccxt."""
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
    except AttributeError:
        print(f"Error: Exchange '{exchange_id}' not found in ccxt.")
        sys.exit(1)

    since_ms = None
    if since:
        since_ms = exchange.parse8601(since + 'T00:00:00Z')

    all_ohlcv = []

    # Pagination loop
    while len(all_ohlcv) < limit:
        remaining_limit = limit - len(all_ohlcv)
        fetch_limit = min(remaining_limit, 1000) # Most exchanges limit 1000 per request

        try:
            print(f"Fetching {fetch_limit} bars of {timeframe} data for {symbol} from {exchange_id} (since: {exchange.iso8601(since_ms) if since_ms else 'latest'})...")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=fetch_limit)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

        if not ohlcv:
            print("No more data available from exchange.")
            break

        all_ohlcv.extend(ohlcv)

        # Determine the next `since` timestamp
        # The last candle's timestamp + 1 ms to avoid fetching the same candle again
        last_candle_ts = ohlcv[-1][0]
        since_ms = last_candle_ts + 1

        # If we got less than requested limit, we've likely hit the end of available history
        if len(ohlcv) < fetch_limit:
            print("Reached the end of available history.")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def fetch_funding_rates(exchange_id: str, symbol: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch historical funding rates from an exchange using ccxt."""
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
    except AttributeError:
        print(f"Error: Exchange '{exchange_id}' not found in ccxt.")
        return pd.DataFrame()

    if not exchange.has['fetchFundingRateHistory']:
        print(f"Warning: Exchange '{exchange_id}' does not support fetchFundingRateHistory via ccxt API.")
        return pd.DataFrame()

    try:
        print(f"Fetching historical funding rates for {symbol} from {exchange_id}...")
        funding = exchange.fetch_funding_rate_history(symbol, limit=limit)
    except Exception as e:
        print(f"Warning fetching funding data: {e}. Funding rates will not be included.")
        return pd.DataFrame()

    if not funding:
        return pd.DataFrame()

    df = pd.DataFrame(funding)
    if 'timestamp' in df.columns and 'fundingRate' in df.columns:
        df = df[['timestamp', 'fundingRate']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Funding is usually paid every 8h. We round down to the nearest day to merge with daily OHLCV,
        # or just keep raw if matching by exact hour. For generic matching we'll set index and resample later.
        df.set_index('timestamp', inplace=True)
        # Rename column for consistency
        df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
        return df

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data for machine learning/statistical arbitrage.")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange ID (e.g. binance, kraken)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe (e.g. 1d, 1h, 15m)")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candles to fetch (max limits apply per exchange)")
    parser.add_argument("--since", type=str, default=None, help="Start date in ISO format (e.g. 2022-01-01)")
    parser.add_argument("--include-funding", action="store_true", help="Attempt to fetch and merge funding rates (Requires Futures symbol e.g. BTC/USDT:USDT)")
    args = parser.parse_args()

    df = fetch_ohlcv(args.exchange, args.symbol, args.timeframe, args.limit, args.since)

    if args.include_funding:
        # For funding rates, ccxt usually requires the perpetual swap symbol format like "BTC/USDT:USDT" or "BTC/USD"
        # If the user passed a spot symbol but wants funding, we try the passed symbol.
        funding_df = fetch_funding_rates(args.exchange, args.symbol, args.limit)

        if not funding_df.empty:
            print("Merging funding rates into OHLCV data...")
            # If timeframe is 1d, resample funding to daily average
            if args.timeframe == '1d':
                funding_resampled = funding_df.resample('D').mean()
            else:
                funding_resampled = funding_df

            # Merge with exact index matching or forward fill
            df = df.join(funding_resampled, how='left')
            # Forward fill the funding rate because if no new funding rate is declared, the last one applies
            df['funding_rate'] = df['funding_rate'].ffill().fillna(0)
            print(f"Successfully merged funding rates. Non-zero funding records: {(df['funding_rate'] != 0).sum()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_symbol = args.symbol.replace("/", "_")
    filename = f"{args.exchange}_{safe_symbol}_{args.timeframe}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    df.to_csv(filepath)
    print(f"Data saved to {filepath} ({len(df)} rows)")

if __name__ == "__main__":
    main()
