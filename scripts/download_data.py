import argparse
import os
import sys

import ccxt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR

def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV data from an exchange using ccxt."""
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
    except AttributeError:
        print(f"Error: Exchange '{exchange_id}' not found in ccxt.")
        sys.exit(1)

    try:
        print(f"Fetching {limit} bars of {timeframe} data for {symbol} from {exchange_id}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data for machine learning/statistical arbitrage.")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange ID (e.g. binance, kraken)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe (e.g. 1d, 1h, 15m)")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candles to fetch (max limits apply per exchange)")
    args = parser.parse_args()

    df = fetch_ohlcv(args.exchange, args.symbol, args.timeframe, args.limit)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_symbol = args.symbol.replace("/", "_")
    filename = f"{args.exchange}_{safe_symbol}_{args.timeframe}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    df.to_csv(filepath)
    print(f"Data saved to {filepath} ({len(df)} rows)")

if __name__ == "__main__":
    main()
