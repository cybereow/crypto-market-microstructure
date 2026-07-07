"""
Live data collection via ccxt, for the paper/live phases (v0.3+) only.

For historical funding-rate backtesting, do NOT use fetch_funding_rate()
below -- use binance_funding.load_symbol() instead, which reads the public
data.binance.vision archive. fetch_funding_rate() calls the live
fapi.binance.com REST endpoint, which is geo-blocked in some environments
and was the reason the original backtest attempt failed.
"""
import time

import ccxt

from . import config
from .storage import Storage


class Collector:
    def __init__(self, exchange_id=None, db_path=None):
        self.exchange = getattr(ccxt, exchange_id or config.EXCHANGE)({"enableRateLimit": True})
        self.storage = Storage(db_path)
        self._init_tables()

    def _init_tables(self):
        self.storage.db.executescript("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT, timeframe TEXT, timestamp INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            );
            CREATE TABLE IF NOT EXISTS funding (
                symbol TEXT, timestamp INTEGER, rate REAL,
                PRIMARY KEY (symbol, timestamp)
            );
            CREATE TABLE IF NOT EXISTS orderbook (
                symbol TEXT, timestamp INTEGER,
                bids TEXT, asks TEXT
            );
        """)

    def fetch_candles(self, symbol, timeframe=None, limit=500):
        data = self.exchange.fetch_ohlcv(symbol, timeframe or config.CANDLE_TIMEFRAME, limit=limit)
        self.storage.db.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?)",
            [(symbol, timeframe or config.CANDLE_TIMEFRAME, *c) for c in data]
        )
        self.storage.db.commit()
        return len(data)

    def fetch_funding_snapshot(self, symbol):
        """One live funding-rate reading, for monitoring an already-open
        position -- not a substitute for the historical archive used in
        backtesting. Requires an exchange/host that isn't geo-blocked from
        fapi.binance.com."""
        try:
            rate = self.exchange.fetch_funding_rate(symbol)
            self.storage.db.execute(
                "INSERT OR REPLACE INTO funding VALUES (?,?,?)",
                (symbol, int(time.time() * 1000), rate["fundingRate"])
            )
            self.storage.db.commit()
            return rate["fundingRate"]
        except Exception as e:
            return None

    def fetch_orderbook(self, symbol, depth=None):
        depth = depth or config.ORDERBOOK_DEPTH
        ob = self.exchange.fetch_order_book(symbol, limit=depth)
        import json
        self.storage.db.execute(
            "INSERT INTO orderbook VALUES (?,?,?,?)",
            (symbol, int(time.time() * 1000), json.dumps(ob["bids"][:depth]), json.dumps(ob["asks"][:depth]))
        )
        self.storage.db.commit()
        return ob
