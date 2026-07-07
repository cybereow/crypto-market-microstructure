"""
SQLite storage for LIVE-collected data (candles, orderbook snapshots, and
funding-rate snapshots polled while monitoring an open position).

This is separate from binance_funding.py / binance_klines.py, which pull
full historical archives for backtesting. Storage is for the paper/live
phases (v0.3+), where the collector polls the exchange over time.
"""
import sqlite3

import pandas as pd

from . import config


class Storage:
    def __init__(self, db_path=None):
        self.db = sqlite3.connect(db_path or config.DB_PATH)

    def get_candles(self, symbol, timeframe="1h", start=None, end=None):
        query = "SELECT * FROM candles WHERE symbol=? AND timeframe=?"
        params = [symbol, timeframe]
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        query += " ORDER BY timestamp"
        return pd.read_sql(query, self.db, params=params)

    def get_funding_snapshots(self, symbol):
        return pd.read_sql(
            "SELECT * FROM funding WHERE symbol=? ORDER BY timestamp",
            self.db, params=[symbol]
        )
