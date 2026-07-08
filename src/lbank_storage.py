"""
SQLite storage for LBank funding-rate snapshots. LBank's own market-data
endpoint only exposes the CURRENT funding rate per symbol, not history --
same situation as Tabdeal's triangular data, so this self-collects a
dataset over time via scripts/scan_lbank_funding.py.
"""
import os
import sqlite3

import pandas as pd

from . import config


class LBankStorage:
    def __init__(self, db_path=None):
        db_path = db_path or config.LBANK_DB_PATH
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS lbank_funding_snapshots (
                symbol TEXT, timestamp INTEGER,
                funding_rate REAL, funding_interval_seconds INTEGER,
                marked_price REAL, turnover_24h REAL
            );
        """)
        try:
            self.db.execute("ALTER TABLE lbank_funding_snapshots ADD COLUMN turnover_24h REAL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # already migrated
        self.db.commit()

    def save_snapshots(self, rows: list[dict]):
        self.db.executemany(
            "INSERT INTO lbank_funding_snapshots VALUES (?,?,?,?,?,?)",
            [
                (r["symbol"], r["timestamp"], r["funding_rate"],
                 r["funding_interval_seconds"], r["marked_price"], r.get("turnover_24h", 0))
                for r in rows
            ]
        )
        self.db.commit()

    def get_snapshots(self, symbol=None) -> pd.DataFrame:
        query = "SELECT * FROM lbank_funding_snapshots"
        params = []
        if symbol:
            query += " WHERE symbol=?"
            params.append(symbol)
        query += " ORDER BY timestamp"
        return pd.read_sql(query, self.db, params=params)
