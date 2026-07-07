"""
SQLite storage for spread snapshots collected while the live scanner runs.

There is no public historical archive for cross-exchange IRT spreads (unlike
Binance's funding-rate CDN) -- the only way to get a historical dataset for
this strategy is to run scripts/scan_irt_arb.py and let it accumulate real
snapshots over time.
"""
import sqlite3

import pandas as pd

from . import config


class Storage:
    def __init__(self, db_path=None):
        self.db = sqlite3.connect(db_path or config.DB_PATH)
        self._init_tables()

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS spread_snapshots (
                pair TEXT, timestamp INTEGER,
                tabdeal_bid REAL, tabdeal_ask REAL,
                nobitex_bid REAL, nobitex_ask REAL,
                net_edge_bps REAL, direction TEXT
            );
        """)
        self.db.commit()

    def save_snapshot(self, snapshot: dict):
        self.db.execute(
            "INSERT INTO spread_snapshots VALUES (?,?,?,?,?,?,?,?)",
            (
                snapshot["pair"], snapshot["timestamp"],
                snapshot["tabdeal_bid"], snapshot["tabdeal_ask"],
                snapshot["nobitex_bid"], snapshot["nobitex_ask"],
                snapshot["net_edge_bps"], snapshot["direction"],
            )
        )
        self.db.commit()

    def get_snapshots(self, pair, start=None, end=None) -> pd.DataFrame:
        query = "SELECT * FROM spread_snapshots WHERE pair=?"
        params = [pair]
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        query += " ORDER BY timestamp"
        return pd.read_sql(query, self.db, params=params)
