"""
SQLite storage for triangular-arbitrage snapshots collected while the live
scanner runs. There is no public historical archive for these spreads --
the only way to get a historical dataset is to run
scripts/scan_triangular.py and let it accumulate real snapshots over time.
"""
import os
import sqlite3

import pandas as pd

from . import config


class Storage:
    def __init__(self, db_path=None):
        db_path = db_path or config.DB_PATH
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS triangular_snapshots (
                base_asset TEXT, timestamp INTEGER,
                x_usdt_bid REAL, x_usdt_ask REAL,
                x_irt_bid REAL, x_irt_ask REAL,
                usdt_irt_bid REAL, usdt_irt_ask REAL,
                direction TEXT, gross_edge_bps REAL, net_edge_bps REAL
            );
        """)
        self.db.commit()

    def save_snapshot(self, snapshot: dict):
        self.db.execute(
            "INSERT INTO triangular_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                snapshot["base_asset"], snapshot["timestamp"],
                snapshot["x_usdt_bid"], snapshot["x_usdt_ask"],
                snapshot["x_irt_bid"], snapshot["x_irt_ask"],
                snapshot["usdt_irt_bid"], snapshot["usdt_irt_ask"],
                snapshot["direction"], snapshot["gross_edge_bps"], snapshot["net_edge_bps"],
            )
        )
        self.db.commit()

    def get_snapshots(self, base_asset=None, start=None, end=None) -> pd.DataFrame:
        query = "SELECT * FROM triangular_snapshots WHERE 1=1"
        params = []
        if base_asset:
            query += " AND base_asset=?"
            params.append(base_asset)
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        query += " ORDER BY timestamp"
        return pd.read_sql(query, self.db, params=params)
