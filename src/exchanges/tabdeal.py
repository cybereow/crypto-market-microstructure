"""
Minimal REST client for Tabdeal's public (no-auth) spot market data.

Endpoint shape confirmed from the official docs.tabdeal.org page provided
by the user (not ccxt -- ccxt has no "tabdeal" exchange class):

    GET https://api1.tabdeal.org/r/api/v1/depth?symbol=USDTIRT&limit=5

Response: {"bids": [[price_str, qty_str], ...], "asks": [[price_str, qty_str], ...]}
same [price, quantity] string-pair format as Binance.
"""
from __future__ import annotations

import requests

from .. import config

TIMEOUT_SECONDS = 10


class TabdealClient:
    def __init__(self, base_url=None):
        self.base_url = base_url or config.TABDEAL_BASE_URL

    def get_depth(self, symbol: str, limit: int = 5) -> dict:
        resp = requests.get(
            f"{self.base_url}/depth",
            params={"symbol": symbol, "limit": limit},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()

    def best_bid_ask(self, symbol: str) -> tuple[float, float]:
        depth = self.get_depth(symbol, limit=1)
        best_bid = float(depth["bids"][0][0]) if depth.get("bids") else None
        best_ask = float(depth["asks"][0][0]) if depth.get("asks") else None
        return best_bid, best_ask
