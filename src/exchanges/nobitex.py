"""
Minimal REST client for Nobitex's public (no-auth) spot market data.

Endpoint shape confirmed from the official nobitex/docs-api GitHub source
(source/includes/_market_data.md), not fabricated or guessed:

    GET https://apiv2.nobitex.ir/v3/orderbook/USDTIRT

Response: {"status": "ok", "lastUpdate": ..., "lastTradePrice": "...",
           "asks": [[price_str, qty_str], ...], "bids": [[price_str, qty_str], ...]}
"""
from __future__ import annotations

import requests

from .. import config

TIMEOUT_SECONDS = 10


class NobitexClient:
    def __init__(self, base_url=None):
        self.base_url = base_url or config.NOBITEX_BASE_URL

    def get_orderbook(self, symbol: str) -> dict:
        resp = requests.get(f"{self.base_url}/v3/orderbook/{symbol}", timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"Nobitex orderbook error for {symbol}: {data}")
        return data

    def best_bid_ask(self, symbol: str) -> tuple[float, float]:
        ob = self.get_orderbook(symbol)
        best_bid = float(ob["bids"][0][0]) if ob.get("bids") else None
        best_ask = float(ob["asks"][0][0]) if ob.get("asks") else None
        return best_bid, best_ask
