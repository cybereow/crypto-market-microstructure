"""
Authenticated Tabdeal spot trading client (TRADE security: X-MBX-APIKEY
header + HMAC-SHA256 signature over the query string). Confirmed from
docs.tabdeal.org.

NEVER hardcode or commit real API keys. This reads them from environment
variables:
    TABDEAL_API_KEY
    TABDEAL_API_SECRET

Base URL quirk confirmed from the docs: GET endpoints use a `/r/` prefix
(e.g. GET /r/api/v1/order), but POST/DELETE (order placement/cancel) do
NOT (e.g. POST /api/v1/order) -- this client uses a different base URL
per HTTP method to match.

This client can place REAL orders with REAL money. There is no dry-run
guard inside this class itself -- that safety belongs in the caller
(see triangular_executor.py), which defaults to dry-run and requires an
explicit flag to place live orders.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from urllib.parse import urlencode

from .tabdeal import TIMEOUT_SECONDS, _make_session

READ_BASE_URL = "https://api1.tabdeal.org/r/api/v1"
WRITE_BASE_URL = "https://api1.tabdeal.org/api/v1"


class TabdealTradingClient:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.environ.get("TABDEAL_API_KEY")
        self.api_secret = api_secret or os.environ.get("TABDEAL_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise RuntimeError(
                "TABDEAL_API_KEY / TABDEAL_API_SECRET not set. Export them as "
                "environment variables first -- never hardcode or commit real API keys."
            )
        self.session = _make_session()

    def _sign(self, params: dict) -> dict:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, path: str, params: dict) -> dict:
        base_url = READ_BASE_URL if method == "GET" else WRITE_BASE_URL
        signed = self._sign(params)
        headers = {"X-MBX-APIKEY": self.api_key}
        resp = self.session.request(
            method, f"{base_url}{path}", params=signed, headers=headers, timeout=TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        return resp.json()

    def get_account(self) -> dict:
        """Real balances + real maker/taker commission -- use this instead
        of the config.yaml fee placeholders once you have real keys."""
        return self._request("GET", "/account", {})

    def place_market_order(self, symbol: str, side: str, quantity: str) -> dict:
        """side: 'BUY' or 'SELL'. quantity: base-asset amount as a string,
        already rounded to the symbol's LOT_SIZE stepSize (see
        exchangeInfo filters) -- an unrounded quantity will likely be
        rejected."""
        return self._request("POST", "/order", {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        })

    def get_order(self, symbol: str, order_id: int) -> dict:
        return self._request("GET", "/order", {"symbol": symbol, "orderId": order_id})
