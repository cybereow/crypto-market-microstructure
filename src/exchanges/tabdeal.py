"""
Minimal REST client for Tabdeal's public (no-auth) spot market data.

Endpoint shapes confirmed from the official docs.tabdeal.org page provided
by the user (not ccxt -- ccxt has no "tabdeal" exchange class):

    GET https://api1.tabdeal.org/r/api/v1/depth?symbol=USDTIRT&limit=5
    GET https://api1.tabdeal.org/r/api/v1/exchangeInfo

depth response: {"bids": [[price_str, qty_str], ...], "asks": [[price_str, qty_str], ...]}
exchangeInfo response: {"symbols": [{"symbol": "BTCIRT", "baseAsset": "BTC",
                                      "quoteAsset": "IRT", "status": "TRADING", ...}, ...]}

Uses a shared Session with connection pooling + retries: under
config.MAX_WORKERS concurrent requests, plain one-shot requests.get() calls
saw frequent DNS resolution failures, read timeouts, and occasional 502s
against the live API -- a pooled session with automatic retry on
transient errors recovers most of these instead of just dropping the
asset from that cycle.
"""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .. import config

TIMEOUT_SECONDS = 15


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class TabdealClient:
    def __init__(self, base_url=None):
        self.base_url = base_url or config.TABDEAL_BASE_URL
        self.session = _make_session()

    def get_exchange_info(self) -> list[dict]:
        resp = self.session.get(f"{self.base_url}/exchangeInfo", timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        # Docs show {"symbols": [...]}, but the live endpoint returns a bare
        # list -- accept either so this doesn't break again if it changes.
        return data["symbols"] if isinstance(data, dict) else data

    def get_depth(self, symbol: str, limit: int = 5) -> dict:
        resp = self.session.get(
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
