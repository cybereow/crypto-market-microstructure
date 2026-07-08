"""
Minimal REST client for LBank's public (no-auth) spot + perpetual-swap
market data. Confirmed live and reachable (unlike Tabdeal, LBank is not
geo-blocked from a non-Iran IP -- it instead bans Iran in its own Terms of
Service, enforced at the account/KYC level, not by blocking arbitrary
non-Iran traffic).

Endpoints confirmed by live testing (docs were incomplete/wrong in
places -- e.g. contract endpoints need a required `productGroup` param
not obvious from the docs alone):

    GET https://api.lbkex.com/v2/currencyPairs.do
    GET https://api.lbkex.com/v2/ticker/24hr.do?symbol=btc_usdt
    GET https://lbkperp.lbank.com/cfd/openApi/v1/pub/instrument?productGroup=SwapU
    GET https://lbkperp.lbank.com/cfd/openApi/v1/pub/marketData?productGroup=SwapU
    GET https://lbkperp.lbank.com/cfd/openApi/v1/pub/marketOrder?productGroup=SwapU&symbol=BTCUSDT

marketData response per symbol includes a real `fundingRate` field (a
genuine perpetual funding-rate mechanism, unlike Tabdeal) plus
`positionFeeTime` (funding interval in seconds) and `nextFeeTime`.
"""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SPOT_BASE_URL = "https://api.lbkex.com"
PERP_BASE_URL = "https://lbkperp.lbank.com/cfd/openApi/v1/pub"
TIMEOUT_SECONDS = 15


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    return session


class LBankClient:
    def __init__(self, product_group: str = "SwapU"):
        self.product_group = product_group
        self.session = _make_session()

    def get_spot_pairs(self) -> list[str]:
        resp = self.session.get(f"{SPOT_BASE_URL}/v2/currencyPairs.do", timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()["data"]

    def get_spot_ticker(self, symbol: str) -> dict:
        resp = self.session.get(
            f"{SPOT_BASE_URL}/v2/ticker/24hr.do", params={"symbol": symbol}, timeout=TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["ticker"]

    def get_perp_instruments(self) -> list[dict]:
        resp = self.session.get(
            f"{PERP_BASE_URL}/instrument", params={"productGroup": self.product_group}, timeout=TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        return resp.json()["data"]

    def get_perp_market_data(self) -> list[dict]:
        """All perpetual symbols in one call, including live fundingRate,
        markedPrice, underlyingPrice, positionFeeTime (funding interval in
        seconds), nextFeeTime (ms epoch of next funding settlement)."""
        resp = self.session.get(
            f"{PERP_BASE_URL}/marketData", params={"productGroup": self.product_group}, timeout=TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        return resp.json()["data"]
