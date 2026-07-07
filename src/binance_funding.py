"""
Download and cache real historical Binance USDT-margined perpetual funding
rates from the public Binance Vision static archive (data.binance.vision).

This is the same reliable, unauthenticated CDN used for klines below --
unlike fapi.binance.com (the live REST API), it is not geo-restricted and
needs no API key. Monthly CSVs go back to ~Jan 2020 for BTCUSDT/ETHUSDT.
"""
from __future__ import annotations

import io
import os
from zipfile import ZipFile

import pandas as pd
import requests

from . import config


def _cache_path(symbol: str, year: int, month: int) -> str:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, f"{symbol}-fundingRate-{year}-{month:02d}.zip")


def _month_range(start: str, end: str):
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _download_month(symbol: str, year: int, month: int) -> bytes | None:
    path = _cache_path(symbol, year, month)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read()
    url = f"{config.FUNDING_ARCHIVE_URL}/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    with open(path, "wb") as fh:
        fh.write(r.content)
    return r.content


def load_symbol(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Return real historical funding payments for `symbol` in [start, end]
    (inclusive, 'YYYY-MM' strings), columns: ts, funding_rate (fraction,
    e.g. 0.0001 = 0.01%), funding_interval_hours."""
    frames = []
    for year, month in _month_range(start, end):
        raw = _download_month(symbol, year, month)
        if raw is None:
            continue
        with ZipFile(io.BytesIO(raw)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                frames.append(pd.read_csv(f))
    if not frames:
        raise RuntimeError(f"No funding-rate data found for {symbol} in {start}..{end}")

    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["calc_time"], unit="ms", utc=True)
    out = out.rename(columns={"last_funding_rate": "funding_rate"})
    return out[["ts", "funding_rate", "funding_interval_hours"]].sort_values("ts").reset_index(drop=True)
