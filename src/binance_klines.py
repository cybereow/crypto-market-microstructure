"""
Download and cache real daily spot klines from Binance Vision, used only to
ground the leverage/liquidation-safety numbers in actual historical price
moves instead of a couple of anecdotal headline events.
"""
from __future__ import annotations

import io
import os
from zipfile import ZipFile

import pandas as pd
import requests

from . import config

_COLS = ["open_time", "open", "high", "low", "close", "volume",
         "close_time", "qav", "trades", "tbb", "tbq", "ignore"]


def _cache_path(symbol: str, year: int, month: int) -> str:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, f"{symbol}-1d-{year}-{month:02d}.zip")


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
    url = f"{config.KLINES_ARCHIVE_URL}/{symbol}/1d/{symbol}-1d-{year}-{month:02d}.zip"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    with open(path, "wb") as fh:
        fh.write(r.content)
    return r.content


def load_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    frames = []
    for year, month in _month_range(start, end):
        raw = _download_month(symbol, year, month)
        if raw is None:
            continue
        with ZipFile(io.BytesIO(raw)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                frames.append(pd.read_csv(f, header=None, names=_COLS, dtype={"open_time": str}))
    if not frames:
        raise RuntimeError(f"No daily klines found for {symbol} in {start}..{end}")

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["open_time"].str.slice(0, 10).astype("int64"), unit="s", utc=True)
    df["close"] = pd.to_numeric(df["close"])
    df["high"] = pd.to_numeric(df["high"])
    return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def worst_move_stats(df: pd.DataFrame, window_days: int) -> dict:
    """Worst / p99 / p95 upside move (entry close -> peak high) over the
    `window_days` FOLLOWING entry, excluding entry day itself (no look-ahead:
    you enter at today's close, risk starts tomorrow)."""
    fwd_high = df["high"].shift(-1)[::-1].rolling(window_days, min_periods=window_days).max()[::-1]
    move_pct = ((fwd_high / df["close"] - 1) * 100).dropna()
    if not len(move_pct):
        return {}
    return {
        "window_days": window_days,
        "worst_pct": round(float(move_pct.max()), 1),
        "p99_pct": round(float(move_pct.quantile(0.99)), 1),
        "p95_pct": round(float(move_pct.quantile(0.95)), 1),
        "median_pct": round(float(move_pct.median()), 1),
    }
