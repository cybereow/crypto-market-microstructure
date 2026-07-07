import os

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(_ROOT, "config.yaml")) as f:
    _cfg = yaml.safe_load(f)

EXCHANGE = _cfg["exchange"]
SYMBOLS = _cfg["symbols"]
CAPITAL = float(_cfg["capital"])
LEVERAGE = float(_cfg["leverage"])

SPOT_MAKER_BPS = float(_cfg["fees"]["spot_maker_bps"])
SPOT_TAKER_BPS = float(_cfg["fees"]["spot_taker_bps"])
FUT_MAKER_BPS = float(_cfg["fees"]["fut_maker_bps"])
FUT_TAKER_BPS = float(_cfg["fees"]["fut_taker_bps"])

FUNDING_START = _cfg["funding"]["start"]
FUNDING_END = _cfg["funding"]["end"]
ROLLING_WINDOW_DAYS = int(_cfg["funding"]["rolling_window_days"])
ENTRY_THRESHOLD = float(_cfg["funding"]["entry_threshold"])
EXIT_THRESHOLD = float(_cfg["funding"]["exit_threshold"])

CANDLE_TIMEFRAME = _cfg["data"]["candle_timeframe"]
ORDERBOOK_DEPTH = int(_cfg["data"]["orderbook_depth"])
CACHE_DIR = os.path.join(_ROOT, _cfg["data"]["cache_dir"])
OUTPUT_DIR = os.path.join(_ROOT, _cfg["data"]["output_dir"])
DB_PATH = os.path.join(_ROOT, _cfg["data"]["db_path"])

# Public, unauthenticated static archive -- NOT the live fapi.binance.com
# REST API. The live funding-rate/futures API is geo-blocked in some
# environments; this CDN is not, and needs no API key. Always fetch
# historical funding/kline data through these, never through ccxt's live
# fetch_funding_rate().
FUNDING_ARCHIVE_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
KLINES_ARCHIVE_URL = "https://data.binance.vision/data/spot/monthly/klines"
