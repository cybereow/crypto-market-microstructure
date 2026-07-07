import os

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(_ROOT, "config.yaml")) as f:
    _cfg = yaml.safe_load(f)

PAIRS = _cfg["pairs"]

TABDEAL_BASE_URL = _cfg["exchanges"]["tabdeal"]["base_url"]
TABDEAL_TAKER_FEE_BPS = float(_cfg["exchanges"]["tabdeal"]["taker_fee_bps"])

NOBITEX_BASE_URL = _cfg["exchanges"]["nobitex"]["base_url"]
NOBITEX_TAKER_FEE_BPS = float(_cfg["exchanges"]["nobitex"]["taker_fee_bps"])

POLL_INTERVAL_SECONDS = int(_cfg["scan"]["poll_interval_seconds"])
MIN_NET_EDGE_BPS = float(_cfg["scan"]["min_net_edge_bps"])
DEPTH_LIMIT = int(_cfg["scan"]["depth_limit"])
DB_PATH = os.path.join(_ROOT, _cfg["scan"]["db_path"])
OUTPUT_DIR = os.path.join(_ROOT, _cfg["scan"]["output_dir"])
