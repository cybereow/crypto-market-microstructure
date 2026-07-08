import os

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(_ROOT, "config.yaml")) as f:
    _cfg = yaml.safe_load(f)

TABDEAL_BASE_URL = _cfg["exchange"]["tabdeal"]["base_url"]
TABDEAL_TAKER_FEE_BPS = float(_cfg["exchange"]["tabdeal"]["taker_fee_bps"])

ANCHOR_SYMBOL = _cfg["triangular"]["anchor_symbol"]
MAX_SYMBOLS = int(_cfg["triangular"]["max_symbols"])
MAX_WORKERS = int(_cfg["triangular"]["max_workers"])
POLL_INTERVAL_SECONDS = int(_cfg["triangular"]["poll_interval_seconds"])
MIN_NET_EDGE_BPS = float(_cfg["triangular"]["min_net_edge_bps"])
DEPTH_LIMIT = int(_cfg["triangular"]["depth_limit"])
DB_PATH = os.path.join(_ROOT, _cfg["triangular"]["db_path"])
OUTPUT_DIR = os.path.join(_ROOT, _cfg["triangular"]["output_dir"])

LBANK_PRODUCT_GROUP = _cfg["lbank"]["product_group"]
LBANK_SPOT_TAKER_FEE_BPS = float(_cfg["lbank"]["spot_taker_fee_bps"])
LBANK_PERP_TAKER_FEE_BPS = float(_cfg["lbank"]["perp_taker_fee_bps"])
LBANK_POLL_INTERVAL_SECONDS = int(_cfg["lbank"]["poll_interval_seconds"])
LBANK_DB_PATH = os.path.join(_ROOT, _cfg["lbank"]["db_path"])
