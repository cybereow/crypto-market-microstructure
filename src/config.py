import os
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(_ROOT, "config.yaml")) as f:
    _cfg = yaml.safe_load(f)

OUTPUT_DIR = os.path.join(_ROOT, _cfg.get("data", {}).get("output_dir", "data/"))
