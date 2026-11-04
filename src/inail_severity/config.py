"""Configuration loading and output folders."""
import os
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_config(path=None):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data_dir", "output_dir"):
        if not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(ROOT, cfg[key])
    return cfg


def out_dir(cfg, *parts):
    """Return (and create) a sub-folder of the output directory."""
    p = os.path.join(cfg["output_dir"], *parts)
    os.makedirs(p, exist_ok=True)
    return p
