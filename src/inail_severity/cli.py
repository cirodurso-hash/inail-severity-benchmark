"""Command-line entry point.

    python run.py all                 # complete pipeline
    python run.py prepare             # data cleaning, Table 1
    python run.py descriptives        # Appendices B-C, Figures 6-9
    python run.py predictive [REGION] # Tables 3-4 (and Figure 4)
    python run.py inference           # Table 2, Table 5, S1-S3
    python run.py pooled              # S4
    python run.py selection           # S5
    python run.py ggt                 # S6
    python run.py bootstrap [REGION]  # S7 (slow)
    python run.py tuning [REGION]     # S8 (slow)
    python run.py chrono [REGION]     # S9
    python run.py stability [REGION]  # SHAP stability (slow)
    python run.py figures             # Figures 4, 5, 10, 11
    python run.py report              # paper tables and supplementary workbook
"""
import argparse

from . import REGIONS
from .config import load_config

STEPS = ["prepare", "descriptives", "predictive", "inference", "pooled", "selection", "ggt", "bootstrap", "tuning", "chrono", "stability", "figures", "report"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("region", nargs="?", choices=REGIONS, help="run a single macro-region (where applicable)")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--skip-slow", action="store_true", help="with 'all': skip bootstrap, tuning and stability")
    a = ap.parse_args(argv)
    cfg = load_config(a.config); regs = [a.region] if a.region else REGIONS
    steps = STEPS if a.step == "all" else [a.step]
    if a.step == "all" and a.skip_slow:
        steps = [s for s in steps if s not in ("bootstrap", "tuning", "stability")]
    for s in steps:
        if s == "prepare":
            from .data import prepare; prepare(cfg, REGIONS)
        elif s == "descriptives":
            from .descriptives import run_descriptives; run_descriptives(cfg, REGIONS)
        elif s == "predictive":
            from .predictive import run_main
            for r in regs: run_main(cfg, r)
        elif s == "inference":
            from .inference import run_inference; run_inference(cfg, REGIONS)
        elif s == "pooled":
            from .pooled import run_pooled; run_pooled(cfg, REGIONS)
        elif s == "selection":
            from .data import selection_stages; selection_stages(cfg, REGIONS)
        elif s == "ggt":
            from .sensitivity import run_ggt; run_ggt(cfg, REGIONS)
        elif s == "bootstrap":
            from .inference import cluster_bootstrap
            for r in regs: cluster_bootstrap(cfg, r)
        elif s == "tuning":
            from .tuning import run_tuning
            for r in regs: run_tuning(cfg, r)
        elif s == "chrono":
            from .predictive import run_chronological
            for r in regs: run_chronological(cfg, r)
        elif s == "stability":
            from .stability import run_stability
            for r in regs: run_stability(cfg, r)
        elif s == "figures":
            from .figures import fig_confusion, fig_forest, fig_roc_shap
            fig_confusion(cfg); fig_forest(cfg, REGIONS); fig_roc_shap(cfg, REGIONS)
        elif s == "report":
            from .reporting import build; build(cfg, REGIONS)


if __name__ == "__main__":
    main()
