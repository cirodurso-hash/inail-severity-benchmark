"""Sensitivity of the results to the treatment of records with an undetermined Large Tariff
Group (GGT), a variable not used by the models (Supplementary Table S6): the main sample keeps
them; the earlier version of the analysis excluded them."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import out_dir
from .data import load_region
from .features import TE, design_ml
from .inference import binlogit, SENS_VARS
from .predictive import fit_model, random_split

VARS = SENS_VARS + ["ATECO_R"]


def run_ggt(cfg, regions):
    rows = []; prof = []
    for r in regions:
        wide, _ = load_region(cfg, r, skip=set())                       # earlier sample: GGT = ND excluded
        main, _ = load_region(cfg, r, skip={"ggt_missing"})            # main sample: GGT = ND kept
        extra = main[main.GrandeGruppoTariffario == "ND"]
        base = wide
        for lab, d in [("Main sample (GGT = ND kept)", main), ("Excluding GGT = ND (earlier version)", base), ("GGT = ND records only", extra)]:
            prof.append(dict(region=r, sample=lab, n=len(d), male_pct=100 * (d.Genere == "M").mean(),
                             age_mean=d.Eta.mean(), nonTE_pct=100 * (d.Indennizzo != "TE").mean(),
                             top_sectors=", ".join(f"{k} {v:.0f}%" for k, v in (d.CodiceAteco.value_counts(normalize=True).head(3) * 100).items())))
        for lab, d in [("Main sample (GGT = ND kept)", main), ("Excluding GGT = ND (earlier version)", base)]:
            m = binlogit(d)
            X, y = design_ml(d); Xtr, Xva, Xte, ytr, yva, yte = random_split(X, y, cfg)
            lr = fit_model("Logistic Regression", {"C": 1.0}, Xtr, ytr, Xva, yva, cfg, weighted=False)
            auc = roc_auc_score((yte == TE).astype(int), lr.predict_proba(Xte)[:, TE])
            for v in VARS:
                b = m.params[v]; se = m.bse[v]
                rows.append(dict(region=r, sample=lab, n=len(d), var=v, OR=np.exp(b), lo=np.exp(b - 1.96 * se),
                                 hi=np.exp(b + 1.96 * se), auc_TE_vs_other_unweighted_LR=auc))
        print(f"[ggt] {r}: +{len(extra):,} records", flush=True)
    od = out_dir(cfg, "supplementary")
    pd.DataFrame(rows).to_csv(os.path.join(od, "S6_ggt_sensitivity.csv"), index=False)
    pd.DataFrame(prof).to_csv(os.path.join(od, "S6_ggt_profiles.csv"), index=False)
