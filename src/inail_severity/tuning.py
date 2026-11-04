"""Comparable hyperparameter tuning (Supplementary Table S8).

Every model receives the same budget: n_iter configurations drawn at random from its search
space (models.SPACES), each fitted on the training partition and scored on the validation
partition with a threshold-independent criterion. The best configuration is then evaluated on
the untouched test set exactly as in Table 4.

Criteria: macro one-vs-rest AUC for the class-weighted four-class models, AUC for the
class-weighted binary CatBoost, log-loss for the unweighted probability models (Panel C).
"""
import json, os, time
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .config import out_dir
from .data import load_clean
from .features import TE, design_ml
from .models import DEFAULTS, SPACES, sample_configs
from .predictive import _dump, evaluate, fit_model, random_split

# (key in the params dict, model family, weighted, criterion)
TARGETS = [
    ("Logistic Regression", "Logistic Regression", True, "macro_auc"),
    ("XGBoost", "XGBoost", True, "macro_auc"),
    ("Balanced Random Forest", "Balanced Random Forest", True, "macro_auc"),
    ("CatBoost (binary)", "CatBoost (binary)", True, "bin_auc"),
    ("LR_unweighted", "Logistic Regression", False, "logloss"),
    ("XGB_unweighted", "XGBoost", False, "logloss"),
    ("CB_unweighted", "CatBoost (binary)", False, "bin_logloss"),
]


def _score(m, crit, Xva, yva):
    if crit == "macro_auc":
        P = m.predict_proba(Xva)
        return float(np.mean([roc_auc_score((yva == c).astype(int), P[:, c]) for c in range(4)])), True
    if crit == "logloss":
        return float(log_loss(yva, m.predict_proba(Xva), labels=range(4))), False
    s = m.predict_proba(Xva)[:, 1]; yb = (yva == TE).astype(int)
    if crit == "bin_auc":
        return float(roc_auc_score(yb, s)), True
    return float(log_loss(yb, s)), False


def run_tuning(cfg, region):
    t0 = time.time()
    d = load_clean(cfg, region); X, y = design_ml(d)
    Xtr, Xva, Xte, ytr, yva, yte = random_split(X, y, cfg)
    n_iter = cfg["tuning"]["n_iter"]; best = {}; log = []
    for i, (key, fam, weighted, crit) in enumerate(TARGETS):
        rng = np.random.default_rng(cfg["seeds"]["tuning"] + i)
        cands = sample_configs(fam, n_iter, rng)
        scored = []
        for p in cands:
            m = fit_model(fam, p, Xtr, ytr, Xva, yva, cfg, weighted=weighted, early_stop=True)
            s, higher = _score(m, crit, Xva, yva)
            if fam == "XGBoost" and getattr(m, "best_iteration", None) is not None:
                p = dict(p, n_estimators=int(m.best_iteration) + 1)
            if fam == "CatBoost (binary)":
                p = dict(p, iterations=int(m.get_best_iteration()) + 1)
            scored.append((s, p)); log.append(dict(model=key, criterion=crit, score=s, **{f"param_{k}": v for k, v in p.items()}))
        sel = max(scored, key=lambda t: t[0]) if higher else min(scored, key=lambda t: t[0])
        best[key] = sel[1]
        print(f"[tuning] {region} {key}: best {crit}={sel[0]:.4f} {sel[1]}", flush=True)
    # XGBoost: the tuned number of trees comes from early stopping, so no further early stopping on refit
    out, _ = evaluate(Xtr, Xva, Xte, ytr, yva, yte, cfg, params=best, early_stop=False)
    out.update(region=region, best_params=best, search_log=log, n_iter=n_iter, time=time.time() - t0)
    _dump(out, os.path.join(out_dir(cfg, "tuning"), f"tuned_{region}.json"))
    print(f"[tuning] {region} done in {time.time() - t0:.0f}s", flush=True)
    return out
