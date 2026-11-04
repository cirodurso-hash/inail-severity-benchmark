"""Predictive comparison (Tables 3 and 4, Figure 4) on a stratified 60/20/20 split.

The same function is reused for the tuned models (S8) and the chronological validation (S9).
"""
import itertools, json, os, time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight

from .config import out_dir
from .data import load_clean
from .features import TE, design_ml
from .metrics import apply_thr, brier, cm_metrics, summarize, youden
from .models import DEFAULTS, make


def random_split(X, y, cfg):
    s = cfg["seeds"]["split"]
    Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=cfg["split"]["test_size"], stratify=y, random_state=s)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=cfg["split"]["val_size"], stratify=ytmp, random_state=s)
    return Xtr, Xva, Xte, ytr, yva, yte


def chrono_split(d, X, y, cfg):
    c = cfg["chronological"]; yr = d.year.values
    m = [np.isin(yr, c[k]) for k in ("train_years", "val_years", "test_years")]
    return X[m[0]], X[m[1]], X[m[2]], y[m[0]], y[m[1]], y[m[2]]


def saturated(Xtr, ytr, Xte, k, alpha=1.0):
    """Cell-frequency model for every (gender, age, ATECO) cell, smoothed towards the class
    frequencies; unseen cells receive the class frequencies."""
    ac = [c for c in Xtr.columns if c.startswith("CodiceAteco_")]
    key = lambda X: (X["Genere_M"].astype(str) + "_" + X["Eta"].astype(str) + "_"
                     + X[ac].values.argmax(1).astype(str) + X[ac].values.sum(1).astype(str))
    ktr = key(Xtr); kte = key(Xte)
    ct = pd.crosstab(ktr, ytr).reindex(columns=range(k), fill_value=0)
    pri = np.bincount(ytr, minlength=k) / len(ytr)
    pr = (ct.values + alpha * k * pri) / (ct.values.sum(1, keepdims=True) + alpha * k)
    out = pd.DataFrame(pr, index=ct.index).reindex(kte).to_numpy(copy=True)
    out[np.isnan(out[:, 0])] = pri
    return out


def fit_model(name, params, Xtr, ytr, Xva, yva, cfg, weighted=True, early_stop=False):
    seed = cfg["seeds"]["split"]; nj = cfg["n_jobs"]
    if name == "CatBoost (binary)":
        m = make(name, params, seed, nj, weighted)
        m.fit(Xtr, (ytr == TE).astype(int), eval_set=(Xva, (yva == TE).astype(int)))
        return m
    m = make(name, params, seed, nj, weighted)
    if name == "XGBoost":
        sw = compute_sample_weight("balanced", ytr) if weighted else None
        if early_stop:
            swv = compute_sample_weight("balanced", yva) if weighted else None
            m.set_params(early_stopping_rounds=50)
            m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)],
                  sample_weight_eval_set=[swv] if weighted else None, verbose=False)
        else:
            m.fit(Xtr, ytr, sample_weight=sw)
        return m
    m.fit(Xtr, ytr)
    return m


def evaluate(Xtr, Xva, Xte, ytr, yva, yte, cfg, params=None, early_stop=False, proper=True):
    """Fit all models and evaluate them on the test set. Returns (result dict, fitted models)."""
    params = params or DEFAULTS
    B, Bauc = cfg["bootstrap"]["metric_B"], cfg["bootstrap"]["auc_B"]
    rng = np.random.default_rng(cfg["seeds"]["metric_bootstrap"])   # same sequence as in the paper
    res = []; yb_te = (yte == TE).astype(int); yb_va = (yva == TE).astype(int); k = 4
    res.append(summarize("Majority baseline", "4class", yte, np.full(len(yte), TE), None, 4, rng, B, Bauc))
    res.append(summarize("Majority baseline", "binary", yb_te, np.ones(len(yte), int), None, 2, rng, B, Bauc))
    fitted = {}; thr = {}
    for name in ["Logistic Regression", "XGBoost", "Balanced Random Forest"]:
        m = fit_model(name, params[name], Xtr, ytr, Xva, yva, cfg, early_stop=early_stop); fitted[name] = m
        Pva = m.predict_proba(Xva); Pte = m.predict_proba(Xte)
        if name == "Balanced Random Forest":
            grid = np.round(np.arange(0.3, 0.71, 0.1), 2); best = (-1, None)
            for t in itertools.product(grid, repeat=4):
                ba = cm_metrics(yva, apply_thr(Pva, t), 4)["mR"]
                if ba > best[0]:
                    best = (ba, t)
            thr[name] = [float(x) for x in best[1]]; pte = apply_thr(Pte, best[1])
        else:
            pte = Pte.argmax(1)
        res.append(summarize(name, "4class", yte, pte, Pte, 4, rng, B, Bauc))
        tb = youden(yb_va, Pva[:, TE]); thr[name + "_bin"] = float(tb)
        res.append(summarize(name, "binary", yb_te, (Pte[:, TE] >= tb).astype(int), Pte[:, TE], 2, rng, B, Bauc))
    name = "CatBoost (binary)"
    cb = fit_model(name, params[name], Xtr, ytr, Xva, yva, cfg); fitted[name] = cb
    sva = cb.predict_proba(Xva)[:, 1]; ste = cb.predict_proba(Xte)[:, 1]; tb = youden(yb_va, sva); thr["CatBoost_bin"] = float(tb)
    res.append(summarize(name, "binary", yb_te, (ste >= tb).astype(int), ste, 2, rng, B, Bauc))
    out = dict(n_train=len(ytr), n_val=len(yva), n_test=len(yte), support=np.bincount(yte, minlength=4).tolist(),
               results=res, thresholds=thr, cb_iters=int(cb.get_best_iteration()))
    if proper:
        out["proper"] = proper_scores(Xtr, Xva, Xte, ytr, yva, yte, cfg, params, early_stop)
    return out, fitted


def proper_scores(Xtr, Xva, Xte, ytr, yva, yte, cfg, params, early_stop=False):
    """Panel C: unweighted probability models evaluated with threshold-independent criteria."""
    yb_te = (yte == TE).astype(int)
    pri = np.bincount(ytr, minlength=4) / len(ytr)
    Ppri = np.tile(pri, (len(yte), 1))
    lru = fit_model("Logistic Regression", params.get("LR_unweighted", {"C": 1.0}), Xtr, ytr, Xva, yva, cfg, weighted=False)
    Psat = saturated(pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]), Xte, 4)
    xgu = fit_model("XGBoost", params.get("XGB_unweighted", DEFAULTS["XGBoost"]), Xtr, ytr, Xva, yva, cfg,
                    weighted=False, early_stop=early_stop)
    cbu = fit_model("CatBoost (binary)", params.get("CB_unweighted", DEFAULTS["CatBoost (binary)"]), Xtr, ytr, Xva, yva, cfg,
                    weighted=False)
    scbu = cbu.predict_proba(Xte)[:, 1]
    ll0 = log_loss(yte, Ppri, labels=range(4)); b0 = brier(yte, Ppri); ps = []
    for name, P in [("Class-frequency (prior) model", Ppri), ("Multinomial LR, unweighted", lru.predict_proba(Xte)),
                    ("XGBoost, unweighted", xgu.predict_proba(Xte)), ("Saturated cell model (age x gender x ATECO)", Psat)]:
        ll = log_loss(yte, P, labels=range(4)); bs = brier(yte, P)
        prior = name.startswith("Class-frequency")
        pcauc = [0.5] * 4 if prior else [roc_auc_score((yte == c).astype(int), P[:, c]) for c in range(4)]
        prauc = [average_precision_score((yte == c).astype(int), P[:, c]) for c in range(4)]
        ps.append(dict(model=name, logloss=ll, pseudoR2=1 - ll / ll0, brier=bs, bss=1 - bs / b0,
                       macro_auc=float(np.mean(pcauc)), pc_auc=pcauc, pr_auc=prauc,
                       prev=[float((yte == c).mean()) for c in range(4)],
                       bin_auc=0.5 if prior else roc_auc_score(yb_te, P[:, TE])))
    ps.append(dict(model="CatBoost binary, unweighted", bin_auc=roc_auc_score(yb_te, scbu)))
    return ps


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def run_main(cfg, region):
    """Tables 3 and 4 for one macro-region; saves the CatBoost model for the SHAP figures."""
    t0 = time.time()
    d = load_clean(cfg, region); X, y = design_ml(d)
    out, fitted = evaluate(*random_split(X, y, cfg), cfg)
    out.update(region=region, n=len(y), time=time.time() - t0)
    od = out_dir(cfg, "predictive")
    _dump(out, os.path.join(od, f"res_{region}.json"))
    fitted["CatBoost (binary)"].save_model(os.path.join(od, f"catboost_{region}.cbm"))
    print(f"[predictive] {region} done in {time.time() - t0:.0f}s", flush=True)
    return out


def run_chronological(cfg, region):
    """Supplementary Table S9: train on 2020-2022, select thresholds on 2023, test on 2024."""
    t0 = time.time()
    d = load_clean(cfg, region); X, y = design_ml(d)
    out, _ = evaluate(*chrono_split(d, X, y, cfg), cfg)
    out.update(region=region, n=len(y), years=cfg["chronological"], time=time.time() - t0)
    _dump(out, os.path.join(out_dir(cfg, "chronological"), f"chrono_{region}.json"))
    print(f"[chronological] {region} done in {time.time() - t0:.0f}s", flush=True)
    return out
