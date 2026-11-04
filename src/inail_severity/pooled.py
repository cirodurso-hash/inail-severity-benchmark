"""Pooled binary logit for the four macro-regions and likelihood-ratio tests of regional
differences (Supplementary Table S4)."""
import gc, json, os
import numpy as np
import pandas as pd
from scipy import stats

from .config import out_dir
from .data import load_clean
from .features import design_inf


def _newton(A, y, maxit=50):
    """Maximum-likelihood logistic regression by Newton-Raphson (memory-light)."""
    A = np.ascontiguousarray(A, dtype=np.float64); b = np.zeros(A.shape[1])
    for _ in range(maxit):
        eta = A @ b; p = 1 / (1 + np.exp(-eta))
        H = (A * (p * (1 - p))[:, None]).T @ A; step = np.linalg.solve(H, A.T @ (y - p)); b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = A @ b; ll = float(np.sum(y * eta - np.logaddexp(0, eta)))
    return b, ll, H


def run_pooled(cfg, regions):
    parts = [load_clean(cfg, r)[["Genere", "Eta", "CodiceAteco", "Indennizzo"]].assign(region=r) for r in regions]
    D = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    y = (D.Indennizzo != "TE").astype(float).values
    X = design_inf(D).astype(np.float64)
    R = pd.get_dummies(D.region).astype(float)[regions]; Rn = R.drop(columns=regions[0])
    inter = lambda v: pd.DataFrame({f"{r}:{v}": Rn[r] * X[v] for r in Rn.columns})
    X0 = pd.concat([X, Rn], axis=1)
    _, ll_noreg, _ = _newton(X.values, y)
    b0, ll0, H0 = _newton(X0.values, y)
    _, llM, _ = _newton(pd.concat([X0, inter("Male")], axis=1).values, y)
    _, llA, _ = _newton(pd.concat([X0, inter("Age10")], axis=1).values, y)
    _, llMA, _ = _newton(pd.concat([X0, inter("Male"), inter("Age10")], axis=1).values, y)
    llF = 0.0                                             # all interactions = separate regional models
    for r in regions:
        m = (D.region == r).values
        llF += _newton(X.values[m], y[m])[1]
    k = X.shape[1]

    def t(l1, l0, df):
        L = 2 * (l1 - l0); return dict(LR=round(L, 1), df=df, p=float(stats.chi2.sf(L, df)))
    out = dict(n=len(y), region_main=t(ll0, ll_noreg, len(regions) - 1),
               all_interactions=t(llF, ll0, (len(regions) - 1) * (k - 1)),
               male=t(llM, ll0, len(regions) - 1), age=t(llA, ll0, len(regions) - 1),
               sector_given_male_age=t(llF, llMA, (len(regions) - 1) * (k - 3)))
    se = np.sqrt(np.diag(np.linalg.inv(H0)))
    out["region_OR_vs_" + regions[0]] = {c: (float(np.exp(b0[i])), float(np.exp(b0[i] - 1.96 * se[i])), float(np.exp(b0[i] + 1.96 * se[i])))
                                          for i, c in enumerate(X0.columns) if c in Rn.columns}
    with open(os.path.join(out_dir(cfg, "inference"), "pooled_lrt.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[pooled] done", flush=True)
    return out
