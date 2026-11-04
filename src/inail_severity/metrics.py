"""Evaluation metrics, decision rules and bootstrap confidence intervals."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve


def cm_metrics(y, p, k):
    """Confusion-matrix metrics; macro averages over the k classes (one-vs-rest)."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(tp + fp > 0, tp / (tp + fp), 0)
        R = np.where(tp + fn > 0, tp / (tp + fn), 0)
        F = np.where(P + R > 0, 2 * P * R / (P + R), 0)
    return dict(acc=tp.sum() / cm.sum(), mP=P.mean(), mR=R.mean(), mF1=F.mean(), P=P, R=R, F=F, cm=cm)


def fast_auc(y01, s):
    n1 = y01.sum(); n0 = len(y01) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y01 == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(y, p, k, rng, proba=None, B=1000, Bauc=300):
    """95% percentile bootstrap CIs (resampling of the test set) for macro-F1, balanced
    accuracy (= macro recall), accuracy and AUC (first Bauc resamples)."""
    n = len(y); out = {"mF1": [], "mR": [], "acc": [], "auc": []}
    for b in range(B):
        i = rng.integers(0, n, n); m = cm_metrics(y[i], p[i], k)
        out["mF1"].append(m["mF1"]); out["mR"].append(m["mR"]); out["acc"].append(m["acc"])
        if proba is not None and b < Bauc:
            if k == 2:
                out["auc"].append(fast_auc(y[i], proba[i]))
            else:
                out["auc"].append(np.nanmean([fast_auc((y[i] == c).astype(int), proba[i][:, c]) for c in range(k)]))
    return {kk: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for kk, v in out.items() if len(v)}


def youden(yv, sv):
    """Threshold maximising Youden's J = TPR - FPR on the validation set."""
    f, t, th = roc_curve(yv, sv)
    return th[np.argmax(t - f)]


def apply_thr(P, thr):
    """Class-specific thresholds: among classes whose probability reaches its threshold the most
    probable is assigned; if none reaches its threshold, the most probable class is assigned."""
    ok = P >= np.array(thr); adj = np.where(ok, P, -1); pred = adj.argmax(1)
    none = ~ok.any(1); pred[none] = P[none].argmax(1)
    return pred


def brier(y, P):
    """Multi-class Brier score."""
    Y = np.eye(P.shape[1])[y]
    return np.mean(((P - Y) ** 2).sum(1))


def summarize(name, task, y, pred, proba, k, rng, B=1000, Bauc=300):
    m = cm_metrics(y, pred, k); ci = boot(y, pred, k, rng, proba, B, Bauc)
    if proba is None:
        auc = np.nan; pcauc = [np.nan] * k
    elif k == 2:
        auc = roc_auc_score(y, proba); pcauc = [auc]
    else:
        pcauc = [roc_auc_score((y == c).astype(int), proba[:, c]) for c in range(k)]; auc = np.mean(pcauc)
    return dict(model=name, task=task, acc=m["acc"], mP=m["mP"], mR=m["mR"], mF1=m["mF1"], auc=auc,
                ci_mF1=ci["mF1"], ci_mR=ci["mR"], ci_acc=ci["acc"], ci_auc=ci.get("auc", (np.nan, np.nan)),
                P=m["P"].tolist(), R=m["R"].tolist(), F=m["F"].tolist(), pc_auc=pcauc, cm=m["cm"].tolist())
