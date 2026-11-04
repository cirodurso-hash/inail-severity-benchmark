"""SHAP stability of the binary CatBoost model over repeated 60/20/20 hold-out splits."""
import json, os, time
from itertools import combinations
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from .config import out_dir
from .data import load_clean
from .features import TE, design_ml
from .models import DEFAULTS


def run_stability(cfg, region):
    d = load_clean(cfg, region); X, y4 = design_ml(d); y = (y4 == TE).astype(int)
    N = cfg["stability"]["n_runs"]; runs = []
    for i in range(N):
        t0 = time.time()
        Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=cfg["split"]["test_size"], stratify=y, random_state=i)
        Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=cfg["split"]["val_size"], stratify=ytmp, random_state=i)
        m = CatBoostClassifier(**DEFAULTS["CatBoost (binary)"], auto_class_weights="Balanced", loss_function="Logloss",
                               eval_metric="AUC", early_stopping_rounds=200, verbose=0, random_seed=i, thread_count=cfg["n_jobs"],
                               allow_writing_files=False)
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
        f, t, th = roc_curve(yva, m.predict_proba(Xva)[:, 1]); thr = th[np.argmax(t - f)]
        ste = m.predict_proba(Xte)[:, 1]
        sv = m.get_feature_importance(type="ShapValues", data=Pool(Xte))[:, :-1]
        runs.append(dict(f1=f1_score(yte, (ste >= thr).astype(int), average="macro"), auc=roc_auc_score(yte, ste),
                         imp=pd.Series(np.abs(sv).mean(0), index=X.columns).to_dict()))
        print(f"[stability] {region} run {i} {time.time() - t0:.0f}s", flush=True)
    I = pd.DataFrame([r["imp"] for r in runs])
    pairs = list(combinations(range(N), 2))
    rho = [spearmanr(I.iloc[a], I.iloc[b])[0] for a, b in pairs]
    top = [I.iloc[k].sort_values(ascending=False) for k in range(N)]
    ov = [len(set(top[a].index[:10]) & set(top[b].index[:10])) / 10 for a, b in pairs]
    lead = I.mean().sort_values(ascending=False).head(6).index
    out = dict(region=region, n_runs=N, f1_mean=float(np.mean([r["f1"] for r in runs])), f1_sd=float(np.std([r["f1"] for r in runs])),
               auc_mean=float(np.mean([r["auc"] for r in runs])), auc_sd=float(np.std([r["auc"] for r in runs])),
               spearman_mean=float(np.mean(rho)), spearman_sd=float(np.std(rho)), spearman_min=float(np.min(rho)),
               top10_mean=float(np.mean(ov)), top10_sd=float(np.std(ov)),
               first_feature=pd.Series([t.index[0] for t in top]).value_counts().to_dict(),
               mean_shap=I.mean()[lead].round(4).to_dict(), sd_shap=I.std(ddof=0)[lead].round(4).to_dict())
    with open(os.path.join(out_dir(cfg, "stability"), f"stability_{region}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
