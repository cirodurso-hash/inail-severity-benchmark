"""Inferential models: multinomial logit (Table 2, S1), ordinal logit (Figure 5, S2),
binary logit by period (S3), predicted probabilities of illustrative profiles (Table 5) and
cluster bootstrap of the multinomial coefficients (S7)."""
import gc, os, time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.miscmodels.ordinal_model import OrderedModel

from .config import out_dir
from .data import load_clean
from .features import design_inf, severity, worker_groups

OUT_CLS = ["CA", "RD", "RS"]
COLS = ["Genere", "Eta", "CodiceAteco", "Indennizzo", "IdentificativoInfortunato", "year"]
SENS_VARS = ["Male", "Age10", "ATECO_C", "ATECO_F", "ATECO_H", "ATECO_O", "ATECO_Q"]


def bh(p, q=0.05):
    """Benjamini-Hochberg: boolean array of rejected hypotheses."""
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    sig = np.zeros(n, bool)
    if ok.any():
        sig[o[: np.max(np.where(ok)[0]) + 1]] = True
    return sig


def nonestimable(d):
    """(outcome, sector) cells without any case: their coefficients are not identified."""
    a = d.CodiceAteco.replace({"T": "S", "U": "S"})
    ct = pd.crosstab(a, d.Indennizzo)
    return {(c, "ATECO_" + s) for c in OUT_CLS for s in ct.index if s != "G" and ct.loc[s].get(c, 0) == 0}


def fit_mnl(X, y, groups=None, start=None, maxiter=100):
    kw = dict(cov_type="cluster", cov_kwds={"groups": groups}) if groups is not None else {}
    return sm.MNLogit(y, X).fit(method="newton", maxiter=maxiter, disp=0, start_params=start, **kw)


def mnl_table(d, region):
    X = design_inf(d); y = severity(d)
    m = fit_mnl(X, y, worker_groups(d))
    ne = nonestimable(d); rows = []
    for j, c in enumerate(OUT_CLS):
        b = m.params.iloc[:, j]; se = m.bse.iloc[:, j]; p = m.pvalues.iloc[:, j]
        for v in X.columns:
            rows.append(dict(region=region, cls=c, var=v, coef=b[v], se=se[v], OR=np.exp(b[v]),
                             lo=np.exp(b[v] - 1.96 * se[v]), hi=np.exp(b[v] + 1.96 * se[v]), p=p[v],
                             estimable=(c, v) not in ne))
    t = pd.DataFrame(rows)
    t["BH"] = bh(t.p.values)          # all coefficients of the regional model, including intercepts
    return t, m


def ordinal_table(d, region):
    X = design_inf(d).drop(columns="const")
    y = pd.Series(pd.Categorical(d.Indennizzo, categories=["TE", "CA", "RD", "RS"], ordered=True))
    m = OrderedModel(y, X, distr="logit").fit(method="bfgs", maxiter=500, disp=0)
    rows = []
    for v in m.params.index:
        if "/" in v:
            continue
        b = m.params[v]; se = m.bse[v]
        rows.append(dict(region=region, var=v, OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=m.pvalues[v]))
    return pd.DataFrame(rows)


def binlogit(d, extra=None):
    X = design_inf(d)
    if extra is not None:
        X = pd.concat([X, extra.reset_index(drop=True)], axis=1)
    y = (d.Indennizzo != "TE").astype(float).values
    return sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": worker_groups(d)})


def period_table(d, region):
    rows = []
    yrs = pd.get_dummies(d.year, prefix="y", drop_first=True).astype(float)
    for lab, mask, ex in [("All years", None, None), ("All years + year FE", None, yrs),
                          ("2020-2022", d.year <= 2022, None), ("2023-2024", d.year >= 2023, None),
                          ("Excluding 2024", d.year <= 2023, None)]:
        sub = d if mask is None else d[mask].reset_index(drop=True)
        m = binlogit(sub, ex)
        for v in SENS_VARS:
            b = m.params[v]; se = m.bse[v]
            rows.append(dict(region=region, sample=lab, n=len(sub), var=v, OR=np.exp(b),
                             lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se)))
    return pd.DataFrame(rows)


PROFILES = [("Female, 30", "Q", 0, 30), ("Female, 55", "Q", 0, 55), ("Female, 45", "M", 0, 45),
            ("Male, 30", "F", 1, 30), ("Male, 55", "F", 1, 55), ("Male, 55", "C", 1, 55),
            ("Male, 45", "H", 1, 45), ("Male, 45", "O", 1, 45)]


def profiles_table(mnl, regions):
    """Table 5: predicted probability (%) of a non-temporary and of an annuity outcome."""
    rows = []
    for lab, s, male, age in PROFILES:
        r = dict(profile=lab, sector=s)
        for rg in regions:
            t = mnl[mnl.region == rg]; eta = []
            for c in OUT_CLS:
                b = t[t.cls == c].set_index("var").coef
                eta.append(b["const"] + b["Male"] * male + b["Age10"] * (age - 45) / 10 + (b.get("ATECO_" + s, 0) if s != "G" else 0))
            ex = np.exp(eta); p = ex / (1 + ex.sum())
            r[rg] = f"{p.sum() * 100:.1f} / {(p[1] + p[2]) * 100:.1f}"
        rows.append(r)
    return pd.DataFrame(rows)


def _boot_once(X, y, g, b, seed, start):
    rng = np.random.default_rng(seed)
    ug = np.unique(g); draw = rng.choice(ug, size=len(ug), replace=True)
    # rows of the drawn workers (with multiplicity)
    order = np.argsort(g, kind="stable"); gs = g[order]
    starts = np.searchsorted(gs, draw, "left"); ends = np.searchsorted(gs, draw, "right")
    idx = np.concatenate([order[s:e] for s, e in zip(starts, ends)])
    try:
        # started from the full-sample estimates; 30 Newton steps are ample for the estimable
        # coefficients (those of outcome-sector cells without cases diverge and are not used)
        m = fit_mnl(X.iloc[idx].reset_index(drop=True), y[idx], start=start, maxiter=30)
        return m.params.values.ravel(order="F")
    except Exception:
        return np.full(start.shape, np.nan)


def cluster_bootstrap(cfg, region, n_jobs=None):
    """S7: worker-level (cluster) bootstrap of the multinomial logit, compared with the
    cluster-robust standard errors; also reports how the bootstrap SE stabilises with B."""
    t0 = time.time()
    d = load_clean(cfg, region)[COLS]; X = design_inf(d); y = severity(d); g = worker_groups(d)
    full = fit_mnl(X, y, g)
    start = full.params.values.ravel(order="F")
    B = cfg["bootstrap"]["coef_B"]; seeds = np.random.SeedSequence(cfg["seeds"]["coef_bootstrap"]).generate_state(B)
    draws = Parallel(n_jobs=n_jobs or cfg["n_jobs"])(delayed(_boot_once)(X, y, g, b, int(seeds[b]), start) for b in range(B))
    D = np.vstack(draws)
    ne = nonestimable(d); rows = []
    k = X.shape[1]
    for j, c in enumerate(OUT_CLS):
        for i, v in enumerate(X.columns):
            col = D[:, j * k + i]; col = col[np.isfinite(col)]
            se_r = full.bse.iloc[i, j]
            r = dict(region=region, cls=c, var=v, coef=full.params.iloc[i, j], se_robust=se_r,
                     estimable=(c, v) not in ne, B_valid=len(col))
            if r["estimable"] and len(col) > 10:
                r.update(se_boot=col.std(ddof=1), ratio=col.std(ddof=1) / se_r,
                         OR_lo_boot=np.exp(np.percentile(col, 2.5)), OR_hi_boot=np.exp(np.percentile(col, 97.5)),
                         OR_lo_robust=np.exp(r["coef"] - 1.96 * se_r), OR_hi_robust=np.exp(r["coef"] + 1.96 * se_r))
                for bb in (50, 100, 150, 200):
                    if bb <= len(col):
                        r[f"se_boot_B{bb}"] = col[:bb].std(ddof=1)
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir(cfg, "inference"), f"bootstrap_{region}.csv"), index=False)
    print(f"[bootstrap] {region}: {B} replicates in {time.time() - t0:.0f}s", flush=True)
    return t


def run_inference(cfg, regions):
    od = out_dir(cfg, "inference"); mn = []; od_ = []; se = []
    for r in regions:
        t0 = time.time(); d = load_clean(cfg, r)[COLS]
        t, _ = mnl_table(d, r); mn.append(t); gc.collect()
        od_.append(ordinal_table(d, r)); gc.collect()
        se.append(period_table(d, r)); del d; gc.collect()
        print(f"[inference] {r} done in {time.time() - t0:.0f}s", flush=True)
    mn = pd.concat(mn); od_ = pd.concat(od_); se = pd.concat(se)
    mn.to_csv(os.path.join(od, "mnlogit_all.csv"), index=False)
    od_.to_csv(os.path.join(od, "ordinal_all.csv"), index=False)
    se.to_csv(os.path.join(od, "sensitivity_period.csv"), index=False)
    profiles_table(mn, regions).to_csv(os.path.join(out_dir(cfg, "tables"), "Table5_profiles.csv"), index=False)
    return mn, od_, se
