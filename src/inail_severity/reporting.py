"""Paper tables (CSV) and the supplementary workbook (S1-S9)."""
import json, os, platform, sys
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

from .config import out_dir

T2_VARS = [("Male", "Male"), ("Age10", "Age (+10 years)"), ("ATECO_C", "C Manufacturing"), ("ATECO_F", "F Construction"),
           ("ATECO_H", "H Transport, storage"), ("ATECO_O", "O Public administration"), ("ATECO_Q", "Q Health, social work"),
           ("ATECO_R", "R Arts, recreation")]
VLAB = {"Male": "Male (vs female)", "Age10": "Age (+10 years, centred at 45)", "const": "Intercept"}
vlab = lambda v: VLAB.get(v, v.replace("ATECO_", "ATECO "))


def _p(v):
    return "<0.001" if v < 0.001 else round(float(v), 3)


def _ci(x, ci, d=3):
    return f"{x:.{d}f} ({ci[0]:.{d}f}-{ci[1]:.{d}f})"


def table2(cfg, regions):
    m = pd.read_csv(os.path.join(cfg["output_dir"], "inference", "mnlogit_all.csv")); rows = []
    for c in ["CA", "RD", "RS"]:
        for v, lab in T2_VARS:
            r = dict(Outcome=c, Variable=lab)
            for rg in regions:
                x = m[(m.region == rg) & (m.cls == c) & (m["var"] == v)].iloc[0]
                r[rg] = "n.e." if not x.estimable else f"{x.OR:.2f} ({x.lo:.2f}-{x.hi:.2f}){'*' if x.BH else ''}"
            rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(out_dir(cfg, "tables"), "Table2_multinomial_OR.csv"), index=False)
    return t


def perf_rows(o, region):
    """Panels A-C of Table 4 from a result dictionary."""
    rows = []
    for task, panel in [("4class", "A"), ("binary", "B")]:
        for r in [x for x in o["results"] if x["task"] == task]:
            base = r["model"].startswith("Majority")
            rows.append(dict(panel=panel, region=region, model=r["model"], accuracy=round(r["acc"], 3),
                             macro_F1=f"{r['mF1']:.3f}" if base else _ci(r["mF1"], r["ci_mF1"]),
                             macro_precision=round(r["mP"], 3),
                             balanced_accuracy=f"{r['mR']:.3f}" if base else _ci(r["mR"], r["ci_mR"]),
                             AUC="-" if base else _ci(r["auc"], r["ci_auc"])))
    for r in o.get("proper", []):
        rows.append(dict(panel="C", region=region, model=r["model"], log_loss=r.get("logloss"), pseudo_R2=r.get("pseudoR2"),
                         brier=r.get("brier"), BSS=r.get("bss"), macro_AUC=r.get("macro_auc"), AUC_TE_vs_other=r.get("bin_auc")))
    return rows


def table3(o, region):
    names = ["Logistic Regression", "XGBoost", "Balanced Random Forest"]; rows = []
    res = {r["model"]: r for r in o["results"] if r["task"] == "4class"}
    lr = [p for p in o["proper"] if p["model"].startswith("Multinomial LR")][0]
    sup = o["support"]; n = sum(sup)
    for c, cl in enumerate(["CA", "RD", "RS", "TE"]):
        r = dict(region=region, cls=cl, prevalence_pct=round(100 * sup[c] / n, 2))
        for nm in names:
            x = res[nm]; r[nm] = f"{x['P'][c]:.3f} / {x['R'][c]:.3f} / {x['F'][c]:.3f}"
        r["AUC / PR-AUC (unweighted LR)"] = f"{lr['pc_auc'][c]:.3f} / {lr['pr_auc'][c]:.3f}"; r["support"] = sup[c]
        rows.append(r)
    return rows


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def environment(cfg):
    import importlib
    env = dict(python=sys.version.split()[0], platform=platform.platform())
    for m in ["numpy", "pandas", "scipy", "sklearn", "imblearn", "xgboost", "catboost", "statsmodels", "shap", "matplotlib", "openpyxl", "joblib"]:
        try:
            env[m] = importlib.import_module(m).__version__
        except Exception:
            env[m] = "not installed"
    with open(os.path.join(cfg["output_dir"], "environment.json"), "w") as f:
        json.dump(env, f, indent=1)
    return env


def build(cfg, regions):
    o = cfg["output_dir"]; tdir = out_dir(cfg, "tables"); sdir = out_dir(cfg, "supplementary")
    sheets = {}; notes = ["Supplementary material for: Using Italian open data and machine learning to characterise regional patterns in workplace accident outcomes.",
                          "p-values below 0.001 are reported as <0.001; n.e. = not estimable (sector without any case of that outcome in the macro-region)."]
    # Tables 2-4
    if os.path.exists(os.path.join(o, "inference", "mnlogit_all.csv")):
        table2(cfg, regions)
    res = {r: _load(os.path.join(o, "predictive", f"res_{r}.json")) for r in regions}
    if all(res.values()):
        pd.DataFrame(sum([perf_rows(res[r], r) for r in regions], [])).to_csv(os.path.join(tdir, "Table4_performance.csv"), index=False)
        pd.DataFrame(sum([table3(res[r], r) for r in regions], [])).to_csv(os.path.join(tdir, "Table3_per_class.csv"), index=False)
    # S1-S3
    p = os.path.join(o, "inference", "mnlogit_all.csv")
    if os.path.exists(p):
        m = pd.read_csv(p)
        s1 = pd.DataFrame({"Macro-region": m.region, "Outcome vs TE": m.cls, "Variable": m["var"].map(vlab),
                           "Coefficient": m.coef.round(4), "Clustered SE": m.se.round(4), "OR": m.OR, "OR 95% low": m.lo,
                           "OR 95% high": m.hi, "p-value": m.p.map(_p), "Significant (BH, FDR 0.05)": np.where(m.BH, "Yes", "No")})
        ne = ~m.estimable.values
        for c in ["Coefficient", "Clustered SE", "OR", "OR 95% low", "OR 95% high", "p-value", "Significant (BH, FDR 0.05)"]:
            s1[c] = s1[c].astype(object); s1.loc[ne, c] = "n.e."
        sheets["S1 Multinomial logit"] = s1
        notes.append("S1: unpenalised multinomial logit, reference outcome TE (temporary disability), reference sector ATECO G; T and U merged into S; age centred at 45, per ten years; SE clustered by worker; BH correction across all coefficients of each regional model.")
        od_ = pd.read_csv(os.path.join(o, "inference", "ordinal_all.csv"))
        sheets["S2 Ordinal logit"] = pd.DataFrame({"Macro-region": od_.region, "Variable": od_["var"].map(vlab), "OR": od_.OR,
                                                   "OR 95% low": od_.lo, "OR 95% high": od_.hi, "p-value": od_.p.map(_p)})
        notes.append("S2: proportional-odds ordinal logit, TE < CA < RD < RS; same covariates; confidence intervals from model-based (not clustered) standard errors.")
        se = pd.read_csv(os.path.join(o, "inference", "sensitivity_period.csv"))
        sheets["S3 Sensitivity by period"] = pd.DataFrame({"Macro-region": se.region, "Sample": se["sample"], "N": se.n,
                                                           "Variable": se["var"].map(vlab), "OR": se.OR, "OR 95% low": se.lo, "OR 95% high": se.hi})
        notes.append("S3: binary logit (non-temporary vs TE) by period, with and without year fixed effects; SE clustered by worker.")
    pl = _load(os.path.join(o, "inference", "pooled_lrt.json"))
    if pl:
        rows = [[lbl, pl[k]["LR"], pl[k]["df"], _p(pl[k]["p"])] for k, lbl in [("region_main", "Region main effects"), ("all_interactions", "All region x feature interactions"),
                ("male", "Region x gender (vs model without interactions)"), ("age", "Region x age (vs model without interactions)"),
                ("sector_given_male_age", "Region x sector (vs model with gender and age interactions)")]]
        key = [k for k in pl if k.startswith("region_OR_vs_")][0]
        rows += [["", "", "", ""], [f"Macro-region OR of a non-temporary outcome (vs {key.split('_vs_')[1]})", "OR", "95% low", "95% high"]]
        rows += [[k] + [round(x, 3) for x in v] for k, v in pl[key].items()]
        sheets["S4 Regional LRT"] = pd.DataFrame(rows, columns=["Test", "LR chi2", "df", "p-value"])
        notes.append(f"S4: pooled binary logit for the four macro-regions (N = {pl['n']:,}); likelihood-ratio tests of region main effects and region x feature interactions; the tests treat records as independent.")
    p = os.path.join(sdir, "S5_selection_stages.csv")
    if os.path.exists(p):
        s5 = pd.read_csv(p)
        keep = ["region", "step", "label", "removed", "remaining"] + [c for c in s5.columns if c.startswith(("removed_", "retained_")) and not c.endswith("_n")]
        sheets["S5 Selection stages"] = s5[keep].round(2)
        notes.append("S5: characteristics of the records removed at each selection stage (applied sequentially) and of those retained after it: share of men, mean age, outcome distribution (including non-indemnified claims, NE) and year of the accident.")
    p = os.path.join(sdir, "S6_ggt_sensitivity.csv")
    if os.path.exists(p):
        s6 = pd.read_csv(p); pr = pd.read_csv(os.path.join(sdir, "S6_ggt_profiles.csv"))
        s6["var"] = s6["var"].map(vlab)
        sheets["S6 GGT sensitivity"] = pd.concat([pr.round(2), pd.DataFrame([{}]), s6.round(3)], ignore_index=True)
        notes.append("S6: records with an undetermined Large Tariff Group (GGT, not used by the models) are kept in the main sample; the earlier version of the analysis excluded them. Profile of these records, binary logit odds ratios (non-temporary vs TE, SE clustered by worker) and test-set AUC of the unweighted logistic regression in the two samples.")
    bs = [pd.read_csv(os.path.join(o, "inference", f"bootstrap_{r}.csv")) for r in regions if os.path.exists(os.path.join(o, "inference", f"bootstrap_{r}.csv"))]
    if bs:
        b = pd.concat(bs); b["var"] = b["var"].map(vlab)
        sheets["S7 Cluster bootstrap"] = b.round(4)
        notes.append(f"S7: worker-level (cluster) bootstrap of the multinomial logit ({cfg['bootstrap']['coef_B']} replicates): bootstrap SE and percentile CIs compared with the cluster-robust SE used in Table 2 and S1; se_boot_B50..B200 show the stability of the bootstrap SE as replicates are added.")
    for tag, sheet, note in [("tuning", "S8 Tuned models", f"S8: models tuned with the same procedure ({cfg['tuning']['n_iter']} random configurations per model, selected on the validation partition by macro AUC, binary AUC or log-loss), evaluated on the test set as in Table 4; selected hyperparameters in the last rows."),
                             ("chronological", "S9 Chronological", "S9: chronological validation: training on 2020-2022, threshold selection and early stopping on 2023, test on 2024.")]:
        fn = "tuned_{}.json" if tag == "tuning" else "chrono_{}.json"
        rr = {r: _load(os.path.join(o, tag, fn.format(r))) for r in regions}
        if all(rr.values()):
            df = pd.DataFrame(sum([perf_rows(rr[r], r) for r in regions], []))
            if tag == "tuning":
                df = pd.concat([df, pd.DataFrame([{}]), pd.DataFrame([dict(panel="selected hyperparameters", region=r, model=k, hyperparameters=json.dumps(v))
                                                                        for r in regions for k, v in rr[r]["best_params"].items()])], ignore_index=True)
            sheets[sheet] = df; notes.append(note)
    if sheets:
        path = os.path.join(sdir, "Supplementary_Tables.xlsx")
        with pd.ExcelWriter(path) as w:
            pd.DataFrame({"Notes": notes}).to_excel(w, sheet_name="Notes", index=False)
            for k, v in sheets.items():
                v.to_excel(w, sheet_name=k, index=False)
        wb = load_workbook(path)
        for ws in wb:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(90, max(10, max(len(str(c.value or "")) for c in col[:60]) + 2))
            for c in ws[1]:
                c.font = Font(bold=True)
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, float):
                        c.number_format = "0.000" if (abs(c.value) >= 0.001 or c.value == 0) else "0.00E+00"
        wb.save(path)
    environment(cfg)
    print("[report] tables and supplementary workbook written", flush=True)
