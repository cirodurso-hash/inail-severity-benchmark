"""Figures: confusion matrix (Figure 4), ordinal forest plot (Figure 5), ROC curves of the
binary CatBoost model (Figure 10) and SHAP summary plots (Figure 11)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score, roc_curve

from .config import out_dir
from .data import load_clean
from .features import CLASSES, TE, design_ml
from .predictive import random_split

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
INK = "#262625"; MUTE = "#6b6a64"; GRID = "#e6e5df"
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; MK = ["o", "s", "^", "D"]


def fig_confusion(cfg, region="North-West"):
    o = json.load(open(os.path.join(cfg["output_dir"], "predictive", f"res_{region}.json")))
    r = [x for x in o["results"] if x["model"] == "Logistic Regression" and x["task"] == "4class"][0]
    cm = np.array(r["cm"]); rown = cm / cm.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(rown, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:,}\n({rown[i, j] * 100:.1f}%)", ha="center", va="center", fontsize=9,
                    color="white" if rown[i, j] > 0.45 else INK)
    ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Observed class")
    ax.set_title(f"Logistic regression, {region} (test set, n = {cm.sum():,})", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.set_label("Share of observed class (row %)", color=MUTE)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir(cfg, "figures"), "Figure4_confusion_matrix.png"), dpi=300); plt.close(fig)


LAB = {"Male": "Male (vs female)", "Age10": "Age (+10 years)", "ATECO_A": "A Agriculture", "ATECO_B": "B Mining",
       "ATECO_C": "C Manufacturing", "ATECO_D": "D Electricity, gas", "ATECO_E": "E Water, waste", "ATECO_F": "F Construction",
       "ATECO_H": "H Transport, storage", "ATECO_I": "I Accommodation, food", "ATECO_J": "J Information, comm.",
       "ATECO_K": "K Finance, insurance", "ATECO_L": "L Real estate", "ATECO_M": "M Professional services",
       "ATECO_N": "N Support services", "ATECO_O": "O Public administration", "ATECO_P": "P Education",
       "ATECO_Q": "Q Health, social work", "ATECO_R": "R Arts, recreation", "ATECO_S": "S Other services (incl. T, U)"}


def fig_forest(cfg, regions):
    d = pd.read_csv(os.path.join(cfg["output_dir"], "inference", "ordinal_all.csv")); order = list(LAB)
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    for k, (rg, c, m) in enumerate(zip(regions, COLS, MK)):
        s = d[d.region == rg].set_index("var").loc[order]
        yy = np.arange(len(order))[::-1] + (1.5 - k) * 0.18
        ax.errorbar(s.OR, yy, xerr=[s.OR - s.lo, s.hi - s.OR], fmt=m, color=c, ms=5, elinewidth=1.4, capsize=0, label=rg,
                    markeredgecolor="white", markeredgewidth=0.6)
    ax.axvline(1, color=MUTE, lw=1, ls="--"); ax.set_xscale("log")
    ax.set_xticks([0.2, 0.3, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2", "3"])
    ax.set_yticks(np.arange(len(order))[::-1], [LAB[v] for v in order])
    ax.set_xlabel("Odds ratio for a more severe outcome (log scale, 95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, 1.06), ncol=4, frameon=False, fontsize=9)
    ax.set_title("Ordinal logit (TE < CA < RD < RS); sector reference: G Trade", fontsize=10, color=INK, pad=26)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir(cfg, "figures"), "Figure5_forest_ordinal.png"), dpi=300); plt.close(fig)


def _display_names(cols):
    return [("Male" if c == "Genere_M" else "Age" if c == "Eta" else "ATECO " + c.split("_")[1]) for c in cols]


def fig_roc_shap(cfg, regions):
    import shap
    od = out_dir(cfg, "figures")
    for rg in regions:
        d = load_clean(cfg, rg); X, y = design_ml(d)
        _, _, Xte, _, _, yte = random_split(X, y, cfg)
        m = CatBoostClassifier(); m.load_model(os.path.join(cfg["output_dir"], "predictive", f"catboost_{rg}.cbm"))
        s = m.predict_proba(Xte)[:, 1]; yb = (yte == TE).astype(int); f, t, _ = roc_curve(yb, s)
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.plot(f, t, color=COLS[0], lw=1.6, label=f"AUC = {roc_auc_score(yb, s):.3f}"); ax.plot([0, 1], [0, 1], color=MUTE, ls="--", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(f"CatBoost (TE vs other), {rg}", fontsize=10)
        ax.legend(loc="lower right", frameon=False); fig.tight_layout()
        fig.savefig(os.path.join(od, f"Figure10_ROC_{rg}.png"), dpi=200); plt.close(fig)
        sv = m.get_feature_importance(type="ShapValues", data=Pool(Xte))[:, :-1]
        Xd = Xte.copy(); Xd.columns = _display_names(Xte.columns)
        plt.figure(); shap.summary_plot(sv, Xd, show=False, max_display=20)
        plt.title(f"SHAP, CatBoost (TE vs rest), {rg}"); plt.savefig(os.path.join(od, f"Figure11_SHAP_{rg}.png"), dpi=200, bbox_inches="tight"); plt.close("all")
        print(f"[figures] ROC and SHAP {rg}", flush=True)
