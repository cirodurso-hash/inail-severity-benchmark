"""Descriptive statistics of the analysed records (Appendices B and C, Figures 6-9)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import out_dir
from .data import load_clean

BLUE = "#2a78d6"; INK = "#262625"


def _bar(counts, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.bar([str(i) for i in counts.index], counts.values, color=BLUE)
    ax.set_title(title, fontsize=11, color=INK); ax.set_xlabel(xlabel); ax.set_ylabel("Number of accidents")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if len(counts) > 8:
        plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def run_descriptives(cfg, regions):
    td = out_dir(cfg, "tables"); fd = out_dir(cfg, "figures")
    age, gen, agegen, ggt, ateco, outc = [], [], [], [], [], []
    for r in regions:
        d = load_clean(cfg, r)
        a = d.Eta.describe(); age.append(dict(region=r, **a.round(2).to_dict()))
        for lab, s, lst in [("Gender", d.Genere, gen), ("Large Tariff Group (INAIL)", d.GrandeGruppoTariffario, ggt),
                            ("Economic Activity (ATECO)", d.CodiceAteco, ateco), ("Compensation", d.Indennizzo, outc)]:
            vc = s.value_counts()
            lst.append(dict(variable=lab, region=r, count=len(s), unique=s.nunique(), top=vc.index[0], freq=int(vc.iloc[0]),
                            pct=round(100 * vc.iloc[0] / len(s), 2)))
        for g, sub in d.groupby("Genere"):
            agegen.append(dict(region=r, gender=g, **sub.Eta.describe().round(2).to_dict()))
        short = r.replace("-", "")
        _bar(d.Genere.value_counts(), f"Gender, {r}", "Gender", os.path.join(fd, f"Figure6_gender_{short}.png"))
        _bar(d.CodiceAteco.value_counts(), f"ATECO section, {r}", "ATECO section", os.path.join(fd, f"Figure8_ateco_{short}.png"))
        _bar(d.GrandeGruppoTariffario.value_counts(), f"Large Tariff Group, {r}", "Large Tariff Group", os.path.join(fd, f"Figure7_ggt_{short}.png"))
        _bar(d.Indennizzo.value_counts().reindex(["TE", "CA", "RD", "RS"]), f"Compensation, {r}", "Compensation", os.path.join(fd, f"Figure9_outcome_{short}.png"))
    pd.DataFrame(age).to_csv(os.path.join(td, "AppendixB_age.csv"), index=False)
    pd.DataFrame(agegen).to_csv(os.path.join(td, "AppendixB_age_by_gender.csv"), index=False)
    pd.DataFrame(gen + ggt + ateco).to_csv(os.path.join(td, "AppendixB_categorical.csv"), index=False)
    pd.DataFrame(outc).to_csv(os.path.join(td, "AppendixC_outcome.csv"), index=False)
    print("[descriptives] appendix tables and figures written", flush=True)
