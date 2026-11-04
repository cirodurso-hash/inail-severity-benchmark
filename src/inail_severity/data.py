"""Loading of the INAIL open data and record selection (Table 1, Supplementary Table S5).

The filters are applied sequentially, in the order below; each step removes records from
those remaining after the previous steps.
"""
import os
import numpy as np
import pandas as pd

from .config import out_dir

OUTCOMES = ["TE", "CA", "RD", "RS"]

# (key, label used in Table 1, function returning the rows to KEEP)
FILTERS = [
    ("sector_missing", "Economic sector = ND (missing)", lambda d: d.SettoreAttivitaEconomica != "ND"),
    ("ateco_missing", "Economic sector = ND (missing)", lambda d: d.CodiceAteco != "ND"),
    ("ggt_missing", "Large Tariff Group = ND", lambda d: d.GrandeGruppoTariffario != "ND"),
    ("age_missing", "Age = -1 / under 15", lambda d: d.Eta != "-1"),
    ("place_missing", "Place of occurrence = ND/NaN", lambda d: (d.LuogoAccadimento != "") & (d.LuogoAccadimento != "ND")),
    ("not_indemnified", "Non-indemnified claim (NE)", lambda d: d.Indennizzo != "NE"),
    ("commuting", "Commuting accident (in itinere)", lambda d: d.ModalitaAccadimento != "S"),
    ("pat_missing", "Employer insurance position (PAT) = -1", lambda d: d.PosizioneAssicurativaTerritoriale != "-1"),
    ("age_under15", "Age = -1 / under 15", lambda d: d.Eta.astype(int) >= 15),
]

REQUIRED = ["DataAccadimento", "Genere", "Eta", "CodiceAteco", "SettoreAttivitaEconomica",
            "GrandeGruppoTariffario", "LuogoAccadimento", "Indennizzo", "ModalitaAccadimento",
            "PosizioneAssicurativaTerritoriale", "IdentificativoInfortunato"]


def read_raw(cfg, region):
    path = os.path.join(cfg["data_dir"], cfg["files"][region])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: download the INAIL file and place it in {cfg['data_dir']}")
    df = pd.read_csv(path, sep=cfg["csv"]["sep"], encoding=cfg["csv"]["encoding"], dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def _profile(d):
    """Observed characteristics of a set of records (used to compare selection stages)."""
    n = len(d)
    if n == 0:
        return dict(n=0)
    age = pd.to_numeric(d.Eta, errors="coerce")
    age = age[(age >= 0)]
    year = pd.to_datetime(d.DataAccadimento, format="%d/%m/%Y", errors="coerce").dt.year
    out = dict(n=n, male_pct=100 * (d.Genere == "M").mean(), age_mean=age.mean() if len(age) else np.nan,
               age_sd=age.std() if len(age) > 1 else np.nan)
    ind = d.Indennizzo.value_counts(normalize=True) * 100
    for o in OUTCOMES + ["NE"]:
        out[f"outcome_{o}_pct"] = ind.get(o, 0.0)
    nonte = d.Indennizzo.isin(["CA", "RD", "RS"]).sum()
    indemn = d.Indennizzo.isin(OUTCOMES).sum()
    out["nonTE_among_indemnified_pct"] = 100 * nonte / indemn if indemn else np.nan
    yr = year.value_counts(normalize=True) * 100
    for y in range(2020, 2025):
        out[f"year_{y}_pct"] = yr.get(y, 0.0)
    return out


def default_skip(cfg):
    """Filters not applied in the main analysis (records with undetermined GGT are kept)."""
    return set() if cfg.get("selection", {}).get("exclude_missing_ggt", False) else {"ggt_missing"}


def load_region(cfg, region, skip=None, profile=False):
    """Apply the selection filters; return the analysed records and the selection log.

    skip: filter keys not applied (default: from config, see default_skip).
    profile: if True, the log also contains the characteristics of removed and retained records.
    """
    skip = default_skip(cfg) if skip is None else set(skip)
    df = read_raw(cfg, region)
    log = [dict(step="raw", label="Original records", removed=0, remaining=len(df),
                **({f"removed_{k}": v for k, v in _profile(df.iloc[:0]).items()} if profile else {}))]
    if profile:
        log[0].update({f"retained_{k}": v for k, v in _profile(df).items()})
    for key, label, keep in FILTERS:
        if key in skip:
            continue
        m = keep(df)
        entry = dict(step=key, label=label, removed=int((~m).sum()), remaining=int(m.sum()))
        if profile:
            entry.update({f"removed_{k}": v for k, v in _profile(df[~m]).items()})
            entry.update({f"retained_{k}": v for k, v in _profile(df[m]).items()})
        log.append(entry)
        df = df[m]
    df = df.copy()
    df["Eta"] = df.Eta.astype(int)
    df["year"] = pd.to_datetime(df.DataAccadimento, format="%d/%m/%Y").dt.year
    return df.reset_index(drop=True), log


def prepare(cfg, regions):
    """Build the cleaned datasets (one pickle per region) and Table 1."""
    od = out_dir(cfg, "data")
    rows = []
    for r in regions:
        d, log = load_region(cfg, r)
        d.to_pickle(os.path.join(od, f"{r}.pkl"))
        for e in log:
            rows.append(dict(region=r, **e))
        print(f"[prepare] {r}: {len(d):,} records retained", flush=True)
    log = pd.DataFrame(rows)
    log.to_csv(os.path.join(od, "selection_log.csv"), index=False)
    # Table 1: records removed by reason (sector and ATECO missing merged; the two age filters merged)
    t1 = log[log.step != "raw"].groupby(["label", "region"], sort=False).removed.sum().unstack("region")[regions]
    raw = log[log.step == "raw"].set_index("region").remaining[regions]
    fin = log.groupby("region").remaining.last()[regions]
    t1.loc["Original records"] = raw.values
    t1.loc["Retained (final)"] = fin.values
    t1.to_csv(os.path.join(out_dir(cfg, "tables"), "Table1_records_removed.csv"))
    return t1


def load_clean(cfg, region):
    p = os.path.join(cfg["output_dir"], "data", f"{region}.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found: run the 'prepare' step first")
    return pd.read_pickle(p)


def selection_stages(cfg, regions):
    """Supplementary Table S5: characteristics of records removed at each selection stage."""
    rows = []
    for r in regions:
        _, log = load_region(cfg, r, profile=True)
        for e in log:
            rows.append(dict(region=r, **e))
        print(f"[selection] {r} done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir(cfg, "supplementary"), "S5_selection_stages.csv"), index=False)
    return t
