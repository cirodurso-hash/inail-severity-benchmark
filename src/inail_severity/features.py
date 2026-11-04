"""Design matrices.

Predictive models (Tables 3-4): gender, age in years and one-hot ATECO sections B-U
(section A is the omitted category).
Inferential models (Table 2, S1-S4): gender, age centred at 45 in ten-year units and ATECO
sections with G (wholesale and retail trade) as reference; T and U merged into S.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

CLASSES = ["CA", "RD", "RS", "TE"]      # label encoding of the predictive models
TE = 3
ATECO_ML = list("BCDEFGHIJKLMNOPQRSTU")
FEATS = ["Genere_M", "Eta"] + ["CodiceAteco_" + a for a in ATECO_ML]
SECT_INF = list("ABCDEFHIJKLMNOPQRS")   # G is the reference


def design_ml(d):
    X = pd.DataFrame({"Genere_M": (d.Genere == "M").astype(int).values, "Eta": d.Eta.astype(int).values})
    for a in ATECO_ML:
        X["CodiceAteco_" + a] = (d.CodiceAteco == a).astype(int).values
    y = d.Indennizzo.map({c: i for i, c in enumerate(CLASSES)}).values
    return X[FEATS], y


def design_inf(d):
    a = d.CodiceAteco.replace({"T": "S", "U": "S"})
    X = pd.DataFrame({"Male": (d.Genere == "M").astype(float).values,
                      "Age10": (d.Eta.astype(float).values - 45) / 10})
    for s in SECT_INF:
        X["ATECO_" + s] = (a == s).astype(float).values
    return sm.add_constant(X)


def severity(d):
    """Ordered severity code: TE=0 < CA=1 < RD=2 < RS=3."""
    return d.Indennizzo.map({"TE": 0, "CA": 1, "RD": 2, "RS": 3}).values


def worker_groups(d):
    return pd.factorize(d.IdentificativoInfortunato)[0]
