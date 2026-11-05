"""Generate small synthetic files with the same structure as the INAIL open data, so that the
whole pipeline can be tested without the real data:  python tests/make_synthetic.py <folder>"""
import os, sys
import numpy as np
import pandas as pd

FILES = {"North-West": "NordOvestAT.csv", "North-East": "NordEstAT.csv", "Center": "CentroAT.csv", "South": "SudAT.csv"}
SECT = list("ABCDEFGHIJKLMNOPQRSTU")


def make(n, seed):
    rng = np.random.default_rng(seed)
    sect = rng.choice(SECT, n, p=np.r_[[0.01, 0.005, 0.2, 0.01, 0.02, 0.1, 0.15, 0.08, 0.06, 0.02, 0.02, 0.01, 0.03, 0.04, 0.04, 0.02, 0.15, 0.01, 0.015, 0.005, 0.005]] / 1.0)
    male = rng.random(n) < 0.63; age = rng.integers(15, 75, n)
    eta = -3.0 + 0.045 * (age - 45) + 0.3 * male + np.where(sect == "F", 0.6, 0) - np.where(sect == "Q", 1.2, 0)
    p_non = 1 / (1 + np.exp(-eta)); u = rng.random(n)
    out = np.where(u < p_non, rng.choice(["CA", "RD", "RS"], n, p=[0.78, 0.2, 0.02]), "TE")
    out = np.where((sect == "P") & (out == "RS"), "RD", out)                       # a sector without fatal cases
    ind = np.where(rng.random(n) < 0.3, "NE", out)
    days = rng.integers(0, 5 * 365, n)
    date = (pd.Timestamp("2020-01-01") + pd.to_timedelta(days, "D")).strftime("%d/%m/%Y")
    worker = rng.integers(0, int(n * 0.85), n)
    df = pd.DataFrame(dict(DataRilevazione="30/04/2025", DataAccadimento=date, Genere=np.where(male, "M", "F"), Eta=age.astype(str),
                           CodiceAteco=sect, SettoreAttivitaEconomica=[s + " 10" for s in sect],
                           GrandeGruppoTariffario=rng.integers(0, 10, n).astype(str), LuogoAccadimento="58",
                           Indennizzo=ind, ModalitaAccadimento=np.where(rng.random(n) < 0.1, "S", "N"),
                           PosizioneAssicurativaTerritoriale=rng.integers(1, 10**6, n).astype(str),
                           IdentificativoInfortunato=worker.astype(str)))
    nd = rng.random(n) < 0.25; df.loc[nd, ["SettoreAttivitaEconomica", "CodiceAteco"]] = "ND"
    gest = np.full(n, "I", dtype=object); other = nd & (rng.random(n) < 0.6)    # Agriculture / State: no sector
    gest[other] = rng.choice(["A", "S"], int(other.sum())); df["Gestione"] = gest
    df.loc[rng.random(n) < 0.04, "GrandeGruppoTariffario"] = "ND"
    df.loc[rng.random(n) < 0.0005, "Eta"] = "-1"
    return df


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "tests/synthetic"
    os.makedirs(folder, exist_ok=True)
    for i, (r, f) in enumerate(FILES.items()):
        make(30000, i).to_csv(os.path.join(folder, f), sep=";", index=False)
    print("synthetic data written to", folder)
