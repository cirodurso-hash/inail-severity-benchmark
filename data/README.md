# Data

The analysis uses the INAIL open dataset on workplace accidents ("Infortuni sul lavoro"),
release of 30 April 2025 (field `DataRilevazione`), covering accidents that occurred in
2020-2024. One CSV file per macro-region (semicolon-separated):

| Macro-region | File | Records |
|---|---|---|
| North-West | NordOvestAT.csv | 920,662 |
| North-East | NordEstAT.csv | 940,296 |
| Center | CentroAT.csv | 578,479 |
| South and Islands | SudAT.csv | 584,255 |

Source: https://dati.inail.it/portale/it/dataset/infortuni-sul-lavoro.html
Licence: Creative Commons Attribution 4.0 (CC BY 4.0), (c) INAIL.

INAIL publishes a rolling five-year window, so the files currently on the portal may differ
from the release analysed here. The exact files used in the article are archived on Zenodo:
https://doi.org/10.5281/zenodo.23945092

SHA-256 checksums of the files analysed (check with `certutil -hashfile <file> SHA256` on
Windows or `sha256sum <file>` on Linux/macOS):

| File | SHA-256 |
|---|---|
| NordOvestAT.csv | 2fb2f4e4ddae4902bb637418bf25ddf8eaeda7b137b1a0620e499ca6b93d6dd3 |
| NordEstAT.csv | c3e8fb40f943911331a24cf04e44f67127713199297d80309947357e62ac1365 |
| CentroAT.csv | 36249acb0ab14aa1f0597f62753f0c18722c27b916ca1b528cdb3e6fb554fc33 |
| SudAT.csv | 9847362f1fc8ef43e6e4fe5411e9e36c56112c3fac45be4f82cd3366f4419706 |

Create the subfolder `data/raw/` and place the four files there (unzip them first if they
were downloaded as an archive), or set `data_dir` in `config.yaml` to the folder that
contains them.

Columns used: DataAccadimento, Genere, Eta, CodiceAteco, SettoreAttivitaEconomica,
GrandeGruppoTariffario, LuogoAccadimento, Indennizzo, ModalitaAccadimento,
PosizioneAssicurativaTerritoriale, IdentificativoInfortunato.
