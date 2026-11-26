
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

# Charger les rendements EEM
df = pd.read_csv(
    DATA / "EEM_rendements.csv",
    parse_dates=["Date"]
)

df = df.sort_values("Date").reset_index(drop=True)

# Variables explicatives : informations du jour
# et des jours précédents
for lag in [1, 2, 3, 5]:
    df[f"rendement_lag_{lag}"] = (
        df["rendement"].shift(lag)
    )

# Volatilité historique sur 5 et 20 jours
for fenetre in [5, 20]:
    df[f"volatilite_{fenetre}j"] = (
        df["rendement"].rolling(fenetre).std()
    )

# Moyennes mobiles des rendements
for fenetre in [5, 20]:
    df[f"moyenne_{fenetre}j"] = (
        df["rendement"].rolling(fenetre).mean()
    )

# Cible : rendement du lendemain
df["cible_rendement"] = df["rendement"].shift(-1)

# Cible classification : hausse = 1, baisse ou zéro = 0
df["cible_direction"] = (
    df["cible_rendement"] > 0
).astype(int)

# Éliminer les lignes sans historique suffisant
# et la dernière ligne sans cible future connue
df = df.dropna(
    subset=[
        "cible_rendement",
        "volatilite_20j",
        "moyenne_20j"
    ]
).reset_index(drop=True)

# Découpage chronologique selon la date de la cible :
# une cible de 2023 ne doit pas entrer dans train.
df["date_cible"] = df["Date"].shift(-1)

# La dernière date_cible doit provenir du fichier
# initial : recalcul explicite par calendrier observé.
dates = pd.read_csv(
    DATA / "EEM_rendements.csv",
    usecols=["Date"],
    parse_dates=["Date"]
).sort_values("Date").reset_index(drop=True)

dates["date_cible"] = dates["Date"].shift(-1)
df = df.drop(columns=["date_cible"]).merge(
    dates,
    on="Date",
    how="left",
    validate="one_to_one"
)

train = df[df["date_cible"] < "2023-01-01"]
validation = df[
    (df["date_cible"] >= "2023-01-01")
    & (df["date_cible"] < "2025-01-01")
]
test = df[df["date_cible"] >= "2025-01-01"]

for nom, partie in [
    ("train", train),
    ("validation", validation),
    ("test", test)
]:
    chemin = DATA / f"D05_EEM_{nom}.csv"
    partie.to_csv(chemin, index=False)

    print(
        f"{nom}: {len(partie)} observations, "
        f"cibles du {partie['date_cible'].min()} "
        f"au {partie['date_cible'].max()}"
    )

print("\nD05 : variables explicatives générées.")
