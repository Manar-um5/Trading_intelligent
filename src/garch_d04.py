
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Chemins du projet
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "EEM_train.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# Charger les données d'entraînement
df = pd.read_csv(DATA, parse_dates=["Date"])
df = df.dropna(subset=["rendement"]).sort_values("Date")

rendements = df["rendement"].astype(float)

# Modèle GARCH(1,1) avec moyenne constante
modele = arch_model(
    rendements,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="t"
)

resultat = modele.fit(disp="off")

# Afficher les résultats
print("\n=== D04 : MODELE GARCH(1,1) EEM ===")
print(resultat.summary())

# Volatilité conditionnelle estimée
df["volatilite_garch_pct"] = resultat.conditional_volatility.to_numpy()

# Enregistrer les résultats
df.to_csv(REPORTS / "D04_volatilite_GARCH.csv", index=False)

# Enregistrer les paramètres du modèle
resultat.params.to_csv(
    REPORTS / "D04_parametres_GARCH.csv",
    header=["valeur"]
)

# Graphique
plt.figure(figsize=(12, 5))
plt.plot(
    df["Date"],
    df["volatilite_garch_pct"],
    label="Volatilité conditionnelle GARCH(1,1)"
)
plt.title("EEM - Volatilité conditionnelle GARCH(1,1)")
plt.xlabel("Date")
plt.ylabel("Volatilité journalière (%)")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS / "D04_volatilite_GARCH.png", dpi=200)
plt.close()

print("\nRésultats enregistrés dans reports/")
