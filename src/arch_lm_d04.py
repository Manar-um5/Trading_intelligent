
from pathlib import Path

import pandas as pd
from statsmodels.stats.diagnostic import het_arch

# Chemins du projet
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "EEM_train.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# Charger les rendements d'entraînement (en %)
df = pd.read_csv(DATA, parse_dates=["Date"])
rendements = df["rendement"].dropna()

# Test ARCH-LM avec 5 et 10 retards
resultats = []

for lag in [5, 10]:
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(
        rendements - rendements.mean(),
        nlags=lag
    )

    resultats.append({
        "retards": lag,
        "LM_statistique": lm_stat,
        "LM_pvalue": lm_pvalue,
        "F_statistique": f_stat,
        "F_pvalue": f_pvalue
    })

resultats_df = pd.DataFrame(resultats)

print("\n=== D04 : TEST ARCH-LM SUR EEM ===")
print("Période :", df["Date"].min().date(),
      "au", df["Date"].max().date())
print("Observations :", len(rendements))
print(resultats_df.to_string(index=False))

# Sauvegarde pour le rapport D04
sortie = REPORTS / "D04_test_ARCH_LM.csv"
resultats_df.to_csv(sortie, index=False)

print("\nRésultats enregistrés dans :", sortie)
