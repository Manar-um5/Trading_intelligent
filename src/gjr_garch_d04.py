
from pathlib import Path

import pandas as pd
from arch import arch_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "EEM_train.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# Données d'entraînement : 2018-2022
df = pd.read_csv(DATA)
rendements = df["rendement"].dropna().astype(float)

# Modèle symétrique GARCH(1,1)
garch = arch_model(
    rendements,
    mean="Constant",
    vol="GARCH",
    p=1,
    o=0,
    q=1,
    dist="t"
).fit(disp="off")

# Modèle asymétrique GJR-GARCH(1,1,1)
gjr = arch_model(
    rendements,
    mean="Constant",
    vol="GARCH",
    p=1,
    o=1,
    q=1,
    dist="t"
).fit(disp="off")

print("\n=== D04 : COMPARAISON GARCH / GJR-GARCH ===")

comparaison = pd.DataFrame({
    "modele": ["GARCH(1,1)", "GJR-GARCH(1,1,1)"],
    "AIC": [garch.aic, gjr.aic],
    "BIC": [garch.bic, gjr.bic],
    "log_vraisemblance": [
        garch.loglikelihood,
        gjr.loglikelihood
    ]
})

print(comparaison.to_string(index=False))

print("\n=== PARAMETRES GJR-GARCH ===")
print(gjr.summary())

gamma = gjr.params["gamma[1]"]
pvalue_gamma = gjr.pvalues["gamma[1]"]

print("\nGamma (asymétrie) :", gamma)
print("p-value gamma :", pvalue_gamma)

if gamma > 0 and pvalue_gamma < 0.05:
    print("Asymétrie positive significative :")
    print("les chocs négatifs augmentent davantage la variance.")
else:
    print("Effet asymétrique positif non démontré au seuil de 5 %.")

comparaison.to_csv(
    REPORTS / "D04_comparaison_GARCH_GJR.csv",
    index=False
)

gjr.params.to_csv(
    REPORTS / "D04_parametres_GJR_GARCH.csv",
    header=["valeur"]
)

print("\nRésultats enregistrés dans reports/")
