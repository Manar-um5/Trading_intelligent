
from pathlib import Path

import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "EEM_train.csv"
REPORTS = ROOT / "reports"

df = pd.read_csv(DATA)
rendements = df["rendement"].dropna().astype(float)

modeles = {
    "GARCH": arch_model(
        rendements, mean="Constant",
        vol="GARCH", p=1, o=0, q=1, dist="t"
    ),
    "GJR-GARCH": arch_model(
        rendements, mean="Constant",
        vol="GARCH", p=1, o=1, q=1, dist="t"
    )
}

resultats = []

for nom, modele in modeles.items():
    estimation = modele.fit(disp="off")

    residus = estimation.std_resid.dropna()

    lm, p_arch, _, _ = het_arch(
        residus, nlags=10, result_object=False
    )

    lb = acorr_ljungbox(
        residus ** 2, lags=[10], return_df=True
    )

    resultats.append({
        "modele": nom,
        "ARCH_LM_pvalue": p_arch,
        "Ljung_Box_carres_pvalue": lb["lb_pvalue"].iloc[0]
    })

comparaison = pd.DataFrame(resultats)

print("\n=== D04 : DIAGNOSTIC DES RESIDUS ===")
print(comparaison.to_string(index=False))

comparaison.to_csv(
    REPORTS / "D04_diagnostic_residus.csv",
    index=False
)
