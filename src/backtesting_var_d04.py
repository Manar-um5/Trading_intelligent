
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

df = pd.read_csv(
    REPORTS / "D04_previsions_VaR_ES.csv"
)

resultats = []

for confiance in [0.95, 0.99]:
    niveau = int(confiance * 100)
    exceptions = df[f"exception_{niveau}"].astype(int)

    n = len(exceptions)
    x = int(exceptions.sum())

    p_attendu = 1 - confiance
    p_observe = x / n

    # Test de Kupiec : couverture inconditionnelle
    def log_vraisemblance(p, x, n):
        if p == 0:
            return 0.0 if x == 0 else -np.inf
        if p == 1:
            return 0.0 if x == n else -np.inf
        return (
            x * np.log(p)
            + (n - x) * np.log(1 - p)
        )

    lr_uc = -2 * (
        log_vraisemblance(p_attendu, x, n)
        - log_vraisemblance(p_observe, x, n)
    )

    p_value = chi2.sf(lr_uc, df=1)

    resultats.append({
        "confiance": confiance,
        "observations": n,
        "exceptions": x,
        "taux_attendu": p_attendu,
        "taux_observe": p_observe,
        "LR_Kupiec": lr_uc,
        "p_value": p_value,
        "rejet_5pct": p_value < 0.05
    })

resultats_df = pd.DataFrame(resultats)

print("\n=== D04 : BACKTESTING VAR - KUPIEC ===")
print(resultats_df.to_string(index=False))

resultats_df.to_csv(
    REPORTS / "D04_backtesting_Kupiec.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
