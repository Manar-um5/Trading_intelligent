
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

df = pd.read_csv(REPORTS / "D04_previsions_VaR_ES.csv")

def log_binom(n0, n1):
    total = n0 + n1
    if total == 0:
        return 0.0
    p = n1 / total
    result = 0.0
    if n0 > 0:
        result += n0 * np.log1p(-p)
    if n1 > 0:
        result += n1 * np.log(p)
    return result

resultats = []

for confiance in [0.95, 0.99]:
    niveau = int(confiance * 100)
    e = df[f"exception_{niveau}"].astype(int).to_numpy()

    n00 = n01 = n10 = n11 = 0

    for precedent, actuel in zip(e[:-1], e[1:]):
        if precedent == 0 and actuel == 0:
            n00 += 1
        elif precedent == 0 and actuel == 1:
            n01 += 1
        elif precedent == 1 and actuel == 0:
            n10 += 1
        else:
            n11 += 1

    # Hypothèse nulle : transitions indépendantes
    log_l0 = log_binom(n00 + n10, n01 + n11)

    # Hypothèse alternative : probabilités distinctes
    log_l1 = (
        log_binom(n00, n01)
        + log_binom(n10, n11)
    )

    lr_ind = 2 * (log_l1 - log_l0)
    p_value = chi2.sf(lr_ind, df=1)

    resultats.append({
        "confiance": confiance,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "LR_independance": lr_ind,
        "p_value": p_value,
        "rejet_5pct": p_value < 0.05
    })

resultats_df = pd.DataFrame(resultats)

print("\n=== D04 : CHRISTOFFERSEN - INDEPENDANCE ===")
print(resultats_df.to_string(index=False))

resultats_df.to_csv(
    REPORTS / "D04_Christoffersen.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
