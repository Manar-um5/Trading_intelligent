
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

df = pd.read_csv(REPORTS / "D04_previsions_VaR_ES.csv")

# Rendements en % : une perte correspond à -rendement
pertes = -df["rendement"].to_numpy()

resultats = []

for niveau in [95, 99]:
    exceptions = (
        df[f"exception_{niveau}"]
        .astype(bool)
        .to_numpy()
    )

    var = df[f"VaR_{niveau}_pct"].to_numpy()
    es = df[f"ES_{niveau}_pct"].to_numpy()

    n = int(exceptions.sum())

    if n > 0:
        perte_moyenne = np.mean(pertes[exceptions])
        es_moyen_exceptions = np.mean(es[exceptions])
        ecart = perte_moyenne - es_moyen_exceptions
    else:
        perte_moyenne = np.nan
        es_moyen_exceptions = np.nan
        ecart = np.nan

    resultats.append({
        "niveau": niveau,
        "exceptions": n,
        "perte_moyenne_depassements_pct": perte_moyenne,
        "ES_moyen_depassements_pct": es_moyen_exceptions,
        "ecart_perte_moins_ES_pct": ecart
    })

resultats_df = pd.DataFrame(resultats)

print("\n=== D04 : EVALUATION DE L'EXPECTED SHORTFALL ===")
print(resultats_df.to_string(index=False))

resultats_df.to_csv(
    REPORTS / "D04_evaluation_ES.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
