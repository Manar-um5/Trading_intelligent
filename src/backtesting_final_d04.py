
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

df = pd.read_csv(
    REPORTS / "D04_previsions_test_final.csv"
)

resultats = []

def log_binom(n0, n1):
    total = n0 + n1
    if total == 0:
        return 0.0

    p = n1 / total
    valeur = 0.0

    if n0 > 0:
        valeur += n0 * np.log1p(-p)
    if n1 > 0:
        valeur += n1 * np.log(p)

    return valeur

for niveau in [95, 99]:
    exceptions = (
        df[f"exception_{niveau}"]
        .astype(int)
        .to_numpy()
    )

    n00 = n01 = n10 = n11 = 0

    for precedent, actuel in zip(
        exceptions[:-1], exceptions[1:]
    ):
        if precedent == 0 and actuel == 0:
            n00 += 1
        elif precedent == 0 and actuel == 1:
            n01 += 1
        elif precedent == 1 and actuel == 0:
            n10 += 1
        else:
            n11 += 1

    # Test d'indépendance de Christoffersen
    log_l0 = log_binom(
        n00 + n10, n01 + n11
    )
    log_l1 = (
        log_binom(n00, n01)
        + log_binom(n10, n11)
    )

    lr_ind = 2 * (log_l1 - log_l0)
    p_ind = chi2.sf(lr_ind, df=1)

    # Couverture conditionnelle :
    # Kupiec + indépendance
    n = len(exceptions)
    x = int(exceptions.sum())
    p0 = 1 - niveau / 100
    phat = x / n

    def log_likelihood(p, x, n):
        if p == 0:
            return 0.0 if x == 0 else -np.inf
        if p == 1:
            return 0.0 if x == n else -np.inf
        return (
            x * np.log(p)
            + (n - x) * np.log1p(-p)
        )

    lr_uc = -2 * (
        log_likelihood(p0, x, n)
        - log_likelihood(phat, x, n)
    )

    lr_cc = lr_uc + lr_ind
    p_cc = chi2.sf(lr_cc, df=2)

    # Évaluation descriptive de l'ES
    pertes = -df["rendement"].to_numpy()
    es = df[f"ES_{niveau}_pct"].to_numpy()
    masque = exceptions.astype(bool)

    perte_moyenne = (
        np.mean(pertes[masque])
        if x > 0 else np.nan
    )
    es_moyen = (
        np.mean(es[masque])
        if x > 0 else np.nan
    )

    resultats.append({
        "niveau": niveau,
        "exceptions": x,
        "n11": n11,
        "LR_ind": lr_ind,
        "p_independance": p_ind,
        "LR_couverture_conditionnelle": lr_cc,
        "p_couverture_conditionnelle": p_cc,
        "perte_moyenne_pct": perte_moyenne,
        "ES_moyen_exceptions_pct": es_moyen,
        "ecart_perte_ES_pct": perte_moyenne - es_moyen
    })

resultats_df = pd.DataFrame(resultats)

print("\n=== D04 : BACKTESTING FINAL ===")
print(resultats_df.to_string(index=False))

resultats_df.to_csv(
    REPORTS / "D04_backtesting_final_complet.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
