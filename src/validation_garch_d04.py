
from pathlib import Path
import numpy as np
import pandas as pd
from arch import arch_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "EEM_train.csv")
validation = pd.read_csv(DATA / "EEM_validation.csv")

r_train = train["rendement"].dropna().astype(float)
r_val = validation["rendement"].dropna().astype(float)

# Estimation uniquement sur 2018-2022
specifications = {
    "GARCH": 0,
    "GJR-GARCH": 1
}

resultats = []
previsions = pd.DataFrame({
    "Date": validation["Date"],
    "rendement": r_val.to_numpy()
})

for nom, o in specifications.items():
    modele = arch_model(
        r_train,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=o,
        q=1,
        dist="t"
    )
    estimation = modele.fit(disp="off")

    # Paramètres fixes estimés sur l'entraînement.
    # Les rendements de validation sont révélés
    # progressivement, jour après jour.
    historique = pd.concat(
        [r_train, r_val], ignore_index=True
    )

    modele_complet = arch_model(
        historique,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=o,
        q=1,
        dist="t"
    )

    filtre = modele_complet.fix(estimation.params)

    # La variance filtrée à t-1 sert à prévoir t.
    residus = historique - estimation.params["mu"]
    variances = filtre.conditional_volatility.to_numpy() ** 2

    omega = estimation.params["omega"]
    alpha = estimation.params["alpha[1]"]
    beta = estimation.params["beta[1]"]
    gamma = estimation.params.get("gamma[1]", 0.0)

    variance_prevue = (
        omega
        + alpha * residus.to_numpy()[:-1] ** 2
        + gamma * (
            residus.to_numpy()[:-1] < 0
        ) * residus.to_numpy()[:-1] ** 2
        + beta * variances[:-1]
    )

    variance_val = variance_prevue[
        len(r_train) - 1:
    ]

    # Proxy de variance réalisée :
    # rendement centré au carré
    variance_realisee = (
        r_val.to_numpy() - estimation.params["mu"]
    ) ** 2

    # QLIKE : métrique de prévision de variance
    qlike = np.mean(
        np.log(variance_val)
        + variance_realisee / variance_val
    )

    mse = np.mean(
        (variance_val - variance_realisee) ** 2
    )

    resultats.append({
        "modele": nom,
        "QLIKE": qlike,
        "MSE_variance": mse
    })

    previsions[f"variance_{nom}"] = variance_val
    previsions[f"volatilite_{nom}"] = np.sqrt(variance_val)

comparaison = pd.DataFrame(resultats)

print("\n=== D04 : VALIDATION 2023-2024 ===")
print(comparaison.to_string(index=False))

comparaison.to_csv(
    REPORTS / "D04_validation_GARCH_GJR.csv",
    index=False
)

previsions.to_csv(
    REPORTS / "D04_previsions_validation.csv",
    index=False
)

print("\nFichiers enregistrés dans reports/")
