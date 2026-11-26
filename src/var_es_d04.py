
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from arch import arch_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# Charger les données et prévisions
train = pd.read_csv(DATA / "EEM_train.csv")
previsions = pd.read_csv(
    REPORTS / "D04_previsions_validation.csv"
)

r_train = train["rendement"].dropna().astype(float)

# Réestimer GARCH uniquement sur l'entraînement
modele = arch_model(
    r_train,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="t"
)
resultat = modele.fit(disp="off")

mu = float(resultat.params["mu"])
nu = float(resultat.params["nu"])

# Volatilité prévue à un jour, en %
sigma = previsions["volatilite_GARCH"].to_numpy()
rendements = previsions["rendement"].to_numpy()

if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
    raise ValueError("Volatilités prévues invalides.")

# Student-t standardisée à variance unitaire
facteur = np.sqrt((nu - 2) / nu)

resultats = []

for confiance in [0.95, 0.99]:
    alpha = 1 - confiance

    quantile = t.ppf(alpha, df=nu)
    z = facteur * quantile

    # VaR positive : seuil de perte en %
    var = -(mu + sigma * z)

    # Expected Shortfall de la queue gauche
    es_z = (
        -facteur
        * t.pdf(quantile, df=nu)
        * (nu + quantile**2)
        / ((nu - 1) * alpha)
    )
    es = -(mu + sigma * es_z)

    # Une exception se produit si le rendement
    # réalisé est inférieur au seuil prévu
    exceptions = rendements < -var

    taux = exceptions.mean()

    resultats.append({
        "confiance": confiance,
        "observations": len(rendements),
        "exceptions": int(exceptions.sum()),
        "taux_observe": taux,
        "taux_attendu": alpha,
        "VaR_moyenne_pct": var.mean(),
        "ES_moyen_pct": es.mean()
    })

    niveau = int(confiance * 100)
    previsions[f"VaR_{niveau}_pct"] = var
    previsions[f"ES_{niveau}_pct"] = es
    previsions[f"exception_{niveau}"] = exceptions

comparaison = pd.DataFrame(resultats)

print("\n=== D04 : VaR ET EXPECTED SHORTFALL ===")
print(comparaison.to_string(index=False))

comparaison.to_csv(
    REPORTS / "D04_resultats_VaR_ES.csv",
    index=False
)

previsions.to_csv(
    REPORTS / "D04_previsions_VaR_ES.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
