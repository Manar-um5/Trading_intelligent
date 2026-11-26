
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, chi2
from arch import arch_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "EEM_train.csv")
val = pd.read_csv(DATA / "EEM_validation.csv")
test = pd.read_csv(DATA / "EEM_test.csv")

# Réestimation sur train + validation
historique = pd.concat(
    [train, val], ignore_index=True
)
r_hist = historique["rendement"].astype(float)
r_test = test["rendement"].astype(float)

modele = arch_model(
    r_hist,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="t"
)
fit = modele.fit(disp="off")

mu = float(fit.params["mu"])
omega = float(fit.params["omega"])
alpha = float(fit.params["alpha[1]"])
beta = float(fit.params["beta[1]"])
nu = float(fit.params["nu"])

# Dernière variance conditionnelle d'entraînement
h = float(fit.conditional_volatility.iloc[-1] ** 2)
dernier_residu = float(r_hist.iloc[-1] - mu)

previsions = []

# Prévisions à un jour, sans réestimation sur test
for rendement in r_test:
    h = omega + alpha * dernier_residu**2 + beta * h
    sigma = np.sqrt(h)

    ligne = {
        "volatilite_prevue_pct": sigma,
        "rendement": rendement
    }

    for confiance in [0.95, 0.99]:
        niveau = int(confiance * 100)
        proba = 1 - confiance

        q = t.ppf(proba, nu)
        facteur = np.sqrt((nu - 2) / nu)
        z = facteur * q

        var = -(mu + sigma * z)

        es_z = (
            -facteur * t.pdf(q, nu)
            * (nu + q**2)
            / ((nu - 1) * proba)
        )
        es = -(mu + sigma * es_z)

        ligne[f"VaR_{niveau}_pct"] = var
        ligne[f"ES_{niveau}_pct"] = es
        ligne[f"exception_{niveau}"] = rendement < -var

    previsions.append(ligne)

    # Le rendement du jour devient disponible
    # seulement pour prévoir le jour suivant
    dernier_residu = rendement - mu

resultats = pd.DataFrame(previsions)
resultats.insert(0, "Date", test["Date"].to_numpy())

# Évaluation de la variance prévue
variance_realisee = (r_test.to_numpy() - mu)**2
variance_prevue = resultats["volatilite_prevue_pct"].to_numpy()**2

qlike = np.mean(
    np.log(variance_prevue)
    + variance_realisee / variance_prevue
)
mse = np.mean((variance_prevue - variance_realisee)**2)

print("\n=== D04 : TEST FINAL EEM 2025-2026 ===")
print("Observations :", len(resultats))
print("QLIKE :", round(qlike, 6))
print("MSE variance :", round(mse, 6))

resume = []

for confiance in [0.95, 0.99]:
    niveau = int(confiance * 100)
    exceptions = resultats[f"exception_{niveau}"]
    n = len(exceptions)
    x = int(exceptions.sum())
    p0 = 1 - confiance
    phat = x / n

    # Test de Kupiec, avec gestion des cas limites
    def log_likelihood(p, x, n):
        if p == 0:
            return 0.0 if x == 0 else -np.inf
        if p == 1:
            return 0.0 if x == n else -np.inf
        return x * np.log(p) + (n-x) * np.log1p(-p)

    lr = -2 * (
        log_likelihood(p0, x, n)
        - log_likelihood(phat, x, n)
    )
    p_value = chi2.sf(lr, 1)

    resume.append({
        "confiance": confiance,
        "observations": n,
        "exceptions": x,
        "taux_observe": phat,
        "taux_attendu": p0,
        "VaR_moyenne_pct": resultats[f"VaR_{niveau}_pct"].mean(),
        "ES_moyen_pct": resultats[f"ES_{niveau}_pct"].mean(),
        "Kupiec_pvalue": p_value
    })

resume_df = pd.DataFrame(resume)

print("\n=== BACKTESTING FINAL VaR ===")
print(resume_df.to_string(index=False))

resultats.to_csv(
    REPORTS / "D04_previsions_test_final.csv",
    index=False
)
resume_df.to_csv(
    REPORTS / "D04_resultats_test_final.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
