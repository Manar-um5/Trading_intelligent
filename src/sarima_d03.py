import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

train = pd.read_csv("data/processed/EEM_train.csv")
validation = pd.read_csv("data/processed/EEM_validation.csv")

y_train = train["rendement"].astype(float)
y_val = validation["rendement"].astype(float).to_numpy()

configurations = [
    ((1,0,1), (1,0,0,5)),
    ((1,0,1), (0,0,1,5)),
    ((2,0,1), (1,0,0,5)),
    ((3,0,1), (1,0,0,5))
]

resultats = []

for ordre, saisonnier in configurations:
    print(f"Test SARIMA{ordre} x {saisonnier}", flush=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modele = SARIMAX(
                y_train,
                order=ordre,
                seasonal_order=saisonnier,
                trend="c",
                enforce_stationarity=True,
                enforce_invertibility=True
            ).fit(disp=False, maxiter=200)

        predictions = np.asarray(
            modele.forecast(steps=len(y_val))
        )

        mae = np.mean(np.abs(y_val - predictions))
        rmse = np.sqrt(np.mean((y_val - predictions)**2))

        lb = acorr_ljungbox(
            modele.resid,
            lags=[10, 20],
            return_df=True
        )

        resultats.append({
            "Ordre": str(ordre),
            "Saisonnier": str(saisonnier),
            "AIC": round(modele.aic, 2),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "LB_p10": round(lb.loc[10, "lb_pvalue"], 6),
            "LB_p20": round(lb.loc[20, "lb_pvalue"], 6),
            "Convergence": bool(
                modele.mle_retvals.get("converged", False)
            )
        })

    except Exception as e:
        print("Erreur :", e)

df = pd.DataFrame(resultats).sort_values("RMSE")

print("\n===== COMPARAISON SARIMA EEM =====")
print(df.to_string(index=False))

df.to_csv("reports/D03_comparaison_SARIMA_EEM.csv", index=False)

print("\nRésultats enregistrés.")
