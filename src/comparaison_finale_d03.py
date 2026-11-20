import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

train = pd.read_csv("data/processed/EEM_train.csv")
val = pd.read_csv("data/processed/EEM_validation.csv")

y_train = train["rendement"].astype(float).to_numpy()
y_val = val["rendement"].astype(float).to_numpy()

# Modèle naïf : moyenne historique fixe
pred_naif = np.full(len(y_val), np.mean(y_train))

# ARIMA : estimation sur train uniquement
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    modele = ARIMA(y_train, order=(3, 0, 1)).fit()

# Prévisions à un pas avec mise à jour des observations
historique = modele
pred_arima = []

for valeur in y_val:
    prediction = historique.forecast(steps=1)
    pred_arima.append(float(prediction[0]))
    historique = historique.append([valeur], refit=False)

pred_arima = np.array(pred_arima)

# Charger les prévisions SARIMAX
sarimax = pd.read_csv(
    "reports/D03_predictions_SARIMAX_EEM.csv"
)

assert np.array_equal(
    val["Date"].to_numpy(),
    sarimax["Date"].to_numpy()
), "Les dates ne correspondent pas."

pred_sarimax = sarimax["prediction_sarimax"].to_numpy()

def metriques(nom, predictions):
    return {
        "Modele": nom,
        "MAE": round(np.mean(np.abs(y_val - predictions)), 4),
        "RMSE": round(np.sqrt(np.mean((y_val - predictions)**2)), 4)
    }

resultats = pd.DataFrame([
    metriques("Naïf", pred_naif),
    metriques("ARIMA(3,0,1)", pred_arima),
    metriques("SARIMAX(3,0,1)+SPY_lag1", pred_sarimax)
]).sort_values("RMSE")

print("\n===== COMPARAISON À UN PAS D03 =====")
print(resultats.to_string(index=False))

resultats.to_csv(
    "reports/D03_comparaison_finale_EEM.csv",
    index=False
)

print("\nComparaison enregistrée.")
