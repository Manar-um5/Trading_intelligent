import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

train = pd.read_csv("data/processed/EEM_train.csv")
val = pd.read_csv("data/processed/EEM_validation.csv")

y_train = train["rendement"].astype(float).to_numpy()
y_val = val["rendement"].astype(float).to_numpy()

print("Entraînement SARIMA...", flush=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    modele = SARIMAX(
        y_train,
        order=(3, 0, 1),
        seasonal_order=(1, 0, 0, 5),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True
    ).fit(disp=False, maxiter=200)

print("Convergence :", modele.mle_retvals.get("converged"))

historique = modele
predictions = []

for valeur in y_val:
    prediction = historique.forecast(steps=1)
    predictions.append(float(prediction[0]))
    historique = historique.append([valeur], refit=False)

predictions = np.array(predictions)

mae = np.mean(np.abs(y_val - predictions))
rmse = np.sqrt(np.mean((y_val - predictions) ** 2))

print("\n===== SARIMA À UN PAS =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

pd.DataFrame({
    "Date": val["Date"],
    "rendement_reel": y_val,
    "prediction_sarima": predictions
}).to_csv(
    "reports/D03_predictions_SARIMA_EEM_1pas.csv",
    index=False
)

print("Prévisions SARIMA enregistrées.")
