import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

train = pd.read_csv("data/processed/EEM_train.csv")
validation = pd.read_csv("data/processed/EEM_validation.csv")

y_train = train["rendement"].astype(float)
y_val = validation["rendement"].astype(float)

print("Entraînement du modèle ARIMA(1,0,1)...")

model = ARIMA(y_train, order=(1, 0, 1))
result = model.fit()

print(result.summary())

# Prévisions hors échantillon sur toute la validation
predictions = result.forecast(steps=len(y_val))

mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\n===== RÉSULTATS ARIMA EEM =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print("RMSE modèle naïf : 0.9740")

pd.DataFrame({
    "Date": validation["Date"],
    "rendement_reel": y_val,
    "prediction_arima": np.asarray(predictions)
}).to_csv("reports/D03_predictions_ARIMA_EEM.csv", index=False)

print("\nPrévisions enregistrées dans reports/D03_predictions_ARIMA_EEM.csv")
