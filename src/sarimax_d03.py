import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Charger et aligner EEM et SPY
eem = pd.concat([
    pd.read_csv("data/processed/EEM_train.csv"),
    pd.read_csv("data/processed/EEM_validation.csv")
], ignore_index=True)

spy = pd.concat([
    pd.read_csv("data/processed/SPY_train.csv"),
    pd.read_csv("data/processed/SPY_validation.csv")
], ignore_index=True)

df = eem[["Date", "rendement"]].merge(
    spy[["Date", "rendement"]],
    on="Date",
    suffixes=("_EEM", "_SPY"),
    validate="one_to_one"
).sort_values("Date").reset_index(drop=True)

# SPY de la séance précédente
df["SPY_retard_1"] = df["rendement_SPY"].shift(1)
df = df.dropna().reset_index(drop=True)

train = df[df["Date"] < "2023-01-01"].copy()
val = df[
    (df["Date"] >= "2023-01-01") &
    (df["Date"] < "2025-01-01")
].copy()

print("Entraînement :", len(train))
print("Validation :", len(val))

# SARIMAX(3,0,1) avec variable exogène SPY retardée
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    modele = SARIMAX(
        train["rendement_EEM"].to_numpy(),
        exog=train[["SPY_retard_1"]].to_numpy(),
        order=(3, 0, 1),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True
    ).fit(disp=False, maxiter=200)

print("Convergence :", modele.mle_retvals.get("converged"))
print("AIC :", round(modele.aic, 2))

# Prévision à un pas, avec mise à jour des observations
historique = modele
predictions = []

for _, ligne in val.iterrows():
    x = np.array([[ligne["SPY_retard_1"]]])

    prevision = historique.forecast(steps=1, exog=x)
    predictions.append(float(prevision[0]))

    # Actualisation de l'état avec la valeur observée
    historique = historique.append(
        endog=[ligne["rendement_EEM"]],
        exog=x,
        refit=False
    )

reel = val["rendement_EEM"].to_numpy()
pred = np.array(predictions)

mae = np.mean(np.abs(reel - pred))
rmse = np.sqrt(np.mean((reel - pred) ** 2))

print("\n===== RÉSULTATS SARIMAX EEM =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

pd.DataFrame({
    "Date": val["Date"],
    "rendement_reel": reel,
    "prediction_sarimax": pred
}).to_csv("reports/D03_predictions_SARIMAX_EEM.csv", index=False)

print("Prévisions enregistrées.")
