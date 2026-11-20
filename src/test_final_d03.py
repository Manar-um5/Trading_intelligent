import numpy as np
import pandas as pd

train = pd.read_csv("data/processed/EEM_train.csv")
validation = pd.read_csv("data/processed/EEM_validation.csv")
test = pd.read_csv("data/processed/EEM_test.csv")

# Réentraînement du modèle sélectionné sur train + validation
historique = pd.concat(
    [train, validation], ignore_index=True
)

moyenne = historique["rendement"].mean()

# Prévisions constantes sur le test, sans utiliser ses rendements
reel = test["rendement"].to_numpy()
predictions = np.full(len(reel), moyenne)

mae = np.mean(np.abs(reel - predictions))
rmse = np.sqrt(np.mean((reel - predictions) ** 2))

print("\n===== TEST FINAL D03 — EEM =====")
print("Modèle sélectionné : Naïf (moyenne historique)")
print("Observations test :", len(test))
print(f"Moyenne estimée : {moyenne:.6f} %")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

pd.DataFrame({
    "Date": test["Date"],
    "rendement_reel": reel,
    "prediction_naive": predictions
}).to_csv(
    "reports/D03_predictions_test_final_EEM.csv",
    index=False
)

print("\nPrévisions enregistrées.")
