import numpy as np
import pandas as pd
from pathlib import Path


folder = Path("data/processed")
reports = Path("reports")
reports.mkdir(exist_ok=True)

results = []

for ticker in ["EEM", "SPY"]:
    train = pd.read_csv(folder / f"{ticker}_train.csv")
    val = pd.read_csv(folder / f"{ticker}_validation.csv")

    # Moyenne historique calculée uniquement sur l'apprentissage
    prediction = train["rendement"].mean()

    y_true = val["rendement"].to_numpy()
    y_pred = np.full(len(val), prediction)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print(f"\n{ticker} — Modèle naïf")
    print(f"Prévision constante : {prediction:.4f} %")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    results.append({
        "Ticker": ticker,
        "Modele": "Naif_moyenne",
        "MAE": mae,
        "RMSE": rmse
    })

    pd.DataFrame({
        "Date": val["Date"],
        "Reel": y_true,
        "Prediction": y_pred
    }).to_csv(
        reports / f"D03_predictions_naif_{ticker}.csv",
        index=False
    )

pd.DataFrame(results).to_csv(
    reports / "D03_baseline_resultats.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
