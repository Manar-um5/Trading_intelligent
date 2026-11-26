
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

train = pd.read_csv(DATA / "D05_EEM_train.csv")
val = pd.read_csv(DATA / "D05_EEM_validation.csv")

y_train = train["cible_rendement"]
y_val = val["cible_rendement"]

# Baseline 1 : rendement prévu égal à zéro
prediction_zero = np.zeros(len(val))

# Baseline 2 : moyenne des rendements d'entraînement
prediction_moyenne = np.full(
    len(val),
    y_train.mean()
)

resultats = []

for nom, prediction in [
    ("Baseline_zero", prediction_zero),
    ("Baseline_moyenne", prediction_moyenne)
]:
    mse = mean_squared_error(y_val, prediction)

    resultats.append({
        "modele": nom,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_val, prediction)
    })

df_resultats = pd.DataFrame(resultats)

print("\n=== D05 : BASELINES ===")
print(df_resultats.to_string(index=False))

df_resultats.to_csv(
    REPORTS / "D05_resultats_baselines.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
