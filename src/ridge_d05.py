
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "D05_EEM_train.csv")
val = pd.read_csv(DATA / "D05_EEM_validation.csv")

variables = [
    "rendement",
    "rendement_lag_1",
    "rendement_lag_2",
    "rendement_lag_3",
    "rendement_lag_5",
    "volatilite_5j",
    "volatilite_20j",
    "moyenne_5j",
    "moyenne_20j"
]

X_train = train[variables]
y_train = train["cible_rendement"]

X_val = val[variables]
y_val = val["cible_rendement"]

resultats = []

for alpha in [0.01, 0.1, 1, 10, 100]:
    modele = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha)
    )

    modele.fit(X_train, y_train)
    predictions = modele.predict(X_val)

    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, predictions)

    resultats.append({
        "alpha": alpha,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    })

resultats_df = pd.DataFrame(resultats)
resultats_df = resultats_df.sort_values("RMSE")

print("\n=== D05 : RIDGE REGRESSION ===")
print(resultats_df.to_string(index=False))

meilleur_alpha = resultats_df.iloc[0]["alpha"]

print("\nMeilleur alpha :", meilleur_alpha)

resultats_df.to_csv(
    REPORTS / "D05_resultats_Ridge.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
