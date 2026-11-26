
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

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
y_train = train["cible_direction"]

X_val = val[variables]
y_val = val["cible_direction"]

resultats = []

for C in [0.01, 0.1, 1, 10, 100]:
    modele = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=2000,
            random_state=42
        )
    )

    modele.fit(X_train, y_train)
    predictions = modele.predict(X_val)

    resultats.append({
        "C": C,
        "Accuracy": accuracy_score(y_val, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_val, predictions
        ),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

resultats_df = pd.DataFrame(resultats)
resultats_df = resultats_df.sort_values(
    ["Balanced_Accuracy", "F1"],
    ascending=False
)

print("\n=== D05 : LOGISTIC REGRESSION ===")
print(resultats_df.to_string(index=False))

print("\nMeilleur C :", resultats_df.iloc[0]["C"])

# Baseline classification : classe majoritaire
classe_majoritaire = int(y_train.mode().iloc[0])
prediction_baseline = np.full(
    len(y_val), classe_majoritaire
)

print("\n=== BASELINE CLASSIFICATION ===")
print("Classe majoritaire :", classe_majoritaire)
print(
    "Accuracy :",
    accuracy_score(y_val, prediction_baseline)
)
print(
    "Balanced Accuracy :",
    balanced_accuracy_score(y_val, prediction_baseline)
)

resultats_df.to_csv(
    REPORTS / "D05_resultats_Logistic.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
