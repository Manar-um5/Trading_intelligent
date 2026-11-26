
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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

configurations = [
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 200, "max_depth": 3},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 300, "max_depth": 5},
    {"n_estimators": 300, "max_depth": 8}
]

resultats = []

for config in configurations:
    modele = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    modele.fit(X_train, y_train)
    predictions = modele.predict(X_val)

    resultats.append({
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "Accuracy": accuracy_score(y_val, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_val, predictions
        ),
        "F1": f1_score(
            y_val, predictions,
            zero_division=0
        )
    })

resultats_df = pd.DataFrame(resultats)
resultats_df = resultats_df.sort_values(
    ["Balanced_Accuracy", "F1"],
    ascending=False
)

print("\n=== D05 : RANDOM FOREST ===")
print(resultats_df.to_string(index=False))

print("\nMeilleure configuration :")
print(resultats_df.iloc[0].to_string())

resultats_df.to_csv(
    REPORTS / "D05_resultats_RandomForest.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
