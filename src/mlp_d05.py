
from pathlib import Path
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
    (16,),
    (32,),
    (32, 16),
    (64, 32)
]

resultats = []

for architecture in configurations:
    modele = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=architecture,
            activation="relu",
            solver="adam",
            alpha=0.01,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        )
    )

    modele.fit(X_train, y_train)
    predictions = modele.predict(X_val)

    resultats.append({
        "architecture": str(architecture),
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

print("\n=== D05 : RESEAU DE NEURONES MLP ===")
print(resultats_df.to_string(index=False))

print("\nMeilleure architecture :")
print(resultats_df.iloc[0].to_string())

resultats_df.to_csv(
    REPORTS / "D05_resultats_MLP.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
