
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "D05_EEM_train.csv")
val = pd.read_csv(DATA / "D05_EEM_validation.csv")
test = pd.read_csv(DATA / "D05_EEM_test.csv")

historique = pd.concat([train, val], ignore_index=True)

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

X_train = historique[variables]
X_test = test[variables]

y_train_reg = historique["cible_rendement"]
y_test_reg = test["cible_rendement"]

y_train_cls = historique["cible_direction"]
y_test_cls = test["cible_direction"]

# Régression : Ridge et baseline
ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=100)
)
ridge.fit(X_train, y_train_reg)

pred_ridge = ridge.predict(X_test)
pred_zero = np.zeros(len(test))

resultats_reg = []

for nom, pred in [
    ("Ridge", pred_ridge),
    ("Baseline_zero", pred_zero)
]:
    resultats_reg.append({
        "modele": nom,
        "RMSE": np.sqrt(mean_squared_error(y_test_reg, pred)),
        "MAE": mean_absolute_error(y_test_reg, pred)
    })

# Classification : paramètres choisis en validation
modeles = {
    "Logistic": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1,
            max_iter=2000,
            random_state=42
        )
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ),
    "MLP": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.01,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        )
    )
}

resultats_cls = []
predictions = pd.DataFrame({
    "Date": test["date_cible"],
    "rendement_reel": y_test_reg,
    "direction_reelle": y_test_cls
})

classe_majoritaire = int(y_train_cls.mode().iloc[0])

pred_baseline = np.full(
    len(test), classe_majoritaire
)

for nom, modele in modeles.items():
    modele.fit(X_train, y_train_cls)
    pred = modele.predict(X_test)

    predictions[f"prediction_{nom}"] = pred

    resultats_cls.append({
        "modele": nom,
        "Accuracy": accuracy_score(y_test_cls, pred),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_test_cls, pred
        ),
        "F1": f1_score(y_test_cls, pred, zero_division=0)
    })

resultats_cls.append({
    "modele": "Baseline_majoritaire",
    "Accuracy": accuracy_score(y_test_cls, pred_baseline),
    "Balanced_Accuracy": balanced_accuracy_score(
        y_test_cls, pred_baseline
    ),
    "F1": f1_score(
        y_test_cls, pred_baseline,
        zero_division=0
    )
})

df_reg = pd.DataFrame(resultats_reg)
df_cls = pd.DataFrame(resultats_cls)

print("\n=== D05 : TEST FINAL REGRESSION ===")
print(df_reg.to_string(index=False))

print("\n=== D05 : TEST FINAL CLASSIFICATION ===")
print(df_cls.to_string(index=False))

df_reg.to_csv(
    REPORTS / "D05_test_final_regression.csv",
    index=False
)
df_cls.to_csv(
    REPORTS / "D05_test_final_classification.csv",
    index=False
)
predictions.to_csv(
    REPORTS / "D05_predictions_test.csv",
    index=False
)

print("\nRésultats enregistrés dans reports/")
