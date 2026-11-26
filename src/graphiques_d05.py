
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures_D05"
FIGURES.mkdir(parents=True, exist_ok=True)

# Résultats déjà calculés
reg = pd.read_csv(
    REPORTS / "D05_test_final_regression.csv"
)
cls = pd.read_csv(
    REPORTS / "D05_test_final_classification.csv"
)

# 1. Comparaison RMSE
plt.figure(figsize=(8, 5))
plt.bar(reg["modele"], reg["RMSE"])
plt.title("D05 - Erreur RMSE sur le test final")
plt.ylabel("RMSE (points de pourcentage)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "01_RMSE.png", dpi=200)
plt.close()

# 2. Accuracy et Balanced Accuracy
x = np.arange(len(cls))
largeur = 0.35

plt.figure(figsize=(10, 5))
plt.bar(
    x - largeur / 2,
    cls["Accuracy"] * 100,
    largeur,
    label="Accuracy"
)
plt.bar(
    x + largeur / 2,
    cls["Balanced_Accuracy"] * 100,
    largeur,
    label="Balanced Accuracy"
)

plt.xticks(x, cls["modele"], rotation=15)
plt.ylabel("Score (%)")
plt.title("D05 - Comparaison des modèles de classification")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES / "02_classification.png",
    dpi=200
)
plt.close()

# 3. Matrices de confusion
train = pd.read_csv(DATA / "D05_EEM_train.csv")
val = pd.read_csv(DATA / "D05_EEM_validation.csv")
test = pd.read_csv(DATA / "D05_EEM_test.csv")

historique = pd.concat(
    [train, val], ignore_index=True
)

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
y_train = historique["cible_direction"]

X_test = test[variables]
y_test = test["cible_direction"]

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

for nom, modele in modeles.items():
    modele.fit(X_train, y_train)
    pred = modele.predict(X_test)

    matrice = confusion_matrix(
        y_test, pred, labels=[0, 1]
    )

    affichage = ConfusionMatrixDisplay(
        confusion_matrix=matrice,
        display_labels=["Baisse/stable", "Hausse"]
    )

    affichage.plot(cmap="Blues", values_format="d")
    plt.title(f"D05 - Matrice de confusion : {nom}")
    plt.tight_layout()
    plt.savefig(
        FIGURES / f"03_confusion_{nom}.png",
        dpi=200
    )
    plt.close()

print("\n=== GRAPHIQUES D05 GENERES ===")
for fichier in sorted(FIGURES.glob("*.png")):
    print(fichier.name)
