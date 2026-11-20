import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

# Graphique 1 : comparaison des modèles sur validation
modeles = ["Naïf", "SARIMAX", "SARIMA", "ARIMA"]
mae = [0.7519, 0.7609, 0.7623, 0.7630]
rmse = [0.9740, 0.9821, 0.9840, 0.9843]

x = np.arange(len(modeles))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, mae, 0.4, label="MAE")
ax.bar(x + 0.2, rmse, 0.4, label="RMSE")
ax.set_xticks(x, modeles)
ax.set_ylabel("Erreur (points de pourcentage)")
ax.set_title("D03 - Comparaison des modèles EEM (2023-2024)")
ax.legend()
fig.tight_layout()
fig.savefig("reports/D03_comparaison_modeles.png", dpi=200)
plt.close(fig)

# Graphique 2 : prévisions du modèle retenu sur le test
df = pd.read_csv(
    "reports/D03_predictions_test_final_EEM.csv",
    parse_dates=["Date"]
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Date"], df["rendement_reel"],
        label="Rendement réel", linewidth=0.8)
ax.plot(df["Date"], df["prediction_naive"],
        label="Prévision naïve", linewidth=1.5)
ax.set_title("D03 - Prévisions EEM sur le test (2025-2026)")
ax.set_xlabel("Date")
ax.set_ylabel("Rendement journalier (%)")
ax.legend()
fig.tight_layout()
fig.savefig("reports/D03_previsions_test_EEM.png", dpi=200)
plt.close(fig)

print("Graphiques générés :")
print("reports/D03_comparaison_modeles.png")
print("reports/D03_previsions_test_EEM.png")
