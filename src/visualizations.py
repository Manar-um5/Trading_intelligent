
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = Path("data/raw")
output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

def load_data(ticker):
    files = sorted(data_dir.glob(f"{ticker}_*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier pour {ticker}")

    df = pd.read_csv(files[-1], parse_dates=["Date"])
    return df.sort_values("Date").set_index("Date")

eem = load_data("EEM")
spy = load_data("SPY")

# Graphique 1 : prix ajustés
plt.figure(figsize=(12, 5))
plt.plot(eem.index, eem["Adj Close"], label="EEM")
plt.plot(spy.index, spy["Adj Close"], label="SPY")
plt.title("Evolution des prix ajustes - EEM et SPY")
plt.xlabel("Date")
plt.ylabel("Prix ajuste (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "01_prix_ajustes.png", dpi=200)
plt.close()

# Graphique 2 : rendements EEM
returns = 100 * np.log(
    eem["Adj Close"] / eem["Adj Close"].shift(1)
)

plt.figure(figsize=(12, 5))
plt.plot(returns.index, returns, linewidth=0.7)
plt.axhline(-10, color="red", linestyle="--",
            label="Seuil -10 %")
plt.axhline(10, color="red", linestyle="--")
plt.title("Rendements logarithmiques journaliers - EEM")
plt.xlabel("Date")
plt.ylabel("Rendement (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "02_rendements_eem.png", dpi=200)
plt.close()

# Graphique 3 : performances normalisées
common = pd.concat(
    [eem["Adj Close"], spy["Adj Close"]],
    axis=1,
    keys=["EEM", "SPY"]
).dropna()

normalized = common.div(common.iloc[0]).mul(100)

plt.figure(figsize=(12, 5))
plt.plot(normalized.index, normalized["EEM"], label="EEM")
plt.plot(normalized.index, normalized["SPY"], label="SPY")
plt.title("Comparaison EEM / SPY - Base 100")
plt.xlabel("Date")
plt.ylabel("Indice base 100")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "03_comparaison_base100.png", dpi=200)
plt.close()

print("Trois graphiques crees dans reports/")

