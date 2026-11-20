
from pathlib import Path
import pandas as pd

files = sorted(Path("data/raw").glob("*.csv"))

for file in files:
    df = pd.read_csv(file, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print("\n" + "=" * 45)
    print("AUDIT :", file.name)
    print("=" * 45)

    print("Nombre de lignes :", len(df))
    print("Dates dupliquées :", df["Date"].duplicated().sum())
    print("Valeurs manquantes :", df.isna().sum().sum())

    prices = ["Open", "High", "Low", "Close", "Adj Close"]

    invalid_prices = (df[prices] <= 0).any(axis=1).sum()
    print("Lignes avec prix invalides :", invalid_prices)

    invalid_ohlc = (
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    ).sum()

    print("Incohérences OHLC :", invalid_ohlc)
    print("Volumes négatifs :", (df["Volume"] < 0).sum())

    # Rendements logarithmiques à partir du prix ajusté
    import numpy as np
    returns = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

    extreme = (returns.abs() > 0.10).sum()

    print("Rendements journaliers > 10 % en absolu :", extreme)
    print("Rendement minimum :", round(returns.min() * 100, 2), "%")
    print("Rendement maximum :", round(returns.max() * 100, 2), "%")
    print("\nDates des chocs extrêmes :")
    chocs = df.loc[returns.abs() > 0.10, ["Date"]].copy()
    chocs["Rendement (%)"] = (returns[returns.abs() > 0.10] * 100).round(2)
    print(chocs.to_string(index=False))
print("\nAudit terminé.")

