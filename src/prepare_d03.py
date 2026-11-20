import pandas as pd
from pathlib import Path

raw_dir = Path("data/raw")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

for ticker in ["EEM", "SPY"]:
    fichier = sorted(raw_dir.glob(f"{ticker}_*.csv"))[-1]

    df = pd.read_csv(fichier, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Rendement logarithmique journalier en pourcentage
    df["rendement"] = 100 * (
        df["Adj Close"] / df["Adj Close"].shift(1)
    ).apply("log") if False else None

    import numpy as np
    df["rendement"] = 100 * np.log(
        df["Adj Close"] / df["Adj Close"].shift(1)
    )

    df = df.dropna(subset=["rendement"])

    output = output_dir / f"{ticker}_rendements.csv"
    df[["Date", "Adj Close", "rendement"]].to_csv(
        output, index=False
    )

    print(f"\n{ticker}")
    print(f"Observations : {len(df)}")
    print(f"Début : {df['Date'].min().date()}")
    print(f"Fin : {df['Date'].max().date()}")
    print(f"Rendement moyen : {df['rendement'].mean():.4f} %")
    print(f"Écart-type : {df['rendement'].std():.4f} %")
    print(f"Fichier créé : {output}")
