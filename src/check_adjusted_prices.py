
from pathlib import Path
import pandas as pd

for ticker in ["EEM", "SPY"]:
    files = sorted(Path("data/raw").glob(f"{ticker}_*.csv"))
    df = pd.read_csv(files[-1])

    difference = (df["Close"] - df["Adj Close"]).abs()
    ratio = df["Adj Close"] / df["Close"]

    print(f"\n=== {ticker} ===")
    print("Nombre d'observations :", len(df))
    print("Dates avec Close différent de Adj Close :",
          (difference > 0.000001).sum())
    print("Ratio ajusté / non ajusté minimum :",
          round(ratio.min(), 6))
    print("Ratio ajusté / non ajusté maximum :",
          round(ratio.max(), 6))
    print("\nExemple des 3 premières lignes :")
    print(df[["Date", "Close", "Adj Close"]].head(3).to_string(index=False))

