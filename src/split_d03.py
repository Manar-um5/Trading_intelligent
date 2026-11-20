import pandas as pd
from pathlib import Path

folder = Path("data/processed")

for ticker in ["EEM", "SPY"]:
    df = pd.read_csv(
        folder / f"{ticker}_rendements.csv",
        parse_dates=["Date"]
    )

    train = df[df["Date"] < "2023-01-01"]
    validation = df[
        (df["Date"] >= "2023-01-01") &
        (df["Date"] < "2025-01-01")
    ]
    test = df[df["Date"] >= "2025-01-01"]

    for name, subset in [
        ("train", train),
        ("validation", validation),
        ("test", test)
    ]:
        output = folder / f"{ticker}_{name}.csv"
        subset.to_csv(output, index=False)

        print(
            f"{ticker} | {name} : "
            f"{len(subset)} observations "
            f"({subset['Date'].min().date()} "
            f"au {subset['Date'].max().date()})"
        )
