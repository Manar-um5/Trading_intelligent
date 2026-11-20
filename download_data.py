
from pathlib import Path
from datetime import datetime, timezone
import yfinance as yf

# Projet B08 - ETF EEM et benchmark SPY
START = "2018-01-01"
END = "2026-10-31"  # Fin exclusive

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

snapshot = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

for ticker in ["EEM", "SPY"]:
    print(f"\nTéléchargement de {ticker}...")

    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=False,
        progress=False,
        multi_level_index=False
    )

    if df.empty:
        print(f"ERREUR : aucune donnée pour {ticker}")
        continue

    df = df.reset_index()

    filename = output_dir / f"{ticker}_{snapshot}.csv"
    df.to_csv(filename, index=False)

    print(f"Fichier : {filename}")
    print(f"Nombre de lignes : {len(df)}")
    print(f"Première date : {df['Date'].min()}")
    print(f"Dernière date : {df['Date'].max()}")

print("\nTéléchargement terminé !")

