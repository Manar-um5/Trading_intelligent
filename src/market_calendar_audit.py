
from pathlib import Path
import pandas as pd
import pandas_market_calendars as mcal

# Calendrier des séances NYSE
nyse = mcal.get_calendar("NYSE")

expected = nyse.valid_days(
    start_date="2018-01-01",
    end_date="2026-10-30"
)

expected_dates = set(expected.date)

for ticker in ["EEM", "SPY"]:
    files = sorted(Path("data/raw").glob(f"{ticker}_*.csv"))

    if not files:
        print(f"Aucun fichier pour {ticker}")
        continue

    df = pd.read_csv(files[-1], parse_dates=["Date"])
    actual_dates = set(df["Date"].dt.date)

    missing = sorted(expected_dates - actual_dates)
    unexpected = sorted(actual_dates - expected_dates)

    print(f"\nAUDIT CALENDRIER : {ticker}")
    print("Séances NYSE attendues :", len(expected_dates))
    print("Séances observées :", len(actual_dates))
    print("Séances manquantes :", len(missing))
    print("Dates inattendues :", len(unexpected))

    if missing:
        print("Dates manquantes :", missing)

    if unexpected:
        print("Dates inattendues :", unexpected)

