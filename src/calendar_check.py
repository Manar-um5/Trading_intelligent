
from pathlib import Path
import pandas as pd

def load_dates(ticker):
    files = sorted(Path("data/raw").glob(f"{ticker}_*.csv"))
    df = pd.read_csv(files[-1], parse_dates=["Date"])
    return df["Date"]

eem_dates = load_dates("EEM")
spy_dates = load_dates("SPY")

print("CONTROLE DU CALENDRIER")
print("------------------------")
print("EEM - dates uniques :", eem_dates.nunique())
print("SPY - dates uniques :", spy_dates.nunique())

eem_set = set(eem_dates)
spy_set = set(spy_dates)

print("Dates communes :", len(eem_set & spy_set))
p(eem_dates.dt.day_of_week >= 5).sum())rint("Dates uniquement EEM :", len(eem_set - spy_set))(eem_dates.dt.day_of_week >= 5).sum())
 print("Dates uniquement SPY :", len(spy_set - eem_set))

print("Dates EEM hors lundi-vendredi :",
      (eem_dates.dt.dayofweek >= 5).sum())

print("Dates SPY hors lundi-vendredi :",
      (spy_dates.dt.dayofweek >= 5).sum())

