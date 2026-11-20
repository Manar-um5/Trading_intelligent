import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

train = pd.read_csv("data/processed/EEM_train.csv")
y = train["rendement"].astype(float)

result = ARIMA(y, order=(1, 0, 1)).fit()
residus = result.resid

print("\n===== TEST DE LJUNG-BOX =====")
print(acorr_ljungbox(residus, lags=[5, 10, 20], return_df=True))

print("\n===== TEST ARCH-LM =====")
stat, pvalue, _, _ = het_arch(residus, nlags=10)
print(f"Statistique ARCH-LM : {stat:.4f}")
print(f"p-value : {pvalue:.6f}")

diagnostics = acorr_ljungbox(
    residus, lags=[5, 10, 20], return_df=True
)
diagnostics.to_csv("reports/D03_diagnostics_ARIMA_EEM.csv")

print("\nDiagnostics enregistrés.")
