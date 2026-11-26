
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures_D04"
FIGURES.mkdir(parents=True, exist_ok=True)

# Données de validation et de test
validation = pd.read_csv(
    REPORTS / "D04_previsions_VaR_ES.csv",
    parse_dates=["Date"]
)

test = pd.read_csv(
    REPORTS / "D04_previsions_test_final.csv",
    parse_dates=["Date"]
)

# 1. Comparaison des volatilités sur validation
plt.figure(figsize=(12, 5))

plt.plot(
    validation["Date"],
    validation["volatilite_GARCH"],
    label="GARCH(1,1)"
)

plt.plot(
    validation["Date"],
    validation["volatilite_GJR-GARCH"],
    label="GJR-GARCH",
    alpha=0.8
)

plt.title("EEM - Prévisions de volatilité (2023-2024)")
plt.xlabel("Date")
plt.ylabel("Volatilité journalière (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES / "01_comparaison_volatilite.png",
    dpi=200
)
plt.close()

# 2. Volatilité prévue sur test
plt.figure(figsize=(12, 5))

plt.plot(
    test["Date"],
    test["volatilite_prevue_pct"],
    label="GARCH(1,1)"
)

plt.title("EEM - Volatilité prévue (2025-2026)")
plt.xlabel("Date")
plt.ylabel("Volatilité journalière (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES / "02_volatilite_test.png",
    dpi=200
)
plt.close()

# 3. Graphiques VaR 95 % et 99 %
for niveau in [95, 99]:
    plt.figure(figsize=(12, 5))

    plt.plot(
        test["Date"],
        test["rendement"],
        label="Rendement observé",
        alpha=0.6
    )

    plt.plot(
        test["Date"],
        -test[f"VaR_{niveau}_pct"],
        label=f"Seuil VaR {niveau} %",
        linestyle="--"
    )

    exceptions = test[
        test[f"exception_{niveau}"].astype(bool)
    ]

    plt.scatter(
        exceptions["Date"],
        exceptions["rendement"],
        label="Dépassements",
        zorder=5
    )

    plt.title(
        f"EEM - Backtesting VaR {niveau} % (2025-2026)"
    )
    plt.xlabel("Date")
    plt.ylabel("Rendement journalier (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES / f"0{4 if niveau == 95 else 5}_VaR_{niveau}.png",
        dpi=200
    )
    plt.close()

print("\n=== GRAPHIQUES D04 GENERES ===")
for fichier in sorted(FIGURES.glob("*.png")):
    print(fichier.name)
