
# D02 — Dictionnaire des données

## Projet
B08 — EEM : Marchés émergents

## Source
Yahoo Finance, via la bibliothèque Python yfinance.

## Instruments
- EEM : iShares MSCI Emerging Markets ETF
- SPY : SPDR S&P 500 ETF Trust (benchmark)

## Période
Du 02/01/2018 au 30/10/2026.

## Fréquence
Journalière, séances boursières américaines.

## Variables

| Variable | Description | Unité |
|---|---|---|
| Date | Date de la séance | YYYY-MM-DD |
| Open | Prix d'ouverture | USD |
| High | Prix le plus haut | USD |
| Low | Prix le plus bas | USD |
| Close | Prix de clôture | USD |
| Adj Close | Prix de clôture ajusté | USD |
| Volume | Volume échangé | Nombre de parts |

## Variable calculée

Rendement logarithmique journalier :

r(t) = ln(AdjClose(t) / AdjClose(t-1))

Le rendement est multiplié par 100 pour être
exprimé en pourcentage.

## Convention de collecte
- auto_adjust=False
- Données OHLC non automatiquement ajustées
- Adj Close utilisé pour calculer les rendements
- Date de fin exclusive : 2026-10-31

## Reproductibilité
Les fichiers CSV sont conservés dans data/raw/
avec un horodatage UTC dans leur nom.

Les snapshots ne doivent pas être modifiés.
Toute transformation doit produire un fichier distinct.

