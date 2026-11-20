
# QUANT LAB — TRADING INTELLIGENT
## Livrable D02 : Données et audit

**Variante :** B08 — EEM, Marchés émergents  
**Enseignant :** Tawfik Masrour  
**Université :** Mohammed V, Rabat  
**Date de collecte :** 14 novembre 2026

---

## 1. Introduction

Ce livrable présente la collecte, la description et
l'audit des données financières utilisées dans le
projet Quant Lab, variante B08.

L'actif principal est EEM, un ETF exposé aux marchés
émergents. L'ETF SPY est utilisé comme instrument
de comparaison.

La question scientifique de B08 est :

"L'asymétrie des chocs affecte-t-elle la fiabilité
des mesures de risque et les décisions sur
marchés émergents ?"

L'objectif de D02 est de constituer une base de
données fiable, traçable et reproductible avant
la construction des modèles prédictifs.

## 2. Collecte des données

Source : Yahoo Finance, via yfinance.

Instruments :
- EEM : iShares MSCI Emerging Markets ETF
- SPY : SPDR S&P 500 ETF Trust

Période : 02/01/2018 au 30/10/2026.
Fréquence : journalière.
Nombre d'observations : 2 220 par ETF.

Les données comprennent :
Date, Open, High, Low, Close, Adj Close et Volume.

Les données ont été enregistrées dans des fichiers
CSV horodatés afin de conserver un snapshot local.

Convention de téléchargement :
auto_adjust=False.

Les prix ajustés (Adj Close) sont utilisés pour
calculer les rendements logarithmiques.

Des avertissements de timeout ont été observés
pendant la collecte, mais les deux fichiers ont
été générés et leur contenu a été vérifié.

## 3. Audit de qualité

| Contrôle | EEM | SPY |
|---|---:|---:|
| Observations | 2220 | 2220 |
| Dates dupliquées | 0 | 0 |
| Valeurs manquantes | 0 | 0 |
| Prix invalides | 0 | 0 |
| Incohérences OHLC | 0 | 0 |
| Volumes négatifs | 0 | 0 |
| Rendements extrêmes (>10%) | 2 | 2 |

Les premiers contrôles ne révèlent aucune
anomalie structurelle dans les fichiers.

## 4. Analyse des chocs extrêmes

Le rendement logarithmique est calculé par :

r(t) = ln(P(t) / P(t-1))

avec P(t) = prix de clôture ajusté.

Les rendements sont multipliés par 100
pour être exprimés en pourcentage.

| Date | EEM | SPY |
|---|---:|---:|
| 12/03/2020 | -10,55 % | -10,06 % |
| 16/03/2020 | -13,33 % | -11,59 % |

Ces observations correspondent à la période
de fortes turbulences financières associées
à la pandémie de COVID-19.

Les pertes observées sur EEM sont plus
importantes que celles de SPY lors de
ces deux séances.

Ces observations ne démontrent pas à elles
seules une asymétrie de volatilité.

Elles seront conservées pour les analyses
économétriques ultérieures.

## 5. Cohérence temporelle

Les deux instruments possèdent :

- 2 220 dates uniques.
- 2 220 dates communes.
- Aucune date exclusive à un instrument.
- Aucune observation pendant le week-end.

Les séries sont donc synchronisées sur
leurs dates observées.

### Vérification avec le calendrier officiel NYSE

Une vérification complémentaire a été réalisée
avec la bibliothèque pandas-market-calendars.

Sur la période du 01/01/2018 au 30/10/2026,
le calendrier NYSE indique 2 220 séances.

Pour EEM et SPY :
- Séances observées : 2 220
- Séances manquantes : 0
- Dates inattendues : 0

Les deux séries couvrent donc intégralement
les séances prévues par le calendrier NYSE
utilisé pour cette période.

## 6. Visualisations

Trois graphiques ont été produits :

1. Evolution des prix ajustés EEM et SPY.
2. Rendements logarithmiques journaliers EEM.
3. Comparaison des performances en base 100.

Fichiers :

- 01_prix_ajustes.png
- 02_rendements_eem.png
- 03_comparaison_base100.png

Ces visualisations permettent d'examiner
l'évolution des séries et les variations
extrêmes des marchés.


## 7. Limites

Le contrôle du calendrier NYSE a confirmé
que les 2 220 séances attendues sont présentes
pour EEM et SPY, sans date manquante ni
date inattendue.

Cependant, certaines limites subsistent :

- Les rendements extrêmes observés en mars 2020
  doivent être analysés avant toute décision
  de correction ou de suppression.

- Les prix ajustés (Adj Close) proviennent
  de Yahoo Finance. Leur convention d'ajustement
  et les éventuelles opérations sur titres
  doivent être documentées précisément.

- La qualité des données a été vérifiée par
  des contrôles automatiques, mais elle ne
  garantit pas l'absence de toute erreur
  provenant de la source financière.

Ces limites seront prises en compte dans
les prochaines étapes de modélisation
et d'analyse du risque.

### Vérification des prix ajustés

Une vérification complémentaire a été réalisée
afin de comparer les prix Close et Adj Close.

Pour EEM :
- Observations : 2 220
- Dates présentant une différence : 2 123
- Ratio Adj Close / Close : de 0,828505 à 1,0

Pour SPY :
- Observations : 2 220
- Dates présentant une différence : 2 189
- Ratio Adj Close / Close : de 0,875729 à 1,0

Ces résultats confirment que les séries de prix
ajustés diffèrent des prix de clôture bruts.

Pour le calcul des rendements logarithmiques,
le projet utilise la colonne Adj Close fournie
par Yahoo Finance, avec auto_adjust=False
lors du téléchargement.

Ce contrôle confirme la présence d'ajustements
historiques, mais ne valide pas individuellement
leur calcul ni leur origine.

La documentation détaillée des opérations sur
titres et des dividendes reste une limite
de l'audit.

## 8. Conclusion

Le livrable D02 a permis de collecter,
d'enregistrer et d'auditer les données
financières historiques des ETF EEM et SPY
sur la période 2018-2026.

Chaque instrument comporte 2 220 observations
journalières.

Les contrôles de qualité ont confirmé :
- Aucune valeur manquante.
- Aucune date dupliquée.
- Aucun prix invalide.
- Aucune incohérence OHLC.
- Aucun volume négatif.
- Aucune séance manquante selon le calendrier NYSE.

L'analyse des rendements logarithmiques a
également permis d'identifier deux séances
de fortes baisses, les 12 et 16 mars 2020,
pour les deux instruments.

Trois visualisations ont été produites pour
examiner l'évolution des prix, les rendements
journaliers et les performances comparées.

Ces résultats constituent une base de données
cohérente et documentée pour la suite du
projet B08.

Les prochaines étapes porteront sur l'étude
de la volatilité et de l'asymétrie des chocs
sur les marchés émergents, après vérification
des conventions d'ajustement des prix.

