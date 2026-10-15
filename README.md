# Prédiction des Prix des Voitures d'Occasion sur le Marché Marocain

> Système intelligent de Machine Learning pour l'estimation automatisée des prix de véhicules d'occasion à partir de données réelles collectées sur le marché marocain.

---

## Présentation du Projet
Dans un marché automobile marocain en constante évolution, fixer ou évaluer le juste prix d'un véhicule d'occasion représente un défi majeur tant pour les acheteurs que pour les vendeurs. 

Ce projet propose une **chaîne complète de Data Science (Data Pipeline)** allant du web scraping automatisé jusqu'au déploiement d'une application web interactive, en passant par un prétraitement rigoureux des données et l'optimisation de modèles de régression.

---

## Architecture & Pipeline du Projet

1. **Web Scraping & Collecte :** Extraction automatisée via `Selenium` et `BeautifulSoup` (avec rotation des User-Agents, délais aléatoires et mode *headless* pour contourner les protections anti-bot).
2. **Nettoyage & Prétraitement Avancé :** 
   * Suppression des doublons et des variables bruitées (*État*, *Nombre de portes*).
   * Conversion et normalisation des formats textuels (*kilométrages sous forme d'intervalles, puissances fiscales en CV*).
   * **Imputation intelligente** des valeurs manquantes à l'aide d'un `DecisionTreeRegressor` (plus performant qu'une simple moyenne/médiane).
   * Détection et traitement des *outliers* via la méthode **IQR**.
3. **Analyse Exploratoire (EDA) :** Étude des corrélations et génération d'un rapport complet via `YData Profiling`.
4. **Modélisation & Évaluation :** Comparaison de 3 algorithmes de régression validés par validation croisée (5-folds).
5. **Déploiement :** Interface utilisateur interactive développée avec **Streamlit**.

---

## Comparaison des Performances des Modèles

Les modèles ont été évalués sur un jeu de test indépendant selon quatre métriques clés : **RMSE**, **MAE**, **MAPE** et **$R^2$**[cite: 33].

Régression Linéaire : Test RMSE = 45 775.53 MAD, Test MAE = 33 483.29 MAD, Test MAPE = 79.46 %, Score \(R^2\) = 0.71, CV RMSE = 45 769.23 ± 962.
Random Forest : Test RMSE = 27 512.70 MAD, Test MAE = 17 079.21 MAD, Test MAPE = 62.50 %, Score \(R^2\) = 0.89, CV RMSE = 28 831.76 ± 1 270.
XGBoost (modèle sélectionné ) : Test RMSE = 27 224.36 MAD, Test MAE = 17 896.01 MAD, Test MAPE = 38.00 %, Score \(R^2\) = 0.90, CV RMSE = 27 999.68 ± 1 333.

> **Conclusion de la modélisation :** Le modèle **XGBoost** a été retenu et sérialisé (`.pkl`) car il offre le meilleur compromis entre précision de prédiction ($R^2 = 0.90$) et robustesse face aux non-linéarités du marché.

---

## Application Web (Streamlit)

L'application interactive permet de renseigner les caractéristiques d'un véhicule (année, kilométrage, marque, modèle, boîte de vitesses, etc.) pour obtenir instantanément une estimation fiable du prix en Dirhams (DH).

* *Encodage dynamique des variables et gestion intelligente des erreurs de saisie.*

---

## Installation & Utilisation

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/votre-nom-utilisateur/prediction-prix-voitures-maroc.git](https://github.com/votre-nom-utilisateur/prediction-prix-voitures-maroc.git)
   cd prediction-prix-voitures-maroc
