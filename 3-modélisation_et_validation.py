# Importations des bibliothèques 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib 

#hajar En-najdy
#mohamed akrame kaichy

# =============================================
# Modélisation - Choix des Algorithmes
# =============================================

# Chargement des données
df = pd.read_excel('dataset_cleaned.xlsx')

# --------------------------------------------
# Définition des features (X) et de la target (y)
X = df.drop(columns=['Prix'])  # Features
y = df['Prix']                 # Target (Prix)

# -------------------------------------------------
# Séparation train/test pour évaluation indépendante
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Comparer modèles simples et avancés
models = {
    "Linear Regression": LinearRegression(),  
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42), 
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42) 
}

# Évaluer la performance des modèles
print("\n=== Évaluation sur le jeu de test ===")

model_names = []
rmse_scores = []
mae_scores = []
mape_scores = []
r2_scores = []
cv_rmse_means = []
cv_rmse_stds = []

for name, model in models.items():
    model_names.append(name)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if name == "XGBoost":
        joblib.dump(model, 'xgboost_model.pkl')
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)
    r2_scores.append(r2)
    
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse_means.append(-scores.mean())
    cv_rmse_stds.append(scores.std())
    
    print(f"\n=== {name} ===")
    print("Test RMSE :", round(rmse, 2))
    print("Test MAE  :", round(mae, 2))
    print("Test MAPE :", round(mape, 2), "%")
    print("Test R²   :", round(r2, 2))
    print("CV RMSE   :", round(-scores.mean(), 2), "±", round(scores.std(), 2))

# =============================================
# Validation croisée
# =============================================

print("\n=== Validation croisée (RMSE moyennes sur 5 folds) ===")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    print(f"{name} : {-scores.mean():.2f} ± {scores.std():.2f}")

# =============================================
# Visualisation des résultats
# =============================================

plt.figure(figsize=(20, 5))

# 1. RMSE sur le test set
plt.subplot(1, 5, 1)
plt.bar(model_names, rmse_scores, color='skyblue')
plt.title('RMSE (Test Set)\nPlus bas = meilleur')
plt.ylabel('RMSE')
plt.xticks(rotation=15)

# 2. MAE sur le test set
plt.subplot(1, 5, 2)
plt.bar(model_names, mae_scores, color='orange')
plt.title('MAE (Test Set)\nPlus bas = meilleur')
plt.ylabel('MAE')
plt.xticks(rotation=15)

# 3. MAPE sur le test set
plt.subplot(1, 5, 3)
plt.bar(model_names, mape_scores, color='violet')
plt.title('MAPE (Test Set)\nPlus bas = meilleur')
plt.ylabel('MAPE (%)')
plt.xticks(rotation=15)

# 4. R² Score sur le test set
plt.subplot(1, 5, 4)
plt.bar(model_names, r2_scores, color='lightgreen')
plt.title('R² (Test Set)\nProche de 1 = meilleur')
plt.ylabel('R² Score')
plt.xticks(rotation=15)

# 5. RMSE en validation croisée
plt.subplot(1, 5, 5)
plt.bar(model_names, cv_rmse_means, yerr=cv_rmse_stds, capsize=5, color='salmon')
plt.title('RMSE Validation Croisée\n(5 folds)')
plt.ylabel('RMSE moyen ± écart-type')
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()
