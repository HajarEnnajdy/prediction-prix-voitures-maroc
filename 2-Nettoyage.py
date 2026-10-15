import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import joblib 

#hajar En-najdy
#mohamed akrame kaichy

# Charger les données
df = pd.read_csv('dataset.csv')

# -------------------------------------------------------------------
# 1. VISUALISATION DES VALEURS MANQUANTES
# -------------------------------------------------------------------
print("Valeurs manquantes initiales :")
print(df.isnull().sum())

msno.matrix(df)
plt.title('Visualisation des valeurs manquantes')
plt.show()

msno.bar(df)
plt.title('Nombre de valeurs manquantes par colonne')
plt.show()

# -------------------------------------------------------------------
# 2. NETTOYAGE INITIAL
# -------------------------------------------------------------------
df.drop(columns=['État', 'Nombre de portes'], inplace=True)

df = df.dropna(subset=['Boite de vitesses','Année-Modèle','Kilométrage','Marque'
                       ,'Modèle','Puissance fiscale','Type de carburant'])
print(df.isnull().sum())

# -------------------------------------------------------------------
# 3. NETTOYAGE DES COLONNES SPÉCIFIQUES
# -------------------------------------------------------------------

# Encodage binaire
df['Première main'] = df['Première main'].map({'Oui': 1, 'Non': 0})

# Nettoyage de la puissance fiscale
df['Puissance fiscale'] = df['Puissance fiscale'].str.replace('CV', '', regex=False).str.strip()
def convertir_puissance(val):
    try:
        return float(val)
    except:
        if isinstance(val, str) and 'plus' in val.lower():
            return 42.0
        return np.nan

df['Puissance fiscale'] = df['Puissance fiscale'].apply(convertir_puissance)
df['Puissance fiscale'] = pd.to_numeric(df['Puissance fiscale'], errors='coerce')

# Nettoyage kilométrage
def moyenne_intervalle(km):
    if pd.isnull(km):
        return np.nan
    try:
        km = km.replace(' ', '')
        if '-' in km:
            val1, val2 = map(int, km.split('-'))
            return (val1 + val2) / 2
        return float(km)
    except:
        return np.nan

df['Kilométrage'] = df['Kilométrage'].apply(moyenne_intervalle)

# Nettoyage prix
df['Prix'] = df['Prix'].replace({' ': '', ' ': '', ',': ''}, regex=True).apply(pd.to_numeric, errors='coerce')

# Conversion numérique
colonnes_numeriques = ['Kilométrage', 'Prix', 'Puissance fiscale', 'Année-Modèle']
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# -------------------------------------------------------------------
# DETECTION ET SUPPRESSION DES OUTILERS
# -------------------------------------------------------------------
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

colonnes_outliers = ['Prix', 'Kilométrage', 'Puissance fiscale' , 'Année-Modèle']
for col in colonnes_outliers:
    outliers = detect_outliers_iqr(df, col)
    print(f"Outliers détectés dans {col} : {outliers.shape[0]}")
    df = df[~df.index.isin(outliers.index)]

# -------------------------------------------------------------------
# IMPUTATION DES VALEURS MANQUANTES
# -------------------------------------------------------------------

def tree_imputation(data, target_col):
    known = data[data[target_col].notnull()]
    unknown = data[data[target_col].isnull()]
    if unknown.empty:
        return data

    X_train = known.drop(columns=[target_col])
    y_train = known[target_col]
    X_pred = unknown.drop(columns=[target_col])

    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], remainder='passthrough')

    model = DecisionTreeRegressor() if data[target_col].dtype in [np.float64, np.int64] else DecisionTreeClassifier()
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    pipeline.fit(X_train, y_train)
    data.loc[data[target_col].isnull(), target_col] = pipeline.predict(X_pred)
    return data

# Imputation
colonnes_a_imputer = df.columns[df.isnull().any()].tolist()
for col in colonnes_a_imputer:
    df = tree_imputation(df, col)

print(df.isnull().sum())

# -------------------------------------------------------------------
# ENCODAGE DES VARIABLES CATÉGORIELLES 
# -------------------------------------------------------------------

cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------------------------------------------------------------------
# 7. STANDARDISATION DES VARIABLES
# -------------------------------------------------------------------
scaler = StandardScaler()
cols_to_scale = ['Kilométrage', 'Puissance fiscale', 'Année-Modèle']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# -------------------------------------------------------------------
# SUPPRESSION DES DOUBLONS
# -------------------------------------------------------------------

df.drop_duplicates(inplace=True)

# -------------------------------------------------------------------
# 9. ANALYSE EXPLORATOIRE
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))
mean_prices = df.groupby('Marque')['Prix'].mean().sort_values(ascending=False)
sns.barplot(x=mean_prices.index[:10], y=mean_prices.values[:10])
plt.xticks(rotation=45)
plt.title('Top 10 Marques avec le prix moyen le plus élevé')
plt.ylabel('Prix moyen')
plt.show()

sns.scatterplot(data=df, x='Puissance fiscale', y='Prix')
plt.title('Prix vs Puissance fiscale')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Corrélation entre les variables numériques")
plt.show()

sns.boxplot(x='Boite de vitesses', y='Prix', data=df)
plt.title('Prix par type de boîte de vitesses')
plt.show()

df.hist(figsize=(15, 10), bins=30)
plt.suptitle('Distribution des variables numériques')
plt.tight_layout()
plt.show()

sns.pairplot(df[['Prix', 'Puissance fiscale', 'Kilométrage', 'Année-Modèle']])
plt.suptitle("Relations entre variables numériques", y=1.02)
plt.show()

sns.lmplot(x='Kilométrage', y='Prix', data=df, aspect=1.5)
plt.title("Prix vs Kilométrage avec droite de régression")
plt.show()

sns.histplot(df['Prix'], kde=True)
plt.title('Distribution des prix')
plt.show()

sns.countplot(x='Type de carburant', data=df)
plt.title('Répartition par type de carburant')
plt.xticks(rotation=45)
plt.show()

sns.scatterplot(x='Année-Modèle', y='Prix', data=df)
plt.title('Prix en fonction de l\'année-modèle')
plt.show()

df.to_excel("dataset_cleaned.xlsx", index=False)
joblib.dump(label_encoders, 'label_encoders.pkl')