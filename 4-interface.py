import streamlit as st
import pandas as pd
import joblib
from PIL import Image

#hajar En-najdy
#mohamed akrame kaichy

# Chargement du modèle et des encodeurs
model = joblib.load('xgboost_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Logo en haut
logo = Image.open("logo.jpeg")  # Remplace par ton logo
st.image(logo, width=120)

# Titre principal
st.markdown("<h1 style='text-align: center; color: #2E86C1;'> Estimation du Prix des Voitures d’Occasion - Maroc 🇲🇦</h1>", unsafe_allow_html=True)
st.markdown("---")

# Interface utilisateur
st.markdown("###  Paramètres de la voiture")

col1, col2 = st.columns(2)

with col1:
    annee = st.number_input(" Année-Modèle", min_value=1990, max_value=2025, value=2015)
    boite = st.selectbox(" Boîte de vitesses", ["Manuelle", "Automatique"])
    kilometrage = st.number_input(" Kilométrage (en KM)", min_value=0, max_value=500000, value=100000)
    marque = st.selectbox(" Marque", sorted([
        "Ford", "Volkswagen", "BMW", "Opel", "Dacia", "Seat", "Hyundai", "Land Rover", 
        "Peugeot", "Toyota", "Renault", "Suzuki", "Ssangyong", "Lifan", "Citroen", "DFSK", "Tesla",
        "Kia", "Volvo", "Alfa Romeo", "Mercedes-Benz", "Audi", "Mini", "Fiat", "Honda", "Jeep",
        "Jaguar", "Skoda", "Nissan", "Porsche", "Maserati", "DS", "Rover", "Chevrolet", "Geely",
        "Foton", "Isuzu", "Mazda", "MG", "Mahindra", "Chery", "Cupra", "Mitsubishi", "Abarth",
        "nan", "Lexus", "Lancia", "GMC", "Daihatsu", "Daewoo", "Cadillac", "Iveco", "Bentley",
        "Lincoln", "Autres", "Zotye", "Smart", "Changan", "Infiniti", "Subaru", "Chrysler", "Tata",
        "Dodge", "BYD", "Faw", "GWM Motors", "SERES", "Changhe", "Ferrari"
    ]))

with col2:
    modele = st.text_input(" Modèle", placeholder="Exemple : Clio 4, Golf 7, etc.")
    origine = st.selectbox(" Origine", ["WW au Maroc", "Importée neuve", "Dédouanée", "Pas encore dédouanée"])
    premiere_main = st.selectbox(" Première main", ["Oui", "Non"])
    puissance = st.number_input(" Puissance fiscale", min_value=1, max_value=20, value=6)
    carburant = st.selectbox(" Type de carburant", ["Essence", "Diesel", "Hybride", "Électrique"])

# Préparation des données
input_df = pd.DataFrame({
    'Année-Modèle': [annee],
    'Boite de vitesses': [boite],
    'Kilométrage': [kilometrage],
    'Marque': [marque],
    'Modèle': [modele],
    'Origine': [origine],
    'Première main': [premiere_main],
    'Puissance fiscale': [puissance],
    'Type de carburant': [carburant],
})

# Encodage
try:
    input_df['Boite de vitesses'] = label_encoders['Boite de vitesses'].transform(input_df['Boite de vitesses'])
    input_df['Marque'] = label_encoders['Marque'].transform(input_df['Marque'])
    input_df['Origine'] = label_encoders['Origine'].transform(input_df['Origine'])
    input_df['Type de carburant'] = label_encoders['Type de carburant'].transform(input_df['Type de carburant'])
    input_df['Première main'] = input_df['Première main'].map({'Oui': 1, 'Non': 0}).astype(int)

    if st.button(" Estimer le prix"):
        try:
            input_df['Modèle'] = label_encoders['Modèle'].transform(input_df['Modèle'])
            prediction = model.predict(input_df)[0]
            st.success(f" Prix estimé : {int(prediction):,} DH")
        except ValueError:
            st.error(" Le modèle saisi est inconnu. Essayez une orthographe différente.")

except Exception as e:
    st.error(" Une erreur est survenue lors du traitement des données.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:13px;'>Projet réalisé par Hajar En-Najdy et mohamed akrame kraichy | Licence d’Excellence en Intelligence Artificielle - FS Ben M'Sik</p>", unsafe_allow_html=True)
