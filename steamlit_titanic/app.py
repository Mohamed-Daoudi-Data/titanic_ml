import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Charger le modèle
model = joblib.load('logreg_model.pkl')

# Configuration de la page
st.set_page_config(page_title='Prédiction Titanic', page_icon=':ship:', layout='wide')

# Titre de l'application
st.title('Application de Prédiction de Survie sur le Titanic :ship:')

# Description
st.markdown("""
Cette application prédit la survie d'un passager sur le Titanic en se basant sur des informations telles que la classe du passager, le sexe, l'âge et le nombre de frères, sœurs ou conjoints à bord.
""")

# Sidebar pour les inputs
st.sidebar.header('Entrez les détails du passager')

pclass = st.sidebar.selectbox('Classe du passager', [1, 2, 3], index=1)
sex = st.sidebar.selectbox('Sexe', ['Masculin', 'Féminin'])
age = st.sidebar.slider('Âge', 1, 100, 28)
sibsp = st.sidebar.slider('Nombre de frères, sœurs ou conjoints à bord', 0, 10, 1)

# Conversion du sexe en 0 ou 1
sex = 0 if sex == 'Masculin' else 1

# Section principale pour les résultats et bouton de prédiction
col1, col2 = st.columns([2, 1])

with col1:
    if st.button('Prédire'):
        # Création d'un DataFrame avec les entrées utilisateur pour correspondre à l'entraînement du modèle
        input_data = pd.DataFrame([[pclass, sex, age, sibsp]], 
                                  columns=['Pclass', 'Sex', 'Age', 'SibSp'])
        
        # Prédiction
        prediction = model.predict(input_data)
        
        # Affichage du résultat
        result = 'Survécu' if prediction[0] == 1 else 'Non Survécu'
        st.markdown(f"### Le passager aurait : **{result}**")

        # Affichage d'une image en fonction du résultat
        image = Image.open('survived.jpg' if prediction[0] == 1 else 'not_survived.jpg')
        st.image(image, caption='Résultat de la prédiction')

with col2:
    st.markdown('## Détails du passager')
    st.write('Classe du passager:', pclass)
    st.write('Sexe:', 'Masculin' if sex == 0 else 'Féminin')
    st.write('Âge:', age)
    st.write('Nombre de frères, sœurs ou conjoints à bord:', sibsp)

# Informations supplémentaires
st.sidebar.markdown("""
**À propos:**  
Cette application est construite avec Streamlit et utilise un modèle de régression logistique pour prédire la survie des passagers du Titanic.
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.info('Créé par Mohamed Daoudi')
