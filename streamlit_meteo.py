import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="🌧️ Analyse Météo Australie")

df = pd.read_csv("https://raw.githubusercontent.com/Prisca-hub/projet-m-t-o/487f43be988a97259d35889f958f6644e5a5fd6c/weatherAUS.csv")

st.sidebar.title("📌 Sommaire")
pages = [
    "📘 Contexte du projet",
    "🔍 Exploration des données",
    "📊 Visualisation des données",
    "🧼 Préparation des données",
    "🤖 Modélisation",
    "🚀 Déploiement"
]
page = st.sidebar.selectbox("Sélectionnez une page :", pages)

# Page 0 : Contexte
if page == pages[0]:
    st.title("📘 Contexte du projet")
    st.subheader("Prévision de Météo en Australie")
    st.write("Présenté par : Prisca Belair, Jimmy Seyer et Samuel Ogez")
    st.image("https://raw.githubusercontent.com/Prisca-hub/projet-m-t-o/487f43be988a97259d35889f958f6644e5a5fd6c/pluie.jpg")

    st.markdown("""
    Ce projet vise à développer un modèle de **machine learning** pour **prédire les précipitations de demain en Australie**.  
    Nous analyserons un jeu de données couvrant **plus de 10 ans d’observations météorologiques**, en combinant :  
    - Exploration des données 🔍  
    - Visualisation interactive 🧭  
    - Préparation des données 🧹  
    - Modélisation prédictive 🤖  
    """)

# Page 1 : Exploration
elif page == pages[1]:
    st.title("👁️ Exploration des données")
    st.subheader("🔎 Aperçu du dataset")
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("📐 Dimensions et types")
    col1, col2 = st.columns(2)
    col1.metric("Lignes", df.shape[0])
    col2.metric("Colonnes", df.shape[1])

    col1, col2 = st.columns(2)
    col1.write("**Catégorielles :**")
    col1.write(df.select_dtypes(include='object').columns.tolist())
    col2.write("**Numériques :**")
    col2.write(df.select_dtypes(include=np.number).columns.tolist())

    st.markdown("---")
    st.subheader("🚨 Analyse des valeurs manquantes")
    missing_percent = df.isnull().mean().sort_values(ascending=False) * 100
    missing_filtered = missing_percent[missing_percent > 5]
    if not missing_filtered.empty:
        st.write("### Variables avec plus de 5% de valeurs manquantes")
        st.dataframe(missing_filtered.rename("Pourcentage de NaN (%)"))
    else:
        st.success("✅ Aucune variable avec plus de 5% de valeurs manquantes.")

    with st.expander("📋 Voir toutes les colonnes avec des valeurs manquantes"):
        missing_total = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Valeurs manquantes': missing_total,
            'Pourcentage (%)': (missing_total / len(df)) * 100
        }).query("`Valeurs manquantes` > 0").sort_values("Pourcentage (%)", ascending=False)
        st.dataframe(missing_df)

# Page 2 : Visualisation
elif page == pages[2]:
    st.title("📊 Visualisation des données")

    st.subheader("🌧️ Déséquilibre des classes de la variable cible")
    if 'RainTomorrow' in df.columns:
        rain_dist = df['RainTomorrow'].value_counts(normalize=True) * 100
        st.write(rain_dist.rename("Pourcentage (%)"))
        fig_pie = px.pie(names=rain_dist.index, values=rain_dist.values, title="Répartition des classes RainTomorrow")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("La variable 'RainTomorrow' n'existe pas dans le dataset.")

    st.markdown("---")
    st.subheader("📍 Répartition géographique")
    nb_stations = df['Location'].nunique()
    st.write(f"- **Nombre de stations météorologiques :** {nb_stations}")
    top_locations = df['Location'].value_counts().head(10).rename_axis('Station').reset_index(name='Nombre d\'observations')
    fig_bar = px.bar(top_locations, x='Station', y='Nombre d\'observations', title="Top 10 des stations les plus fréquentes")
    st.plotly_chart(fig_bar)

    st.markdown("---")
    st.subheader("🗓️ Couverture temporelle")
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        st.write(f"- **Début :** {df['Date'].min().date()}")
        st.write(f"- **Fin :** {df['Date'].max().date()}")
        st.write(f"- **Durée :** {(df['Date'].max() - df['Date'].min()).days} jours")
        df['Year'] = df['Date'].dt.year
        st.bar_chart(df['Year'].value_counts().sort_index())
    except Exception as e:
        st.warning(f"Problème de format de date : {e}")

    st.markdown("---")
    st.subheader("🧊 Carte thermique des corrélations")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig_corr = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',  # Palette inverse avec une bonne différenciation entre les positives et négatives
        aspect='auto',
        labels=dict(color="Corrélation"),
        title="Corrélations entre les variables numériques"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.subheader("🌡️ Température minimale selon la pluie")
    fig_hist = px.histogram(df, x='MinTemp', color='RainTomorrow', barmode='overlay', nbins=40, opacity=0.6)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.subheader("🧭 Pression à 15h par ville")
    selected_locations = st.multiselect("🏙️ Sélectionnez les villes :", df['Location'].dropna().unique(), default=['Sydney', 'Perth', 'Melbourne'])
    df_filtered = df[df['Location'].isin(selected_locations)]
    fig_pressure = px.box(df_filtered, x='Location', y='Pressure3pm', color='RainTomorrow', title="Pression à 15h par ville")
    st.plotly_chart(fig_pressure, use_container_width=True)

# Page 3 : Préparation des données
elif page == pages[3]:
    st.title("🧼 Préparation des données")
    
    st.subheader("🧹 Nettoyage des données")
    
    st.markdown("""
    Avant d’entraîner notre modèle, il était nécessaire de nettoyer les données pour les rendre cohérentes et exploitables.
    Ce processus a inclus plusieurs étapes clés :
    - **Suppression des colonnes inutiles** : Nous avons supprimé les colonnes non pertinentes pour la prédiction de la pluie.
    - **Gestion des valeurs manquantes** : Les colonnes avec plus de 35% de valeurs manquantes ont été supprimées.
    - **Imputation des valeurs** : Les colonnes restantes ont été imputées avec des moyennes ou des médianes.
    - **Standardisation des données** : Les variables numériques ont été standardisées pour faciliter l'entraînement du modèle.
    """)
    
    for col in ['MaxTemp', 'MinTemp']:
        med = df.groupby('Location')[col].transform('median')
        df[col] = df[col].fillna(med)
    df['amplitude_thermique'] = df['MaxTemp'] - df['MinTemp']
    df.drop(['MinTemp', 'Temp9am', 'Temp3pm'], axis=1, inplace=True)

    df['Rainfall'] = df['Rainfall'].fillna(0)
    df.drop('RainToday', axis=1, inplace=True)

# Colonnes trop vides
    na_percent = df.isna().mean() * 100
    cols_to_drop = na_percent[na_percent > 38].index.tolist()
    df.drop(cols_to_drop, axis=1, inplace=True)

# Vent
    df['WindSpeed3pm'] = df['WindSpeed3pm'].fillna(df.groupby('Location')['WindSpeed3pm'].transform('median'))
    df.drop('WindSpeed9am', axis=1, inplace=True)

# Humidité
    df['Humidity3pm'] = df['Humidity3pm'].fillna(df.groupby('Location')['Humidity3pm'].transform('median'))
    df.drop('Humidity9am', axis=1, inplace=True)

# Pression
    df['Pressure3pm'] = df['Pressure3pm'].fillna(df['Pressure3pm'].median())

    # Suppression colonnes inutiles
    df.drop(['WindGustDir', 'WindDir9am', 'WindDir3pm', 'WindGustSpeed', 'Pressure9am'], axis=1, inplace=True)

    # Cible
    df.dropna(subset=['RainTomorrow'], inplace=True)
    df['RainTomorrow'] = (df['RainTomorrow'] == 'Yes').astype(int)

    # Features d'interaction
    df['humidity_pressure'] = df['Humidity3pm'] * df['Pressure3pm']
    df['humidity_temp'] = df['Humidity3pm'] * df['MaxTemp']
    df['pressure_rainfall'] = df['Pressure3pm'] * df['Rainfall']

    # Standardisation
    exclude_cols = ['RainTomorrow', 'month', 'season']
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_cols)
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    st.markdown("---")
    st.subheader("⬇️Récupération de données externes")
    st.markdown("""
    Certaines données de notre dataset, telles que **Cloud9am**, **Cloud3pm**, **Sunshine** et **Evaporation**, ont été supprimées en raison d'un trop grand nombre de valeurs manquantes. Afin de compléter notre jeu de données, nous allons récupérer ces informations à partir du site web météorologique **ERA5** (European Centre for Medium-Range Weather Forecasts), qui fournit des données climatiques précises et fiables. Ces nouvelles données permettront d'améliorer la qualité de notre modèle de prédiction.

    """)
    st.image("https://raw.githubusercontent.com/Prisca-hub/projet-m-t-o/487f43be988a97259d35889f958f6644e5a5fd6c/ERA5.png")

    st.markdown("---")
    
    st.subheader("🔧 Feature Engineering")
    st.markdown("""
     Pour améliorer la prédiction de la pluie de demain, nous avons ajouté plusieurs nouvelles *features* :

    - **Interactions clés** : 
      - `pressure_rainfall`, `humidity_temp`, `humidity_pressure` capturent les relations complexes entre température, humidité et pression, des facteurs importants pour la pluie.

    - **Historique des précipitations** : 
      - `rolling_rain_3d`, `7d`, `14d` : Moyennes mobiles pour détecter les tendances récentes de la pluie.

    - **Saisonnalité** : 
      - `month_sin`, `month_cos`, et `season` : Capturent les effets saisonniers sur la pluie.

    - **Relief géographique** : 
      - `relief_type` : Prend en compte l’impact du terrain sur les conditions météorologiques locales.

    - **Amplitudes thermiques** : 
      - `amplitude_thermique` : Indicateur de la variation de température qui peut influencer la pluie.

    Ces nouvelles variables enrichiront notre modèle pour mieux prédire la pluie de demain.
    """)
    
    geographic_features = {
    'Sydney': 'coastal', 'Melbourne': 'coastal', 'Brisbane': 'coastal',
    'Perth': 'coastal', 'Adelaide': 'coastal', 'Darwin': 'coastal',
    'Hobart': 'coastal', 'Albany': 'coastal', 'Cairns': 'coastal',
    'CoffsHarbour': 'coastal', 'GoldCoast': 'coastal', 'Newcastle': 'coastal',
    'NorahHead': 'coastal', 'NorfolkIsland': 'coastal', 'Portland': 'coastal',
    'Townsville': 'coastal', 'Wollongong': 'coastal', 'Williamtown': 'coastal',
    'Sale': 'coastal', 'Walpole': 'coastal', 'Witchcliffe': 'coastal',
    'PerthAirport': 'coastal', 'SydneyAirport': 'coastal',
    'Canberra': 'inland_plateau', 'Albury': 'inland_plateau', 'Ballarat': 'inland_plateau',
    'Bendigo': 'inland_plateau', 'Tuggeranong': 'inland_plateau', 'WaggaWagga': 'inland_plateau',
    'Katherine': 'inland_plateau', 'Launceston': 'inland_plateau', 'Moree': 'inland_plateau',
    'Nhil': 'inland_plateau', 'Nuriootpa': 'inland_plateau', 'Penrith': 'inland_plateau',
    'Richmond': 'inland_plateau', 'SalmonGums': 'inland_plateau', 'Watsonia': 'inland_plateau',
    'BadgerysCreek': 'inland_plateau', 'MelbourneAirport': 'inland_plateau', 'Dartmoor': 'inland_plateau',
    'PearceRAAF': 'inland_plateau',
    'Alice Springs': 'desert', 'Uluru': 'desert', 'Cobar': 'desert',
    'Mildura': 'desert', 'Woomera': 'desert', 'AliceSprings': 'desert',
    'MountGinini': 'mountain', 'MountGambier': 'mountain'
}
    df['relief_type'] = df['Location'].map(geographic_features).fillna('unknown')

    st.markdown("---")   

    df = df.set_index(['Date', 'Location'])
    df.index = df.index.set_levels([pd.to_datetime(df.index.levels[0]), df.index.levels[1]])
    df['month'] = df.index.get_level_values('Date').month
    df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], labels=['Été', 'Automne', 'Hiver', 'Printemps'])
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    for window in [3, 7, 14]:
        df[f'rolling_rain_{window}d'] = df.groupby(level='Location')['Rainfall'].transform(
            lambda x: x.rolling(window=window, min_periods=1, closed='left').mean()
        )
    
    # Aperçu du dataset nettoyé 
    st.subheader("✅ Données nettoyées")
    st.dataframe(df.head())
    st.download_button("📥 Télécharger les données nettoyées", 
                       data=df.reset_index().to_csv(index=False).encode('utf-8'), 
                       file_name="weather_cleaned.csv", mime='text/csv')
