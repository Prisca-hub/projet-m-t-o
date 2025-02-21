#Chargement des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Chargement du dataset
df = pd.read_csv("C:\\Users\\Prisc\\Desktop\\weatherAUS.csv")

#Affichage des 5 premières lignes
print(df.head())

#Affichage des informations sur le dataset
print(df.info())

#Affichage des statistiques descriptives
print(df.describe())

#Affichage des colonnes catégorielles
print(df.select_dtypes(include=['object']).columns)

#Affichage des valeurs manquantes
print(df.isnull().sum())

#Carte thermique de corrélations
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Carte thermique des corrélations')
plt.show()

#Relation entre humidité à 9h et à 15h
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Humidity9am', y='Humidity3pm', hue='RainTomorrow', data=df)
plt.title('Relation entre Humidité à 9h, Humidité à 15h et Pluie de demain')
plt.xlabel('Humidité 9h')
plt.ylabel('Humidité 15h')
plt.show()

#Relation entre humidité à 15h et pluie de demain
df['RainTomorrow'] = df['RainTomorrow'].replace({'No': 0, 'Yes': 1})
corr1 = df[['Humidity3pm', 'RainTomorrow']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr1, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Corrélation entre Humidité à 15h et Pluie de demain')
plt.show()

#Relation entre température minimale, maximale et pluie de demain
plt.figure(figsize=(10, 8))
sns.scatterplot(x='MinTemp', y='MaxTemp', hue='RainTomorrow', data=df)
plt.title('Relation entre Température minimale, Température maximale et Pluie de demain')
plt.xlabel('Température minimale')
plt.ylabel('Température maximale')
plt.show()

#Pluie de demain selon la localisation et la température
plt.figure(figsize=(10, 8))
sns.scatterplot(x='MinTemp', y='MaxTemp', hue='Location', data=df, palette='tab20')
plt.title('Relation entre Température minimale, Température maximale, Pluie de demain et Localisation')
plt.xlabel('Température minimale')
plt.ylabel('Température maximale')
plt.show()

#Relation entre température maximale par ville et risque de pluie
df_filtered = df[df['Location'].isin(['Sydney', 'Cairns', 'Perth', 'Canberra', 'Melbourne'])]
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_filtered, x='Location', y='Pressure3pm', hue='RainTomorrow')
plt.title("Pression Atmosphérique à 15h par Ville et Risque de Pluie")
plt.xlabel("Ville")
plt.ylabel("Pression Atmosphérique à 15h")
plt.show()

#Relation entre pression à 15h et risque de pluie
plt.figure(figsize=(10, 8))
sns.boxplot(data=df_filtered, x='Location', y='Pressure3pm', hue='RainTomorrow')
plt.title("Pression Atmosphérique à 15h par Ville et Risque de Pluie")
plt.xlabel("Ville")
plt.ylabel("Pression Atmosphérique à 15h")
plt.show()