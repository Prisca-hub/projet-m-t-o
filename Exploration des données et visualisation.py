#Exploration des données et visualisation
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


# Relations température maximale vs pluie de demain pour chaque ville
g = sns.FacetGrid(df, col='Location', hue='RainTomorrow_Yes', col_wrap=3, height=5)
g.map(sns.scatterplot, 'MaxTemp', 'RainTomorrow_Yes', alpha=0.7, s=60)
g.set_axis_labels('Température Maximale (MaxTemp)', 'Pluie Demain (RainTomorrow)')
g.set_titles('{col_name}')
g.add_legend()
plt.show()
