# =============================================================
# Projet : Prédiction de notes d'étudiants
# Auteur  : Abderrahmane BELARIF
# Dataset : Student Performance (UCI)
# Modèle  : Régression linéaire (scikit-learn)
# =============================================================

# --- 1. Importer les bibliothèques nécessaires ---
import pandas as pd                          # Pour manipuler les données (tableaux)
import matplotlib.pyplot as plt              # Pour visualiser les résultats
from sklearn.linear_model import LinearRegression   # Le modèle ML
from sklearn.model_selection import train_test_split # Pour diviser les données
from sklearn.metrics import mean_absolute_error, r2_score  # Pour évaluer le modèle

# =============================================================
# --- 2. Charger le dataset ---
# =============================================================
# Télécharge le fichier student-mat.csv depuis UCI Machine Learning Repository :
# https://archive.ics.uci.edu/dataset/320/student+performance
# Place le fichier dans le même dossier que ce script.

import os
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "student-mat.csv"), sep=";")  # Le séparateur est ";" dans ce fichier

# Afficher les 5 premières lignes pour vérifier que le chargement est correct
print("=== Aperçu du dataset ===")
print(df.head())
print(f"\nNombre d'élèves dans le dataset : {len(df)}")

# =============================================================
# --- 3. Sélectionner les variables utiles ---
# =============================================================
# On garde uniquement les colonnes qui influencent la note finale (G3)
# studytime = heures de révision par semaine (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)
# absences  = nombre d'absences
# G1        = note du 1er trimestre
# G2        = note du 2ème trimestre
# G3        = note finale (c'est ce qu'on veut prédire)

colonnes_utiles = ["studytime", "absences", "G1", "G2", "G3"]
df = df[colonnes_utiles]

# Vérifier qu'il n'y a pas de valeurs manquantes
print("\n=== Valeurs manquantes par colonne ===")
print(df.isnull().sum())

# =============================================================
# --- 4. Séparer les données en entrées (X) et sortie (y) ---
# =============================================================
# X = les variables que le modèle utilise pour apprendre
# y = ce que le modèle doit prédire (la note finale G3)

X = df[["studytime", "absences", "G1", "G2"]]  # Entrées
y = df["G3"]                                    # Sortie à prédire

# =============================================================
# --- 5. Diviser le dataset : 80% entraînement / 20% test ---
# =============================================================
# Le modèle apprend sur 80% des données
# On teste sa précision sur les 20% restants (données inconnues)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% pour le test
    random_state=42      # Pour avoir les mêmes résultats à chaque exécution
)

print(f"\n=== Division des données ===")
print(f"Données d'entraînement : {len(X_train)} élèves")
print(f"Données de test        : {len(X_test)} élèves")

# =============================================================
# --- 6. Créer et entraîner le modèle ---
# =============================================================
# La régression linéaire cherche la meilleure droite qui relie
# les heures de révision (et autres facteurs) à la note finale

modele = LinearRegression()       # Créer le modèle
modele.fit(X_train, y_train)      # Entraîner le modèle sur les données d'entraînement

print("\n=== Modèle entraîné avec succès ===")

# =============================================================
# --- 7. Évaluer la précision du modèle ---
# =============================================================
# On teste le modèle sur les données qu'il n'a jamais vues (X_test)

y_pred = modele.predict(X_test)   # Prédictions du modèle

mae = mean_absolute_error(y_test, y_pred)   # Erreur moyenne en points
r2  = r2_score(y_test, y_pred)              # Score R² : 1.0 = parfait, 0 = nul

print("\n=== Résultats du modèle ===")
print(f"Erreur moyenne (MAE) : {mae:.2f} points")
print(f"Score R²             : {r2:.2f}  (plus c'est proche de 1, mieux c'est)")

# =============================================================
# --- 8. Visualiser les prédictions vs les vraies notes ---
# =============================================================

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="steelblue", alpha=0.7, label="Prédictions")
plt.plot([0, 20], [0, 20], color="red", linestyle="--", label="Prédiction parfaite")
plt.xlabel("Note réelle (G3)")
plt.ylabel("Note prédite")
plt.title("Comparaison : Notes réelles vs Notes prédites")
plt.legend()
plt.tight_layout()
plt.savefig("resultats_predictions.png")   # Sauvegarde le graphique
plt.show()
print("\nGraphique sauvegardé : resultats_predictions.png")

# =============================================================
# --- 9. Faire une prédiction personnalisée ---
# =============================================================
# Exemple : un élève avec ces caractéristiques :
# - studytime = 2  (révise 2 à 5h par semaine)
# - absences  = 1  (1 absences)
# - G1        = 14 (note du 1er trimestre)
# - G2        = 15 (note du 2ème trimestre)

print("\n=== Prédiction personnalisée ===")
nouvel_eleve = [[2, 1, 14, 15]]
note_predite = modele.predict(nouvel_eleve)
print(f"Note finale prédite : {note_predite[0]:.1f} / 20")
