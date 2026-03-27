# Prédicteur de Notes d'Étudiants

Un projet de machine learning pour prédire la note finale d'un élève en maths
à partir de ses habitudes de travail et de ses notes trimestrielles.

## Ce que fait le programme

- Charge un dataset réel de 395 élèves (UCI Machine Learning Repository)
- Entraîne un modèle de régression linéaire pour prédire la note finale (G3)
- Évalue la précision du modèle avec le MAE et le score R²
- Visualise les notes prédites vs les notes réelles (graphique)
- Compare 4 scénarios d'élèves selon leur temps de révision

## Dataset

**Student Performance Dataset** — University of California, Irvine  
Source : [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)

Variables utilisées :

`studytime` : Temps de révision hebdomadaire (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)
`absences` : Nombre d'absences scolaires
`G1` : Note du 1er trimestre (sur 20)
`G2` : Note du 2ème trimestre (sur 20)
`G3` : Note finale (sur 20) — variable cible à prédire

---

## Ce que j'ai observé

On constate que G1 et G2 dominent largement la prédiction de G3, bien plus que studytime et absences. 
Ceci s'explique par le fait que l'effet de ces deux paramètres est déjà contenu dans les notes trimestrielles : 
un élève qui révise peu ou s'absente souvent le paie directement sur G1 et G2. Comme illustration, G2 affiche une 
corrélation d'environ 0.90 avec G3, contre seulement 0.10 pour studytime

## Installation et exécution

**1. Installer les dépendances**
```bash
pip install pandas matplotlib scikit-learn
```

**2. Placer le fichier dataset dans le même dossier**  
Télécharger `student-mat.csv` depuis [UCI](https://archive.ics.uci.edu/dataset/320/student+performance)
et le placer dans le même répertoire que le script.

**3. Lancer le script**
```bash
python student_grade_predictor.py
```

## Résultats produits

- Score MAE (erreur moyenne en points)
- Score R² (qualité globale du modèle)
- Graphique `resultats_predictions.png` (notes réelles vs prédites)
- Tableau comparatif des 4 scénarios selon studytime

## Technologies utilisées

- Python 3
- pandas
- scikit-learn
- matplotlib

## Auteur

Abderrahmane BELARIF — Terminale Générale (Maths-Physique)

