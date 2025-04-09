# 💳 Détection de Fraude sur les Transactions Bancaires

Ce projet utilise des techniques de **Deep Learning (TensorFlow/Keras)** et de **machine learning** pour détecter automatiquement les fraudes sur les cartes bancaires.  
Il s'appuie sur un dataset public hautement déséquilibré et propose un modèle entraîné, évalué et optimisé.

---

## 🎯 Objectifs

- Détecter les transactions frauduleuses avec une grande précision.
- Utiliser un modèle de réseau de neurones profond (MLP) via **TensorFlow/Keras**.
- Rééquilibrer les classes grâce à **SMOTE**.
- Évaluer les performances avec des métriques adaptées : matrice de confusion, ROC AUC, f1-score.
- Interpréter le modèle avec **SHAP**.

---

## 📁 Structure du projet

fraude_project/ ├── fraude_detection.py # Script principal ├── .gitignore # Fichiers à exclure (ex: creditcard.csv) ├── requirements.txt # Dépendances Python ├── README.md # Ce fichier └── outputs/ # (Optionnel) Graphiques générés

yaml
Copier
Modifier

---

## 🔍 Dataset utilisé

📚 **Dataset : [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**  
- 284 807 transactions
- 492 fraudes (soit 0.17 %)
- Caractéristiques anonymisées (`V1` à `V28`) via PCA

> ⚠️ Le fichier `creditcard.csv` **n’est pas inclus dans ce dépôt GitHub** (car >100 Mo).  
Veuillez le télécharger manuellement depuis Kaggle.

---

## ⚙️ Installation

### 1. Cloner le projet :

```bash
git clone https://github.com/GemimaOndele/Projet-Deep-Learning.git
cd Projet-Deep-Learning


**Détails du projet **

# Projet-Deep-Learning

PROJET DEEP LEARNING  

         Détection de Fraude sur les Transactions Bancaires Utilisation de Deep Learning avec TensorFlow et Scikit-learn 


Introduction et Objectifs 

Objectif principal : Détecter les transactions bancaires frauduleuses en temps réel. 

Technologies utilisées : Deep Learning avec TensorFlow. 

Dataset : Utilisation du dataset public de Kaggle. 

Description du Dataset 

Dataset : Credit Card Fraud Detection sur Kaggle. 

Nombre de transactions : 284,807. 

Pourcentage de fraudes : 0.17%. 

Caractéristiques : 28 caractéristiques anonymisées issues de PCA (V1 à V28). 

Prétraitement des Données 

Normalisation des données : Utilisez StandardScaler de Scikit-learn pour normaliser les données. 

Rééquilibrage des classes : Utilisez SMOTE (Synthetic Minority Over-sampling Technique) pour équilibrer les classes. 

Séparation des données : Divisez les données en 80% pour l'entraînement et 20% pour le test. 

Modélisation avec TensorFlow 

Architecture du modèle : Utilisez un réseau de neurones multicouche. 

Fonctions d'activation : ReLU pour les couches cachées et Sigmoïde pour la sortie. 

Optimisation : Utilisez l'optimiseur Adam. 

Ajustement des hyperparamètres : Utilisez Keras Tuner pour ajuster les hyperparamètres. 

Formule de base : $ y = \sigma(W \cdot X + b) $ 

Évaluation et Optimisation 

Matrice de confusion : Pour évaluer les performances du modèle. 

Courbe ROC-AUC : Pour mesurer la performance du modèle. 

Optimisation : Utilisez GridSearchCV pour optimiser les paramètres du modèle. 

Technologies Utilisées 

Python : Pour le développement. 

TensorFlow/Keras : Pour le modèle de Deep Learning. 

Scikit-learn : Pour l’évaluation des performances. 

Pandas et NumPy : Pour la manipulation des données. 

Défis Techniques et Solutions 

Déséquilibre des classes : Utilisez SMOTE pour équilibrer les classes. 

Overfitting : Utilisez la régularisation avec Dropout. 

Interprétabilité : Utilisez SHAP et LIME pour interpréter les résultats du modèle. 

Extensions et Perspectives 

Détection en temps réel : Utilisez TensorFlow Serving. 

Modèles avancés : Utilisez LSTM et GRU pour capturer les séquences temporelles des transactions. 

Déploiement sur le Cloud : Utilisez GCP ou AWS avec TensorFlow Serving pour des prédictions à grande échelle. 

Questions ? 

Merci pour votre attention ! Questions ? 

Ce projet a été réalisé par **@Hubert CHAVASSE, @Hadrien LENNON, @Mohamed GHARMAOUI, @Gemima ONDELE POUROU et @Niangoran Esther BOKA.**
