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
