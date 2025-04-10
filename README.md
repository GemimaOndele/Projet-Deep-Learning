# 💳 Détection de Fraude sur les Transactions Bancaires

Ce projet a pour objectif de détecter les fraudes sur les transactions bancaires en utilisant des modèles de Deep Learning (MLP, GRU, LSTM) et un déploiement en production avec **TensorFlow Serving** via **Docker** et **Render**.

---

## 📊 Dataset utilisé

- **Nom** : creditcard.csv
- **Taille** : ~144 MB
- **Source** : Transactions réelles (anonymisées)
- **Caractéristiques** :
  - 30 colonnes d'entrée (V1, V2, ..., V28, Amount, Time)
  - Colonne cible : `Class` (0 = transaction normale, 1 = fraude)

---

## ⚙️ Étapes du projet

1. **Chargement et analyse exploratoire du dataset**
2. **Prétraitement des données** :
   - Normalisation avec `StandardScaler`
   - Rééquilibrage avec **SMOTE**
   - Séparation des données (80% train / 20% test)
3. **Modélisation Deep Learning** :
   - Modèle dense MLP avec `Keras Tuner`
   - Modèle LSTM
   - Modèle GRU
4. **Évaluation** :
   - Matrice de confusion
   - Classification report
   - Courbe ROC
   - Score ROC AUC
   - Interprétation des résultats avec **SHAP**
5. **Déploiement** :
   - Export du modèle au format TensorFlow Serving
   - Conteneurisation avec **Docker**
   - Déploiement cloud avec **Render**

---

## 🧠 Modèles testés

| Modèle | Accuracy | ROC AUC |
|--------|----------|---------|
| MLP (dense) | ~99.9% | 0.9996 |
| GRU | ~99.9% | 0.9995 |
| LSTM | ~99.9% | 0.9996 |

---

## 🚀 Déploiement API (Render)

- Le modèle est déployé avec **TensorFlow Serving** via Docker
- Accès via une API REST :
  ```bash
  POST https://projet-deep-learning.onrender.com/v1/models/fraude_model:predict
Exemple d’appel (client Python) :

python
Copier
Modifier
import requests
import json
import numpy as np

url = "https://projet-deep-learning.onrender.com/v1/models/fraude_model:predict"
data = {
    "instances": [[[0.1, 0.2, ..., 0.3]]]  # Format GRU/LSTM : shape (1, 1, 30)
}

response = requests.post(url, json=data)
print(response.json())
🐳 Dockerfile utilisé
dockerfile
Copier
Modifier
FROM tensorflow/serving
COPY export_gru_model /models/fraude_model
ENV MODEL_NAME=fraude_model
💻 Lancer en local (optionnel)
bash
Copier
Modifier
docker run -p 8501:8501 \
  --name fraude_model_serving \
  --mount type=bind,source=$(pwd)/export_gru_model,target=/models/fraude_model \
  -e MODEL_NAME=fraude_model \
  -t tensorflow/serving
👥 Travail collaboratif
Projet réalisé en groupe dans le cadre d'un cours de Deep Learning.
Chaque membre a contribué aux parties : nettoyage, modélisation, tuning, interprétation, et déploiement.

📂 Arborescence du projet
bash
Copier
Modifier
fraude_project/
│
├── creditcard.csv
├── fraude_detection.py              # MLP + SHAP
├── fraude_detection_lstm.py        # Modèle LSTM
├── fraude_detection_gru.py         # Modèle GRU
├── client.py                       # Exemple appel API
├── Dockerfile                      # Déploiement Render
├── export_model/                   # MLP SavedModel
├── export_lstm_model/              # LSTM SavedModel
├── export_gru_model/               # GRU SavedModel
└── README.md
📈 Résultats finaux
Précision du modèle très élevée (presque 100%)

Modèles robustes sur données rééquilibrées

API en production disponible sur Render 🚀

