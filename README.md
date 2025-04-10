# 💳 Projet Deep Learning – Détection de Fraude Bancaire

Ce projet a pour objectif de détecter des transactions bancaires frauduleuses en utilisant des techniques de Machine Learning et de Deep Learning, avec un **déploiement en temps réel** grâce à **TensorFlow Serving** et **Docker**, hébergé sur **Render**.

---

## 📁 Structure du Projet

```bash
Projet-Deep-Learning/
│
├── creditcard.csv              # Données initiales (non push sur GitHub)
├── fraude_detection.py         # Modèle de base (MLP) avec visualisations
├── fraude_detection_lstm.py    # Modèle avec LSTM (séries temporelles)
├── fraude_detection_gru.py     # Modèle avec GRU
├── client.py                   # Script de test de l'API en ligne
├── Dockerfile                  # Pour Docker + TensorFlow Serving
├── export_model/               # Modèle MLP exporté pour déploiement
├── export_lstm_model/          # Modèle LSTM exporté
├── export_gru_model/           # Modèle GRU exporté
└── README.md                   # Ce fichier

🧠 Méthodologie
1. Préparation des données
Dataset : creditcard.csv

Standardisation avec StandardScaler

Rééquilibrage des classes via SMOTE

2. Split des données
80% pour l'entraînement

20% pour le test

🧪 Modélisation
✅ Modèle MLP (Multi-Layer Perceptron)
Tuné avec KerasTuner (Random Search)

Sauvegarde avec best_model.save(...)

Visualisation : matrice de confusion, courbe ROC, SHAP

✅ Modèle LSTM
Capture les relations temporelles entre les transactions

Utilisation de return_sequences=True pour empiler plusieurs couches

✅ Modèle GRU
Similaire à LSTM, plus léger, parfois plus rapide

Bonne performance sur les séquences

📊 Visualisations incluses
Matrice de confusion

Classification report (f1-score par classe)

Courbe ROC

Graphique SHAP pour interprétation des prédictions

🐳 Docker & TensorFlow Serving
Objectif :
Déployer le modèle comme API REST accessible depuis le cloud.

Étapes :
Export du modèle :
best_model.export("export_model/1")

Dockerfile :

FROM tensorflow/serving
COPY export_model /models/fraude_model
ENV MODEL_NAME=fraude_model

Build Docker :
docker build -t fraude-serving .

Lancer en local (optionnel) :
docker run -p 8501:8501 fraude-serving

🌍 Déploiement Cloud (Render)
Nous avons utilisé Render (https://render.com) pour exposer le modèle sur le web.

➡️ L’API est accessible via :

https://projet-deep-learning.onrender.com/v1/models/fraude_model:predict

⚠️ Render Free s’endort après inactivité → 1ère requête peut prendre 50 sec.

🧪 Tester l’API (client.py)

import json
import requests
import numpy as np

# Exemple de transaction (à adapter selon tes colonnes)
example = np.random.rand(30).astype(np.float32)

# Reshape en (1, 1, 30)
payload = {
    "instances": example.reshape(1, 1, 30).tolist()
}

RENDER_URL = "https://projet-deep-learning.onrender.com/v1/models/fraude_model:predict"

response = requests.post(RENDER_URL, json=payload)

print("Résultat de l'API :")
print(json.dumps(response.json(), indent=2))

🔁 À quoi sert Docker ?
Crée un environnement portable et isolé

Facilite le déploiement rapide du modèle

Permet l’intégration avec TensorFlow Serving

Rendu compatible avec le cloud (Render, GCP, AWS)

☁️ Pourquoi déployer sur le cloud ?
Permet de rendre le modèle accessible en temps réel

Utile pour des systèmes embarqués ou applications web

Possibilité de faire des requêtes automatiques (API)

Scalabilité facile à grande échelle

🎯 Fonctionnement global
Chaque ligne représente une transaction.

Le modèle prédit 0 (normale) ou 1 (fraude).

Les résultats sont affichés dans les graphiques et peuvent être testés via l’API.

👥 Projet en groupe
L’un des membres héberge le modèle sur Render.

Tous les autres peuvent tester l’API en appelant l’URL dans client.py.

Le dépôt GitHub contient tous les fichiers nécessaires.

✨ Résultat
Précision du modèle : ~99.9%

ROC-AUC Score : ~0.999

Déploiement complet sur le web, avec tests en temps réel.