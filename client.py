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
