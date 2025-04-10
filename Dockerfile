# Utilise l'image TensorFlow Serving officielle
FROM tensorflow/serving

# Copie le modèle dans le bon répertoire attendu par TF Serving
COPY export_gru_model /models/fraude_model

# Définir le nom du modèle à servir
ENV MODEL_NAME=fraude_model
