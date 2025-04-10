# fraude_detection_lstm.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")

# 1. Chargement du dataset
data = pd.read_csv("creditcard.csv")

# 2. Prétraitement
X = data.drop(["Class"], axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 3. Mise en forme pour LSTM (reshape en [samples, timesteps=1, features])
X_train_seq = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_seq = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# 4. Modèle LSTM
def create_lstm_model(input_shape):
    model = keras.Sequential()
    model.add(layers.LSTM(64, input_shape=input_shape))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_lstm_model((1, X_train.shape[1]))

# 5. Entraînement
history = model.fit(X_train_seq, y_train, epochs=10, batch_size=64, validation_split=0.2)

# 6. Évaluation
predictions = (model.predict(X_test_seq) > 0.5).astype("int32")

# Matrice de confusion
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matrice de Confusion - LSTM")
plt.show()

# Rapport de classification
print(classification_report(y_test, predictions))

# ROC AUC
print("ROC-AUC Score:", roc_auc_score(y_test, predictions))
fpr, tpr, thresholds = roc_curve(y_test, predictions)
plt.plot(fpr, tpr, label='ROC Curve - LSTM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LSTM')
plt.legend()
plt.show()

# 7. Sauvegarde du modèle pour TensorFlow Serving
model_path = "export_lstm_model/1"
model.export(model_path)
print(f"Modèle LSTM exporté dans : {model_path}")
