# fraude_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
import shap
import warnings
warnings.filterwarnings("ignore")

# 1. Chargement du dataset
data = pd.read_csv("creditcard.csv")

# 2. Prétraitement
X = data.drop(["Class"], axis=1)
y = data["Class"]

#Mise à l'échelle

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 3. Modélisation
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units_input', min_value=32, max_value=256, step=32), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    for i in range(hp.Int('n_layers', 1, 3)):
        model.add(layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32), activation='relu'))
        model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, executions_per_trial=1, directory='tuner', project_name='fraude_model')
tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

best_model = tuner.get_best_models(num_models=1)[0]

# 4. Évaluation
predictions = (best_model.predict(X_test) > 0.5).astype("int32")

# Matrice de confusion
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matrice de Confusion")
plt.show()

# Rapport de classification
print(classification_report(y_test, predictions))
report = classification_report(y_test, predictions, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(8, 5))
sns.barplot(x=df_report.index, y="f1-score", data=df_report.reset_index())
plt.title("F1-Score par classe")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ROC AUC
print("ROC-AUC Score:", roc_auc_score(y_test, predictions))
fpr, tpr, thresholds = roc_curve(y_test, predictions)
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 5. Interprétabilité avec SHAP
try:
    X_sample = X_test[:100]
    explainer = shap.Explainer(best_model.predict, X_sample.astype(np.float32))
    shap_values = explainer(X_sample.astype(np.float32))
    shap.plots.waterfall(shap_values[0])
except Exception as e:
    print("Erreur avec SHAP:", e)
