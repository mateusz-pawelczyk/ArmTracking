import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# CSV-Daten laden (ersetze 'your_data_file.csv' mit dem tatsächlichen Dateinamen)
df = pd.read_csv('processed_Trajectories.csv')

# Extrahiere die relevanten Daten (Schulter, Ellbogen, Handgelenk, ThumbTip)
shoulder = df[['Tasch:shoulder:X (mm)', 'Tasch:shoulder:Y (mm)', 'Tasch:shoulder:Z (mm)']].values
elbow = df[['Tasch:elbow:X (mm)', 'Tasch:elbow:Y (mm)', 'Tasch:elbow:Z (mm)']].values
wrist = df[['Tasch:wrist:X (mm)', 'Tasch:wrist:Y (mm)', 'Tasch:wrist:Z (mm)']].values
thumb_tip = df[['Tasch:ThumbTip:X (mm)', 'Tasch:ThumbTip:Y (mm)', 'Tasch:ThumbTip:Z (mm)']].values

# Normalisierung der Daten
scaler = MinMaxScaler()
shoulder = scaler.fit_transform(shoulder)
elbow = scaler.fit_transform(elbow)
wrist = scaler.fit_transform(wrist)
thumb_tip = scaler.fit_transform(thumb_tip)

# Features (Schulter, Ellbogen, Handgelenk) kombinieren
X_train = np.hstack([shoulder, elbow, wrist])

# Ziel (ThumbTip-Position)
Y_train = thumb_tip

# Definiere das Imitation Learning Modell
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3)  # Vorhersage der X, Y, Z Position des ThumbTips
])

# Modell kompilieren
model.compile(
    optimizer='adam',                 # Adam Optimizer
    loss='mean_squared_error',         # Mittlerer quadratischer Fehler für Regression
    metrics=['mean_absolute_error']    # Optionale Metrik für Überwachung
)

# Modell trainieren
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

# Visualisiere die Verlustkurve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Verlustkurve während des Trainings')
plt.xlabel('Epochen')
plt.ylabel('Verlust')
plt.legend()
plt.show()

# Beispielhafte Vorhersage mit neuen Eingaben (simulierte Daten, ersetze durch tatsächliche Werte)
new_shoulder_position = np.array([[0.2, 0.3, 0.4]])  # Normalisierte Eingabe
new_elbow_position = np.array([[0.3, 0.4, 0.5]])     # Normalisierte Eingabe
new_wrist_position = np.array([[0.4, 0.5, 0.6]])     # Normalisierte Eingabe

# Kombiniere die neuen Eingaben
new_input = np.hstack([new_shoulder_position, new_elbow_position, new_wrist_position])

# Vorhersage der ThumbTip Position
predicted_thumb_tip = model.predict(new_input)
print(f"Vorhergesagte ThumbTip Position: {predicted_thumb_tip}")
