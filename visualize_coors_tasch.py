import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Pfad zur CSV-Datei
csv_file_path = 'hand_tracking_data.csv'

# Daten aus der CSV-Datei laden
df = pd.read_csv(csv_file_path)

# Relative Koordinaten extrahieren und umordnen
x = df['Z']            # Tiefe wird zur X-Achse
y = df['X']            # Horizontale Position wird zur Y-Achse
z = df['Y']        # Vertikale Position wird zur Z-Achse und invertiert
z_inverted = df['Z']  # Invertiere Z-Achse für Tiefe

# 3D-Plot erstellen
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Punkte plotten und verbinden
ax.plot(z_inverted, y, z, marker='o')

# Achsenbeschriftungen hinzufügen
ax.set_xlabel('Depth (Z)')
ax.set_ylabel('Horizontal (X)')
ax.set_zlabel('Vertical (Y)')


# Plot anzeigen
plt.show()