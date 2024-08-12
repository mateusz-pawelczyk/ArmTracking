import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load data
df = pd.read_csv('abs_to_rel.csv')
df_rel = pd.DataFrame()
df_rel["Depth"] = df[df['Ratio'] == "finger"]["Depth"].reset_index().drop(columns=["index"]) - df[df['Ratio'] == "wrist"]["Depth"].reset_index().drop(columns=["index"])
df_rel["Landmark.z"] = df[df['Ratio'] == "finger"]["Landmark.z"].reset_index().drop(columns=["index"]) - df[df['Ratio'] == "wrist"]["Landmark.z"].reset_index().drop(columns=["index"])
df_rel["Ratio"] = df_rel["Depth"] / df_rel["Landmark.z"]
# Plotting multiple relationships
plt.figure(figsize=(12, 8))
sc = plt.scatter(df_rel['Landmark.z'], df_rel['Depth'], c=df_rel['Ratio'], cmap='viridis', alpha=0.6, edgecolors='w', s=100)
plt.colorbar(sc, label='Ratio')
plt.title('Depth vs. Landmark.z with Ratio Color Coding')
plt.xlabel('Landmark.z (relative depth)')
plt.ylabel('Depth (absolute depth)')
plt.grid(True)
plt.show()
