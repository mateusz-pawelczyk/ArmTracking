import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Define the same model structure
class ImitationModel(nn.Module):
    def __init__(self):
        super(ImitationModel, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)  # Predict X, Y, Z positions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model and weights
model = ImitationModel()
model.load_state_dict(torch.load('imitation_model.pth'))
model.eval()  # Set model to evaluation mode

# Load and preprocess the data
df = pd.read_csv('csv/processed_Trajectories.csv')
shoulder = df[['Tasch:shoulder:X (mm)', 'Tasch:shoulder:Y (mm)', 'Tasch:shoulder:Z (mm)']].values
elbow = df[['Tasch:elbow:X (mm)', 'Tasch:elbow:Y (mm)', 'Tasch:elbow:Z (mm)']].values
wrist = df[['Tasch:wrist:X (mm)', 'Tasch:wrist:Y (mm)', 'Tasch:wrist:Z (mm)']].values

# Normalize the data using MinMaxScaler (same as during training)
scaler = MinMaxScaler()
X_all = np.vstack([shoulder, elbow, wrist])
X_all_scaled = scaler.fit_transform(X_all)

# Split back to shoulder, elbow, wrist
shoulder = X_all_scaled[:len(shoulder)]
elbow = X_all_scaled[len(shoulder):len(shoulder) + len(elbow)]
wrist = X_all_scaled[len(shoulder) + len(elbow):]

# Combine the features for model input
X_test = np.hstack([shoulder, elbow, wrist])

# Convert to torch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Predict the thumb_tip positions using the trained model
with torch.no_grad():
    predicted_thumb_tip = model(X_test_tensor).numpy()

# Prepare data for animation (we are predicting thumb_tip position)
all_xs, all_ys, all_zs = predicted_thumb_tip[:, 0], predicted_thumb_tip[:, 1], predicted_thumb_tip[:, 2]

# Setup initial plot for 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis limits based on predicted data
ax.set_xlim([all_xs.min(), all_xs.max()])
ax.set_ylim([all_ys.min(), all_ys.max()])
ax.set_zlim([all_zs.min(), all_zs.max()])

# Scatter plot for initial frame
graph = ax.scatter(all_xs[0], all_ys[0], all_zs[0], c="blue", marker='o')
title = ax.set_title('3D Predicted Movement')

# Define the update function for animation
def update_graph(num):
    graph._offsets3d = ([all_xs[num]], [all_ys[num]], [all_zs[num]])
    title.set_text(f'3D Predicted Movement, Frame={num}')
    return graph, title

# Create and show the animation
ani = animation.FuncAnimation(fig, update_graph, frames=len(all_xs), interval=5, blit=False)
plt.show()
