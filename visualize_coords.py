import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Path to the CSV file containing hand tracking data
input_file = 'hand_tracking_data.csv'

# Lists to store coordinates
frames = []
x_coords = []
y_coords = []
z_coords = []

# Read data from CSV file
with open(input_file, mode='r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        frame_num = int(row[0])
        x_coord = float(row[1])
        y_coord = float(row[2])
        z_coord = float(row[3])
        
        frames.append(frame_num)
        x_coords.append(x_coord)
        y_coords.append(y_coord)
        z_coords.append(z_coord)

# Plotting the data in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of hand coordinates
scatter = ax.scatter(x_coords, y_coords, z_coords, c=frames, cmap='viridis', marker='o', label='Hand Coordinates')

# Connect points as a path
ax.plot(x_coords, y_coords, z_coords, color='gray', alpha=0.5, linewidth=2, label='Hand Path')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('Hand Tracking Path in 3D')

# Add color bar which maps frame numbers to colors
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Frame Number')

# Add legend
ax.legend()

plt.tight_layout()
plt.show()
