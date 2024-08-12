import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Load joint coordinates from CSV file
input_file = "joint_coordinates.csv"
data = []

with open(input_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header
    for row in reader:
        data.append([float(val) for val in row])

# Extract frames and landmarks
timestamps = [frame[0] for frame in data]
landmarks = [frame[1:] for frame in data]

# Define the connections for the stickman (Mediapipe's POSE_CONNECTIONS)
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (12, 14), (14, 16), (16, 20), (20, 18),
    (18, 16), (11, 13), (13, 15), (15, 19), (19, 17), (17, 15),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30)
]

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis limits and labels
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Initialize the plot elements we want to animate
lines = []
for connection in connections:
    line, = ax.plot([], [], [], 'o-', lw=2)
    lines.append(line)

def update(num):
    # Update each line according to the current frame's joint coordinates
    frame_data = landmarks[num]

    # Find the minimum Y value (this would be the lowest point, usually the feet)
    y_values = [-frame_data[i * 4 + 1] for i in range(len(frame_data) // 4)]
    min_y = min(y_values)

    # Find the average X and Z values to center the stickman
    x_values = [frame_data[i * 4] for i in range(len(frame_data) // 4)]
    z_values = [-frame_data[i * 4 + 2] for i in range(len(frame_data) // 4)]  # Z controls depth
    avg_x = sum(x_values) / len(x_values)
    avg_z = sum(z_values) / len(z_values)

    for line, connection in zip(lines, connections):
        start_idx, end_idx = connection
        xs = [frame_data[start_idx * 4] - avg_x, frame_data[end_idx * 4] - avg_x]
        ys = [frame_data[start_idx * 4 + 2] - avg_z, frame_data[end_idx * 4 + 2] - avg_z]  # Z now controls depth
        zs = [-frame_data[start_idx * 4 + 1] - min_y, -frame_data[end_idx * 4 + 1] - min_y]  # Y controls height, adjusted by min_y

        line.set_data(xs, ys)
        line.set_3d_properties(zs)

    ax.set_title(f"Frame {num}, Time: {timestamps[num]:.2f}s")
    return lines

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(landmarks), interval=50, blit=False)

# Show the animation
plt.show()
