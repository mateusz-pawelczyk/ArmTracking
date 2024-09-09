import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Load and prepare data
df = pd.read_csv('csv/processed_data.csv')
df.fillna(df.mean(), inplace=True)

num_landmarks = (df.shape[1] - 2) // 3  # Assume frame column + X, Y, Z per landmark

# Precompute all coordinates and determine global min/max for setting axis limits
all_xs, all_ys, all_zs = [], [], []
global_min_x, global_max_x = float('inf'), float('-inf')
global_min_y, global_max_y = float('inf'), float('-inf')
global_min_z, global_max_z = float('inf'), float('-inf')

for num in range(df.shape[0]):
    xs, ys, zs = [], [], []
    for i in range(2, num_landmarks * 3 + 2, 3):
        x, y, z = df.iloc[num, i], df.iloc[num, i+1], df.iloc[num, i+2]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        # Update global min and max
        global_min_x, global_max_x = min(global_min_x, x), max(global_max_x, x)
        global_min_y, global_max_y = min(global_min_y, y), max(global_max_y, y)
        global_min_z, global_max_z = min(global_min_z, z), max(global_max_z, z)
    all_xs.append(xs)
    all_ys.append(ys)
    all_zs.append(zs)

# Define the update function using precomputed data
def update_graph(num):
    graph._offsets3d = (all_xs[num], all_ys[num], all_zs[num])
    for index, (s, t) in enumerate(connections):
        lines[index].set_data([all_xs[num][s], all_xs[num][t]], [all_ys[num][s], all_ys[num][t]])
        lines[index].set_3d_properties([all_zs[num][s], all_zs[num][t]])
    title.set_text('3D Test, time={}'.format(num))
    return graph, title

# Setup initial plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set fixed axis limits based on the global minimum and maximum values
ax.set_xlim([global_min_x, global_max_x])
ax.set_ylim([global_min_y, global_max_y])
ax.set_zlim([global_min_z, global_max_z])

graph = ax.scatter(all_xs[0], all_ys[0], all_zs[0], c="blue", marker='o')
lines = []
connections = [(0, 2), (0,1), (1,2), (2, 3), (2,4),(3,4), (3, 5), (3, 6), (6,7)]
for s, t in connections:
    line, = ax.plot([all_xs[0][s], all_xs[0][t]], [all_ys[0][s], all_ys[0][t]], [all_zs[0][s], all_zs[0][t]], color="blue")
    lines.append(line)
title = ax.set_title('3D Test')

# Create and show the animation
ani = animation.FuncAnimation(fig, update_graph, frames=len(all_xs), interval=1, blit=False)
plt.show()
