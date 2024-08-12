import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm

# CSV file read
df = pd.read_csv('recorded_data.csv')
# Fill NaN values with the mean of the column (or choose another appropriate method)
df.fillna(df.mean(), inplace=True)

# Replace Inf values with the maximum finite value in the column
#for col in df.columns:
#    max_val = df[df[col] != float('inf')][col].max()
#    df[col].replace(float('inf'), max_val, inplace=True)
#
#    mask = (df[col] < 0.10) & (df.index > 0) & (df.index < len(df) - 1)
#    df.loc[mask, col] = (df[col].shift(-1) + df[col].shift(1)) / 2
#    # Number of landmarks
num_landmarks = (df.shape[1] - 1) // 3  # Subtract 1 for the Frame column, divide by 3 (X, Y, Z per landmark)

connections = [(0, 1), (1, 2)]

connections_hand = [ 
               (2, 3), (3,4), (4,5),(5,6),
               (2,7),(7,8),(8,9),(9,10),
               (7,11),(11,12),(12,13),(13,14),
               (11,15),(15,16),(16,17),(17,18),
               (15,19),(19,20),(20,21),(21,22),
               (2,19)]  # Example connections

# Define a function to update the graph for each frame
def update_graph(num):
    # Prepare empty lists to hold all coordinates for this frame
    xs, ys, zs = [], [], []
    
    # Accumulate all coordinates from this frame
    for i in range(1, num_landmarks * 3 + 1, 3):  # Assuming num_landmarks * 3 total landmark-related columns
        xs.append(df.iloc[num, i])   # X coordinate
        ys.append(df.iloc[num, i + 1]) # Y coordinate
        zs.append(df.iloc[num, i + 2]) # Z coordinate

    # Update scatter plot data
    graph._offsets3d = (xs, ys, zs)

    # Clear previous lines and draw new ones
    for line in lines:
        line.remove()
    lines.clear()
    
    # Draw lines between points
    cmap = cm.get_cmap('inferno')  # This can be any colormap that matplotlib supports

    # Assume the index range for color mapping is between 0 and the maximum index in your connections
    norm = plt.Normalize(0, max(max(connections)))  # Normalize to scale colormap

    for s, t in connections:
        # Normalize the index 's' for the color mapping
        color = cmap(norm(s))
        line, = ax.plot([xs[s], xs[t]], [ys[s], ys[t]], [zs[s], zs[t]], color="blue")
        lines.append(line)

    for s, t in connections_hand:
        # Normalize the index 's' for the color mapping
        color = cmap(norm(s))
        line, = ax.plot([xs[s], xs[t]], [ys[s], ys[t]], [zs[s], zs[t]], color="red")
        lines.append(line)
    
    title.set_text('3D Test, time={}'.format(num))
    return graph, title

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

lines = []  # Initialize a list to store line objects
graph = ax.scatter(df.iloc[0, 1:4:num_landmarks*3].values, df.iloc[0, 2:4:num_landmarks*3].values, df.iloc[0, 3:4:num_landmarks*3].values, c="blue", marker='o')

# Create the animation
ani = animation.FuncAnimation(fig, update_graph, frames=df.shape[0], interval=100, blit=False)

plt.show()
