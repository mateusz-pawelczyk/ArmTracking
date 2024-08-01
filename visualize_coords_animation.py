import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Load the data from CSV
data = pd.read_csv('hand_tracking_data.csv')

# IQR-based outlier removal for each column
for column in ['X', 'Y', 'Z']:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    filter = (data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)
    data = data.loc[filter]

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 1)  # Assuming normalized x coordinates
ax.set_ylim(0, 1)  # Assuming normalized z coordinates (depth)
ax.set_zlim(0, 1)  # Assuming normalized y coordinates (height)
ax.set_xlabel('X Axis')
ax.set_ylabel('Z Axis (Depth)')
ax.set_zlabel('Y Axis (Height)')
line, = ax.plot([], [], [], 'ro-', markersize=4)  # 'ro-' means red circles with lines

# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

# Animation function which updates figure data. This is called sequentially
def animate(i):
    x = data['X'].iloc[:i+1]
    y = data['Y'].iloc[:i+1]
    z = data['Z'].iloc[:i+1]
    line.set_data(x, z)
    line.set_3d_properties(y)
    return line,

# Call the animator. blit=True means only re-draw the parts that have changed.
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(data), interval=20, blit=False)

# Show the animation
plt.show()
