import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import glob
import itertools

# Function to load and clean data from a CSV file
def load_and_clean_data(filename):
    data = pd.read_csv(filename)
    for column in ['X', 'Y', 'Z']:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        filter = (data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)
        data = data.loc[filter]
    return data

# Flag to determine whether to plot all paths in one plot
plot_all_in_one = True

# Get all CSV files matching the pattern
csv_files = glob.glob('hand_tracking_data_*.csv')

# If plot_all_in_one is True, set up a combined plot
if plot_all_in_one:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)  # Assuming normalized x coordinates
    ax.set_ylim(0, 1)  # Assuming normalized z coordinates (depth)
    ax.set_zlim(0, 1)  # Assuming normalized y coordinates (height)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Z Axis (Depth)')
    ax.set_zlabel('Y Axis (Height)')

    # Define a list of colors for different paths
    colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])

    # List to hold all lines and their data
    lines = []
    all_data = []

    for csv_file in csv_files:
        data = load_and_clean_data(csv_file)
        color = next(colors)
        line, = ax.plot([], [], [], 'o-', markersize=4, color=color, label=csv_file)  # 'o-' means circles with lines
        lines.append(line)
        all_data.append(data)

    # Initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    # Animation function which updates figure data. This is called sequentially
    def animate(i):
        for line, data in zip(lines, all_data):
            if i < len(data):
                x = data['X'].iloc[:i+1]
                y = data['Y'].iloc[:i+1]
                z = data['Z'].iloc[:i+1]
                line.set_data(x, z)
                line.set_3d_properties(y)
        return lines

    # Call the animator
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=max(len(data) for data in all_data), interval=20, blit=False)

    plt.legend()
    plt.show()

else:
    # Iterate through each CSV file for separate plots
    for csv_file in csv_files:
        data = load_and_clean_data(csv_file)

        # Set up a separate plot for each file
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
