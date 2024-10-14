import pandas as pd
import numpy as np
import pyvista as pv
import time

# Load and prepare data
df = pd.read_csv('csv/processed_data.csv')
df.fillna(df.mean(), inplace=True)

num_landmarks = (df.shape[1] - 2) // 3  # Assume frame column + X, Y, Z per landmark

# Precompute all frames
all_frames = []
global_min_x, global_max_x = float('inf'), float('-inf')
global_min_y, global_max_y = float('inf'), float('-inf')
global_min_z, global_max_z = float('inf'), float('-inf')

for num in range(df.shape[0]):
    frame_data = {'xs': [], 'ys': [], 'zs': [], 'lines': []}
    for i in range(2, num_landmarks * 3 + 2, 3):
        x, y, z = df.iloc[num, i], df.iloc[num, i+1], df.iloc[num, i+2]
        frame_data['xs'].append(x)
        frame_data['ys'].append(y)
        frame_data['zs'].append(z)
        # Update global min and max
        global_min_x, global_max_x = min(global_min_x, x), max(global_max_x, x)
        global_min_y, global_max_y = min(global_min_y, y), max(global_max_y, y)
        global_min_z, global_max_z = min(global_min_z, z), max(global_max_z, z)
    
    # Precompute line segments
    frame_lines = []
    connections = [(0, 2), (0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5), (3, 6), (6, 7)]
    for s, t in connections:
        line_data = {
            'x': [frame_data['xs'][s], frame_data['xs'][t]],
            'y': [frame_data['ys'][s], frame_data['ys'][t]],
            'z': [frame_data['zs'][s], frame_data['zs'][t]]
        }
        frame_lines.append(line_data)
    frame_data['lines'] = frame_lines
    
    all_frames.append(frame_data)

# Now, create a PyVista plotter
plotter = pv.Plotter()
plotter.show_axes()

# Add initial scatter plot for landmarks
points = np.column_stack((all_frames[0]['xs'], all_frames[0]['ys'], all_frames[0]['zs']))
point_cloud = pv.PolyData(points)
scatter = plotter.add_mesh(point_cloud, color='red', point_size=10, render_points_as_spheres=True)

# Add initial lines
line_actors = []
for line in all_frames[0]['lines']:
    line_pts = np.column_stack((line['x'], line['y'], line['z']))
    line_mesh = pv.Line(line_pts[0], line_pts[1])
    actor = plotter.add_mesh(line_mesh, color='blue', line_width=2)
    line_actors.append(actor)

# Set the camera bounds to ensure all points are in view
plotter.camera.position = [
    (global_max_x + global_min_x) / 2,
    (global_max_y + global_min_y) / 2,
    global_max_z + (global_max_z - global_min_z) * 2
]
plotter.camera.focal_point = [
    (global_max_x + global_min_x) / 2,
    (global_max_y + global_min_y) / 2,
    (global_max_z + global_min_z) / 2
]
plotter.camera.zoom(1.5)  # Adjust zoom level if needed

# Define variables for animation
current_frame = 0
num_frames = len(all_frames)
playback_speed = 1.0  # Initial playback speed

# Now, create a slider to adjust playback speed
def update_playback_speed(value):
    global playback_speed
    playback_speed = value

plotter.add_slider_widget(
    update_playback_speed,
    [0.1, 4.0],
    value=1.0,
    title='Playback Speed',
    pointa=(0.2, 0.1),
    pointb=(0.8, 0.1),
)

# Start the interactive plotter and animation loop
plotter.show(auto_close=False)

while True:
    frame_rate = 120  # Original data frame rate
    frame_interval = 1.0 / (frame_rate * playback_speed)  # Time per frame
    
    # Update points
    frame_data = all_frames[current_frame]
    points = np.column_stack((frame_data['xs'], frame_data['ys'], frame_data['zs']))
    point_cloud.points = points

    # Update lines
    for i, line in enumerate(frame_data['lines']):
        line_pts = np.column_stack((line['x'], line['y'], line['z']))
        line_actors[i].SetInputData(pv.Line(line_pts[0], line_pts[1]))

    plotter.update()

    # Increment frame index
    current_frame = (current_frame + 1) % num_frames

    # Wait for the next frame
    time.sleep(frame_interval)

    if not plotter.is_active:
        break

plotter.close()
