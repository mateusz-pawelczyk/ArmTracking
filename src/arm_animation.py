from data_cleaning_tools import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Function to visualize arm movement for two sequences side by side
def visualize_two_sequences(df1, sequence_index1, df2, sequence_index2, animation_speed=1):
    # Filter the data for the selected sequences
    sequence_data1 = df1[df1['Sequence'] == sequence_index1].copy()
    sequence_data2 = df2[df2['Sequence'] == sequence_index2].copy()

    # Interpolate missing data
    sequence_data1.interpolate(method='linear', inplace=True)
    sequence_data2.interpolate(method='linear', inplace=True)

    # Extract the relevant joint positions for the first sequence
    shoulder1 = sequence_data1[['shoulder:X', 'shoulder:Y', 'shoulder:Z']].values
    elbow1 = sequence_data1[['elbow:X', 'elbow:Y', 'elbow:Z']].values
    wrist1 = sequence_data1[['wrist:X', 'wrist:Y', 'wrist:Z']].values
    thumb_tip1 = sequence_data1[['ThumbTip:X', 'ThumbTip:Y', 'ThumbTip:Z']].values

    # Extract the relevant joint positions for the second sequence
    shoulder2 = sequence_data2[['shoulder:X', 'shoulder:Y', 'shoulder:Z']].values
    elbow2 = sequence_data2[['elbow:X', 'elbow:Y', 'elbow:Z']].values
    wrist2 = sequence_data2[['wrist:X', 'wrist:Y', 'wrist:Z']].values
    thumb_tip2 = sequence_data2[['ThumbTip:X', 'ThumbTip:Y', 'ThumbTip:Z']].values

    # Compute global min and max for both sequences to fix axis limits
    all_points1 = np.vstack([shoulder1, elbow1, wrist1, thumb_tip1])
    all_points2 = np.vstack([shoulder2, elbow2, wrist2, thumb_tip2])
    x_limits = (min(np.nanmin(all_points1[:, 0]), np.nanmin(all_points2[:, 0])), 
                max(np.nanmax(all_points1[:, 0]), np.nanmax(all_points2[:, 0])))
    y_limits = (min(np.nanmin(all_points1[:, 1]), np.nanmin(all_points2[:, 1])), 
                max(np.nanmax(all_points1[:, 1]), np.nanmax(all_points2[:, 1])))
    z_limits = (min(np.nanmin(all_points1[:, 2]), np.nanmin(all_points2[:, 2])), 
                max(np.nanmax(all_points1[:, 2]), np.nanmax(all_points2[:, 2])))

    # Setting up two subplots for the 3D plots
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

    # Initialize lines to connect the joints for both sequences
    # Sequence 1
    line_shoulder_elbow1, = ax1.plot([], [], [], 'bo-', label="Upper Arm (Shoulder to Elbow)", lw=2)
    line_elbow_wrist1, = ax1.plot([], [], [], 'go-', label="Forearm (Elbow to Wrist)", lw=2)
    line_wrist_thumb1, = ax1.plot([], [], [], 'ro-', label="Hand (Wrist to Thumb)", lw=2)
    
    # Dashed line to trace thumb path (sequence 1)
    thumb_trace1, = ax1.plot([], [], [], 'r--', lw=1, label="Thumb Path (dashed)")

    # Sequence 2
    line_shoulder_elbow2, = ax2.plot([], [], [], 'bo-', label="Upper Arm (Shoulder to Elbow)", lw=2)
    line_elbow_wrist2, = ax2.plot([], [], [], 'go-', label="Forearm (Elbow to Wrist)", lw=2)
    line_wrist_thumb2, = ax2.plot([], [], [], 'ro-', label="Hand (Wrist to Thumb)", lw=2)

    # Dashed line to trace thumb path (sequence 2)
    thumb_trace2, = ax2.plot([], [], [], 'r--', lw=1, label="Thumb Path (dashed)")

    # Set axis labels and limits for both plots
    for ax in [ax1, ax2]:
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper right')

    # Initialize the plot function
    def init():
        # Clear both sequences' lines
        line_shoulder_elbow1.set_data([], [])
        line_shoulder_elbow1.set_3d_properties([])
        line_elbow_wrist1.set_data([], [])
        line_elbow_wrist1.set_3d_properties([])
        line_wrist_thumb1.set_data([], [])
        line_wrist_thumb1.set_3d_properties([])

        thumb_trace1.set_data([], [])
        thumb_trace1.set_3d_properties([])

        line_shoulder_elbow2.set_data([], [])
        line_shoulder_elbow2.set_3d_properties([])
        line_elbow_wrist2.set_data([], [])
        line_elbow_wrist2.set_3d_properties([])
        line_wrist_thumb2.set_data([], [])
        line_wrist_thumb2.set_3d_properties([])

        thumb_trace2.set_data([], [])
        thumb_trace2.set_3d_properties([])

        return line_shoulder_elbow1, line_elbow_wrist1, line_wrist_thumb1, thumb_trace1, \
               line_shoulder_elbow2, line_elbow_wrist2, line_wrist_thumb2, thumb_trace2

    # Update the plot for each frame
    def update(frame):
        # Sequence 1
        line_shoulder_elbow1.set_data([shoulder1[frame, 0], elbow1[frame, 0]], 
                                     [shoulder1[frame, 1], elbow1[frame, 1]])
        line_shoulder_elbow1.set_3d_properties([shoulder1[frame, 2], elbow1[frame, 2]])
        
        line_elbow_wrist1.set_data([elbow1[frame, 0], wrist1[frame, 0]], 
                                  [elbow1[frame, 1], wrist1[frame, 1]])
        line_elbow_wrist1.set_3d_properties([elbow1[frame, 2], wrist1[frame, 2]])
        
        line_wrist_thumb1.set_data([wrist1[frame, 0], thumb_tip1[frame, 0]], 
                                  [wrist1[frame, 1], thumb_tip1[frame, 1]])
        line_wrist_thumb1.set_3d_properties([wrist1[frame, 2], thumb_tip1[frame, 2]])

        # Update thumb trace (sequence 1)
        thumb_trace1.set_data(thumb_tip1[:frame+1, 0], thumb_tip1[:frame+1, 1])
        thumb_trace1.set_3d_properties(thumb_tip1[:frame+1, 2])

        # Sequence 2
        line_shoulder_elbow2.set_data([shoulder2[frame, 0], elbow2[frame, 0]], 
                                     [shoulder2[frame, 1], elbow2[frame, 1]])
        line_shoulder_elbow2.set_3d_properties([shoulder2[frame, 2], elbow2[frame, 2]])
        
        line_elbow_wrist2.set_data([elbow2[frame, 0], wrist2[frame, 0]], 
                                  [elbow2[frame, 1], wrist2[frame, 1]])
        line_elbow_wrist2.set_3d_properties([elbow2[frame, 2], wrist2[frame, 2]])
        
        line_wrist_thumb2.set_data([wrist2[frame, 0], thumb_tip2[frame, 0]], 
                                  [wrist2[frame, 1], thumb_tip2[frame, 1]])
        line_wrist_thumb2.set_3d_properties([wrist2[frame, 2], thumb_tip2[frame, 2]])

        # Update thumb trace (sequence 2)
        thumb_trace2.set_data(thumb_tip2[:frame+1, 0], thumb_tip2[:frame+1, 1])
        thumb_trace2.set_3d_properties(thumb_tip2[:frame+1, 2])

        return line_shoulder_elbow1, line_elbow_wrist1, line_wrist_thumb1, thumb_trace1, \
               line_shoulder_elbow2, line_elbow_wrist2, line_wrist_thumb2, thumb_trace2

    # Create the animation (both animations will be synchronized)
    frames_count = min(len(sequence_data1), len(sequence_data2))
    anim = FuncAnimation(fig, update, frames=frames_count, init_func=init, 
                         interval=animation_speed, blit=True)

    # Show the animation
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example dataset structure
    TRAJECTORIES_PATH_CLEANED = "data/processed/tables/Cleaned_Trajectories.csv"
    TRAJECTORIES_PATH_RAW = "data/raw/tables/Trajectories.csv"

    seq = 10

    df_cleaned = pd.read_csv(TRAJECTORIES_PATH_CLEANED)
    df_raw = pd.read_csv(TRAJECTORIES_PATH_RAW)
    df_interpolated = interpolate_missing_data(df_raw)
    df_cleaned = df_cleaned[df_cleaned["Sequence"] == seq]
    df_raw = df_interpolated[df_interpolated["Sequence"] == seq]

    # Visualize arm movement for sequence 1
    visualize_two_sequences(df_cleaned, seq, df_raw, seq, animation_speed=(1000/200))
