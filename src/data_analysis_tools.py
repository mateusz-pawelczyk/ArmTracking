import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_joint_speeds(df, sequence_idx, max_frames=None):
    """
    Plots the speed of various joints over time for a specified sequence.

    Parameters:
    df (pd.DataFrame): DataFrame containing the trajectory data with columns for each joint's X, Y, Z positions.
    sequence_idx (int): The sequence index to filter the data and visualize.
    max_frames (int, optional): Maximum number of frames to display for clarity. If None, it will display all frames.

    Returns:
    None: Displays the plot.
    """
    joint_names = set([col.split(':')[0] for col in df.columns if ':' in col])

    filtered_data = df[df["Sequence"] == sequence_idx].copy()

    if max_frames is not None:
        filtered_data = filtered_data.iloc[:max_frames]

    for joint in joint_names:
        x_col = f'{joint}:X'
        y_col = f'{joint}:Y'
        z_col = f'{joint}:Z'
        
        if all(col in filtered_data.columns for col in [x_col, y_col, z_col]):
            filtered_data[f'{joint}_speed'] = np.sqrt(filtered_data[x_col].diff()**2 + 
                                                      filtered_data[y_col].diff()**2 + 
                                                      filtered_data[z_col].diff()**2) / (1 / 200)

    plt.figure(figsize=(12, 8))
    
    for joint in joint_names:
        speed_col = f'{joint}_speed'
        if speed_col in filtered_data.columns:
            plt.plot(filtered_data.index, filtered_data[speed_col], label=f'{joint} Speed')
    
    plt.title(f'Speed of Various Joints Over Time (Sequence {sequence_idx})')
    plt.xlabel('Frame')
    plt.ylabel('Speed (mm/s)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_missing_values(df, sequence_idx, max_frames=None):
    """
    Plots the 1D reduced positional data (Euclidean distance from initial position) for shoulder, elbow, wrist, 
    and thumb joints over time. NaN values are interpolated before calculating Euclidean distance and marked in the plot.

    Parameters:
    df (pd.DataFrame): DataFrame containing the trajectory data with columns for each joint's X, Y, Z positions.
    sequence_idx (int): The sequence index to filter the data and visualize.
    max_frames (int, optional): Maximum number of frames to display for clarity. If None, it will display all frames.

    Returns:
    None: Displays the plot.
    """
    joint_names = ['shoulder', 'elbow', 'wrist', 'ThumbTip']
    xyz_columns = {joint: [f'{joint}:X', f'{joint}:Y', f'{joint}:Z'] for joint in joint_names}

    filtered_data = df[df["Sequence"] == sequence_idx].copy()

    if max_frames is not None:
        filtered_data = filtered_data.iloc[:max_frames]

    for joint, cols in xyz_columns.items():
        nan_mask = filtered_data[cols].isna().any(axis=1)
        filtered_data[cols] = filtered_data[cols].interpolate(limit_direction="both")
        filtered_data[f'{joint}_nan_interpolated'] = nan_mask

    for joint, cols in xyz_columns.items():
        x_col, y_col, z_col = cols
        initial_pos = filtered_data.iloc[0][[x_col, y_col, z_col]].values
        filtered_data[f'{joint}_magnitude'] = np.sqrt(
            (filtered_data[x_col] - initial_pos[0]) ** 2 +
            (filtered_data[y_col] - initial_pos[1]) ** 2 +
            (filtered_data[z_col] - initial_pos[2]) ** 2
        )

    plt.figure(figsize=(12, 10))
    for i, joint in enumerate(joint_names):
        plt.subplot(4, 1, i + 1)
        plt.plot(filtered_data.index, filtered_data[f'{joint}_magnitude'], label=f'{joint.capitalize()} Positional Data', color='blue')

        nan_interpolated = filtered_data[filtered_data[f'{joint}_nan_interpolated']]
        plt.scatter(nan_interpolated.index, nan_interpolated[f'{joint}_magnitude'], color='red', label='Interpolated NaN')

        plt.title(f'{joint.capitalize()} Joint Positional Data with Interpolations (Sequence {sequence_idx})')
        plt.ylabel('Magnitude')
        plt.xlabel('Frame')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def visualize_joint_speed(df, joint, coordinate, k=2, sequence=0, sampling_rate=200):
    """
    Visualizes the speed of a specific joint and coordinate, together with the mean and k * std range.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing joint position data.
    joint (str): The joint to analyze (e.g., 'wrist', 'elbow', etc.).
    coordinate (str): The coordinate to analyze ('X', 'Y', 'Z').
    k (int): Range factor.
    sampling_rate (int): The sampling rate in Hz (default is 200 Hz).
    
    Returns:
    None
    """
    # Construct the column name for the specific joint and coordinate
    column_name = f"{joint}:{coordinate}"
    
    # Drop rows with missing data in the selected column
    df_clean = df.dropna(subset=[column_name])
    df_clean = df_clean[df_clean["Sequence"] == sequence]
    
    # Calculate the time step based on the sampling rate (1/sampling_rate)
    time_step = 1 / sampling_rate
    
    # Calculate the speed (difference between consecutive positions divided by time step)
    df_clean['Speed'] = df_clean[column_name].diff() / time_step
    
    # Calculate mean and standard deviation of the speed
    speed_mean = df_clean['Speed'].mean()
    speed_std = df_clean['Speed'].std()
    
    # Calculate upper and lower bounds (mean ± k * std)
    upper_bound = speed_mean + k * speed_std
    lower_bound = speed_mean - k * speed_std
    
    # Plot the speed and bounds
    plt.figure(figsize=(12, 8))
    plt.plot(df_clean.index, df_clean['Speed'], label=f'{joint} {coordinate} Speed', color='blue')
    plt.axhline(y=speed_mean, color='green', linestyle='--', label='Mean Speed')
    plt.axhline(y=upper_bound, color='red', linestyle='--', label=f'Mean + {k}*STD')
    plt.axhline(y=lower_bound, color='red', linestyle='--', label=f'Mean - {k}*STD')
    
    # Add labels and title
    plt.title(f'Speed of {joint} {coordinate} over Time')
    plt.xlabel('Frame')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()