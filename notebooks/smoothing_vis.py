import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error

def apply_kalman_filter(df, smoothness_factor=1):
    """
    Applies Kalman Filter to each sequence in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing joint coordinates (X, Y, Z) for each joint.
        smoothness_factor (float): Controls the smoothness of the filter (higher values mean smoother data).
    
    Returns:
        pd.DataFrame: The DataFrame with smoothed coordinates.
    """
    
    def kalman_smooth_single_sequence(df_sequence):
        # Kalman Filter initialization
        initial_state_mean = df_sequence.iloc[0, 1:13].fillna(0).values  # Start with the first row's values
        observation_covariance = np.eye(12) * smoothness_factor  # Measurement noise covariance
        transition_covariance = np.eye(12) * 0.01  # Process noise covariance, small process noise
        transition_matrix = np.eye(12)  # We assume a simple constant velocity model

        # Create the Kalman Filter
        kf = KalmanFilter(
            initial_state_mean=initial_state_mean,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix,
        )
        
        # Apply Kalman filter on the full sequence (ignoring the frame number and sequence columns)
        observations = df_sequence.iloc[:, 1:13].values  # Only take the joint coordinates
        smoothed_state_means, _ = kf.smooth(observations)
        
        # Replace the original joint columns with the smoothed ones
        df_sequence.iloc[:, 1:13] = smoothed_state_means
        
        return df_sequence
    
    # Apply the Kalman filter sequence by sequence
    df_grouped = df.groupby('Sequence', group_keys=False)
    smoothed_df = df_grouped.apply(lambda group: kalman_smooth_single_sequence(group), include_groups=False).reset_index(drop=True)
    
    return smoothed_df


def precompute_smoothing(df, joint, sequence, max_smoothness, num_steps, sampling_rate=200):
    """
    Precomputes smoothed velocities for different smoothing factors for X, Y, and Z coordinates.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing joint position data.
        joint (str): The joint to analyze (e.g., 'wrist', 'elbow', etc.).
        sequence (int): The sequence number to filter by.
        max_smoothness (float): The maximum smoothing factor for Kalman Filter.
        num_steps (int): The number of smoothing steps to precompute.
        sampling_rate (int): The sampling rate in Hz (default is 200 Hz).
    
    Returns:
        dict: A dictionary of lists for each coordinate ('X', 'Y', 'Z') with progressively smoothed velocities.
        dict: A dictionary containing the MSE values for each coordinate at every step.
    """
    smoothed_dfs = {'X': [], 'Y': [], 'Z': []}
    mse_values = {'X': [], 'Y': [], 'Z': []}
    time_step = 1 / sampling_rate
    
    for coord in ['X', 'Y', 'Z']:
        df_clean = df[df["Sequence"] == sequence].dropna(subset=[f"{joint}:{coord}"])
        df_clean[f'Original Speed {coord}'] = df_clean[f"{joint}:{coord}"].diff() / time_step

        for step in range(num_steps):
            smoothness_factor = (step / num_steps) * max_smoothness
            df_smoothed = apply_kalman_filter(df_clean.copy(), smoothness_factor=smoothness_factor)
            df_smoothed[f'Smoothed Speed {coord}'] = df_smoothed[f"{joint}:{coord}"].diff() / time_step
            smoothed_dfs[coord].append((df_smoothed, smoothness_factor))
            
            # Calculate MSE between original and smoothed velocities
            mse = mean_squared_error(df_clean[f'Original Speed {coord}'].dropna(), df_smoothed[f'Smoothed Speed {coord}'].dropna())
            mse_values[coord].append(mse)

    return smoothed_dfs, mse_values


def visualize_animation_with_smoothing_and_mse(df, joint, sequence=0, sampling_rate=200, max_smoothness=10, num_frames=100):
    """
    Creates an animation showing the original and progressively smoothed velocity data for X, Y, and Z coordinates,
    and also visualizes the MSE for each coordinate.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing joint position data.
        joint (str): The joint to analyze (e.g., 'wrist', 'elbow', etc.).
        sequence (int): The sequence number to filter by.
        sampling_rate (int): The sampling rate in Hz (default is 200 Hz).
        max_smoothness (int): The maximum value for the Kalman filter smoothness factor.
        num_frames (int): The number of frames in the animation.
    
    Returns:
        None
    """
    # Precompute the smoothed data for each frame and each coordinate (X, Y, Z)
    smoothed_dfs, mse_values = precompute_smoothing(df, joint, sequence, max_smoothness, num_frames, sampling_rate=sampling_rate)
    
    # Initialize plot with 3 subplots for X, Y, and Z coordinates + 1 subplot for MSE
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    # Titles for the axes
    coords = ['X', 'Y', 'Z']
    ax_titles = [f"Velocity of {joint}:{coord} - Original vs Smoothing" for coord in coords]

    # Plot for each coordinate (X, Y, Z)
    lines = []
    for i, coord in enumerate(coords):
        df_clean = df[df["Sequence"] == sequence].dropna(subset=[f"{joint}:{coord}"])
        time_step = 1 / sampling_rate
        df_clean[f'Original Speed {coord}'] = df_clean[f"{joint}:{coord}"].diff() / time_step
        
        # Plot original speed
        axs[i].set_title(ax_titles[i], fontsize=14)
        axs[i].set_ylabel('Speed (mm/s)')
        axs[i].grid(True)
        original_line, = axs[i].plot(df_clean['Frame'], df_clean[f'Original Speed {coord}'], label="Original Speed", color='blue')
        smoothed_line, = axs[i].plot([], [], label="Smoothed Speed", color='orange')
        lines.append(smoothed_line)

    # Add the MSE subplot
    axs[3].set_title("MSE Between Original and Smoothed Velocities", fontsize=14)
    axs[3].set_xlabel('Smoothing Step')
    axs[3].set_ylabel('MSE')
    axs[3].grid(True)

    # Plot the MSE for each coordinate
    mse_lines = []
    for i, coord in enumerate(coords):
        mse_line, = axs[3].plot([], [], label=f'MSE {coord}', linestyle='--')
        mse_lines.append(mse_line)

    plt.xlabel('Frame')
    axs[0].legend()

    # Add a text label to display the smoothing factor
    smoothing_text = axs[0].text(0.05, 0.95, '', transform=axs[0].transAxes, fontsize=12, verticalalignment='top')

    # Function to update the plot at each frame
    def update(frame_num):
        # Update each subplot for X, Y, and Z
        for i, coord in enumerate(coords):
            df_smoothed, smoothness_factor = smoothed_dfs[coord][frame_num]
            
            # Update the smoothed line data for each coordinate
            lines[i].set_data(df_smoothed['Frame'], df_smoothed[f'Smoothed Speed {coord}'])
            
            # Update the MSE line for each coordinate
            mse_lines[i].set_data(range(frame_num + 1), mse_values[coord][:frame_num + 1])
        
        # Dynamically adjust the y-axis limits of the MSE plot
        max_mse = max(max(mse_values['X'][:frame_num + 1]), max(mse_values['Y'][:frame_num + 1]), max(mse_values['Z'][:frame_num + 1]))
        axs[3].set_ylim(0, max_mse * 1.1)  # Set a 10% buffer above the current maximum MSE value
        
        # Update the smoothing factor text
        smoothing_text.set_text(f'Smoothing Factor: {smoothness_factor:.2f}')
        
        return lines + mse_lines + [smoothing_text]

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_frames, blit=False, interval=100)
    
    plt.show()




PROJECT_DIR = "../"
TRAJECTORIES_PATH = "data/raw/tables/Trajectories.csv"

import sys
import os


# Add the absolute path to the parent directory of 'src'
sys.path.append(os.path.abspath(os.path.join('..')))
df = pd.read_csv(TRAJECTORIES_PATH)

def interpolate_missing_data(df, sequence_col='Sequence'):
    """
    Replaces the very first NaN value in each sequence with the mean of the initial coordinates
    for each joint (column), and returns a new DataFrame without altering the original one.

    Parameters:
    df (pd.DataFrame): The input DataFrame with joint positions.
    sequence_col (str): The name of the column that groups the sequences.

    Returns:
    pd.DataFrame: A new DataFrame with NaNs handled.
    """
    # Create a copy of the original DataFrame to avoid altering it
    new_df = df.copy()

    # Step 1: Calculate the mean of the first coordinates for each joint
    initial_values = df.groupby(sequence_col).apply(lambda X: X.iloc[0]).mean()

    # Step 2: Replace only the very first NaN value of each sequence with the mean
    for col in df.columns:
        if col != sequence_col:  # Ignore the 'Sequence' column
            # Find the first value in each sequence
            first_values = df.groupby(sequence_col)[col].head(1)
            # Identify which first values are NaN
            nan_mask = first_values.isna()
            
            # For sequences where the first value is NaN, replace it with the mean value
            new_df.loc[new_df.groupby(sequence_col).head(1)[nan_mask].index, col] = initial_values[col]

    # Step 3: Apply interpolation for other NaNs if needed
    new_df = new_df.groupby("Sequence").apply(lambda x : x.interpolate(method='linear')).reset_index(drop=True)


    return new_df

df_interpolated = interpolate_missing_data(df)

# Example usage:
visualize_animation_with_smoothing_and_mse(df_interpolated, 'elbow', sequence=1, sampling_rate=200, max_smoothness=0.4, num_frames=100)
