import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

import sys
import os


from data_cleaning_tools import *

PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")
TRAJECTORIES_PATH = "data/raw/tables/Trajectories.csv"

df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR,TRAJECTORIES_PATH))

df_interpolated = interpolate_missing_data(df)

df_smoothed = apply_kalman_filter_df(df_interpolated, smoothness_factor=0.5)

# Updated function to mark outliers for every joint separately
def detect_outliers_per_joint(df, df_smoothed, k=2):
    # Columns to process (excluding Frame and Sequence)
    joint_columns = [col for col in df.columns if col not in ['Frame', 'Sequence']]
    
    # DataFrame to store outlier detection results
    df_outliers = df.copy()
    
    # Initialize outlier flag columns for each joint separately
    for joint in joint_columns:
        df_outliers[f'is_outlier_{joint}'] = False
    
    # Group by sequences to avoid large velocity variances at start/end points of sequences
    for seq, group in df.groupby('Sequence'):
        group_smoothed = df_smoothed[df_smoothed['Sequence'] == seq]
        
        # Loop through each joint coordinate (X, Y, Z for each joint)
        for joint in joint_columns:
            velocity_original = np.gradient(group[joint].to_numpy(), edge_order=2) # Velocity for original data
            velocity_smoothed = np.gradient(group_smoothed[joint].to_numpy(), edge_order=2) # Velocity for smoothed data

            # Calculate standard deviation using the smoothed velocity as the mean
            std = np.sqrt(np.sum((velocity_original - velocity_smoothed) ** 2) / (len(velocity_original) - 1))
            
            # Check for outliers
            outliers = (velocity_original < (velocity_smoothed - k * std)) | (velocity_original > (velocity_smoothed + k * std))
            
            # Mark outliers in the dataframe for the specific joint
            df_outliers.loc[group.index, f'is_outlier_{joint}'] = outliers

    return df_outliers

K = 3

df_outliers = detect_outliers_per_joint(df_interpolated, df_smoothed, k=K)

# Updated function to detect clustered outliers for all joints automatically
def detect_clustered_outliers(df, df_outliers, window_width=5):
    # Identify all joints based on the column names (exclude 'Frame' and 'Sequence')
    joints = {col.split(':')[0] for col in df.columns if col not in ['Frame', 'Sequence']}
    
    # Loop through each joint and detect clustered outliers for X, Y, and Z axes
    for joint in joints:
        # Joint columns (X, Y, Z)
        joint_axes = [f'{joint}:X', f'{joint}:Y', f'{joint}:Z']
        
        # Process each sequence separately
        for seq, group in df.groupby('Sequence'):
            group_outliers = df_outliers[df_outliers['Sequence'] == seq]
            
            # Loop through each axis (X, Y, Z)
            for axis in joint_axes:
                # Compute velocity for the specific axis in this sequence
                velocity = np.gradient(group[axis].to_numpy(), edge_order=2)
                
                # Identify the index and sign of the already detected outliers (based on velocity spikes)
                outlier_indices = np.where(group_outliers[f'is_outlier_{axis}'])[0]
                signs = np.sign(velocity[outlier_indices])
                
                # Initialize index vector X and sign vector S
                X = outlier_indices
                S = signs

                # Identify isolated and clustered outliers
                i = 0
                while i < len(X) - 1:
                    if S[i] != S[i + 1]:  # Check for opposite signs (spike out, spike back)
                        if (X[i + 1] - X[i] == 1):  # Isolated outlier (consecutive frames)
                            # Mark as isolated outlier (already marked by velocity detection)
                            pass
                        elif (X[i + 1] - X[i]) <= window_width:  # Clustered outliers (within window width)
                            # Mark all frames between X[i] and X[i+1] as clustered outliers
                            df_outliers.loc[group.index[X[i]:X[i + 1]], f'is_outlier_{axis}'] = True
                        i += 2  # Move to the next pair
                    else:
                        i += 1

    return df_outliers

WINDOW_WIDTH = 100

df_outliers = detect_clustered_outliers(df_interpolated, df_outliers, window_width=WINDOW_WIDTH)

# set outliers to NaN
df_cleaned = df_interpolated.copy()
df_cleaned[df_outliers.filter(like='is_outlier').any(axis=1)] = np.nan

# interpolate again
df_cleaned = interpolate_missing_data(df_cleaned)


# smooth data
df_smoothed = apply_kalman_filter_df(df_cleaned, smoothness_factor=0.5)

# save cleaned data
CLEANED_TRAJECTORIES_PATH = "data/processed/tables/Cleaned_Trajectories.csv"

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(CLEANED_TRAJECTORIES_PATH), exist_ok=True)

df_cleaned.to_csv(os.path.join(PROJECT_ROOT_DIR,CLEANED_TRAJECTORIES_PATH), index=False)