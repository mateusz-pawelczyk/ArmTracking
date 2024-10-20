import pandas as pd
import numpy as np
from joblib import Parallel, delayed


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




def apply_kalman_filter_ndarray(observations, smoothness_factor):
    """
    Applies Kalman smoothing to a single sequence of observations.

    Parameters:
        observations (np.ndarray): 2D array of joint coordinates (X, Y, Z) for each frame.
        smoothness_factor (float): Controls the smoothness of the filter.

    Returns:
        np.ndarray: Smoothed sequence.
    """
    from pykalman import KalmanFilter

    _, dimension = observations.shape

    initial_state_mean = observations[0, :].copy()  # First frame as initial state
    observation_covariance = np.eye(dimension) * smoothness_factor
    transition_covariance = np.eye(dimension) * 0.01  # Process noise covariance
    transition_matrix = np.eye(dimension)  # Assuming constant velocity model

    # Create Kalman Filter instance
    kf = KalmanFilter(
        initial_state_mean=initial_state_mean,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix,
    )

    # Apply the Kalman filter
    smoothed_state_means, _ = kf.smooth(observations)

    return smoothed_state_means

def apply_kalman_filter_df(df, smoothness_factor=1, n_jobs=-1):
    """
    Applies Kalman Filter to each sequence in the DataFrame using parallel processing.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing joint coordinates (X, Y, Z) for each joint.
        smoothness_factor (float): Controls the smoothness of the filter.
        n_jobs (int): Number of jobs for parallel processing (-1 means use all available cores).

    Returns:
        pd.DataFrame: The DataFrame with smoothed coordinates.
    """
    columns = [col for col in df.columns if ":" in col]

    # Extract sequence indices and observation matrix
    sequences = df['Sequence'].unique()
    sequence_groups = [df[df['Sequence'] == seq][columns].values for seq in sequences]
    
    # Parallel processing: apply Kalman filter to each sequence
    smoothed_sequences = Parallel(n_jobs=n_jobs)(
        delayed(apply_kalman_filter_ndarray)(obs, smoothness_factor) for obs in sequence_groups
    )
    
    # Reconstruct the DataFrame with smoothed data
    smoothed_data = np.vstack(smoothed_sequences)
    
    # Create a new DataFrame with the original columns
    result_df = pd.DataFrame(smoothed_data, columns=columns)
    result_df["Sequence"] = df['Sequence']
    result_df["Frame"] = df['Frame']

    return result_df



# Define the smoothing function for a single sequence
def apply_savgol_filter_ndarray(sequence_data, window_length, polyorder):
        from scipy.signal import savgol_filter

        return savgol_filter(sequence_data, window_length=window_length, polyorder=polyorder, axis=0)

def apply_savgol_filter(df, window_length=11, polyorder=2, n_jobs=-1):
    """
    Applies Savitzky-Golay filter to each sequence in the DataFrame using parallel processing.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing joint coordinates (X, Y, Z) for each joint.
        window_length (int): The length of the filter window (must be odd).
        polyorder (int): The order of the polynomial used to fit the samples.
        n_jobs (int): The number of CPU cores to use for parallel processing (-1 to use all available).

    Returns:
        pd.DataFrame: The DataFrame with smoothed coordinates.
    """

    # Extract the joint columns (assuming joint columns are all except 'Frame' and 'Sequence')
    joint_columns = df.columns[1:]  # Assuming 'Sequence' is in the first column

    # Extract sequence groups using NumPy for efficiency
    sequences = df['Sequence'].unique()
    sequence_groups = [df[df['Sequence'] == seq][joint_columns].values for seq in sequences]

    

    # Parallel processing of each sequence
    smoothed_sequences = Parallel(n_jobs=n_jobs)(
        delayed(apply_savgol_filter_ndarray)(seq, window_length, polyorder) for seq in sequence_groups
    )

    # Combine smoothed data back into a DataFrame
    smoothed_data = np.vstack(smoothed_sequences)
    result_df = pd.DataFrame(smoothed_data, columns=joint_columns)
    result_df["Sequence"] = df["Sequence"]
    result_df["Frame"] = df["Frame"]
    result_df = result_df[df.columns]

    return result_df





def apply_woltring_spline_ndarray(observations, smoothness_factor):
    """
    Applies spline smoothing (similar to Woltring's method) to a single sequence of observations.

    Parameters:
        observations (np.ndarray): 2D array of joint coordinates (X, Y, Z) for each frame.
        smoothness_factor (float): Smoothing factor (controls the smoothness of the spline).

    Returns:
        np.ndarray: Smoothed sequence.
    """
    from scipy.interpolate import UnivariateSpline

    # Number of time frames
    num_frames = observations.shape[0]

    # Generate a time array (assuming equally spaced frames)
    time = np.arange(num_frames)

    # Apply smoothing spline to each dimension (X, Y, Z) independently
    smoothed_sequence = np.zeros_like(observations)
    for i in range(observations.shape[1]):
        # Univariate spline with smoothing
        spline = UnivariateSpline(time, observations[:, i], s=smoothness_factor)
        smoothed_sequence[:, i] = spline(time)
    
    return smoothed_sequence


def apply_woltring_spline(df, smoothness_factor=10, n_jobs=-1):
    """
    Applies a spline filter (similar to Woltring's spline) to each sequence in the DataFrame using parallel processing.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing joint coordinates (X, Y, Z) for each joint.
        smoothness_factor (float): Smoothing parameter for the spline filter.
        n_jobs (int): Number of jobs for parallel processing (-1 means use all available cores).

    Returns:
        pd.DataFrame: The DataFrame with smoothed coordinates.
    """
    # Extract sequence indices and observation matrix
    sequences = df['Sequence'].unique()
    joint_columns = df.columns[1:13]  # Assuming joint columns are from index 1 to 12 (excluding 'Sequence' and 'Frame')
    sequence_groups = [df[df['Sequence'] == seq][joint_columns].values for seq in sequences]
    
    # Parallel processing: apply spline filter to each sequence
    smoothed_sequences = Parallel(n_jobs=n_jobs)(
        delayed(apply_woltring_spline_ndarray)(obs, smoothness_factor) for obs in sequence_groups
    )
    
    # Reconstruct the DataFrame with smoothed data
    smoothed_data = np.vstack(smoothed_sequences)
    
    # Create a new DataFrame with the original columns
    result_df = pd.DataFrame(smoothed_data, columns=joint_columns)
    result_df["Sequence"] = df['Sequence']
    result_df["Frame"] = df['Frame']
    result_df = result_df[df.columns]

    return result_df


def apply_ema_smoothing_ndarray(observations, alpha):
    """
    Applies Exponential Moving Average (EMA) smoothing to a single sequence of observations.

    Parameters:
        observations (np.ndarray): 2D array of joint coordinates (X, Y, Z) for each frame.
        alpha (float): Smoothing factor, controls how much weight is given to recent observations.
                       (0 < alpha <= 1, with higher values giving more weight to recent frames)

    Returns:
        np.ndarray: Smoothed sequence.
    """
    smoothed_sequence = np.zeros_like(observations)
    
    # Initialize the first value as the first observation
    smoothed_sequence[0] = observations[0]

    # Apply the EMA formula to each joint coordinate (X, Y, Z)
    for t in range(1, observations.shape[0]):
        smoothed_sequence[t] = alpha * observations[t] + (1 - alpha) * smoothed_sequence[t - 1]
    
    return smoothed_sequence


def apply_ema_smoothing(df, alpha=0.3, n_jobs=-1):
    """
    Applies Exponential Moving Average (EMA) smoothing to each sequence in the DataFrame using parallel processing.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing joint coordinates (X, Y, Z) for each joint.
        alpha (float): Smoothing factor (0 < alpha <= 1).
        n_jobs (int): Number of jobs for parallel processing (-1 means use all available cores).

    Returns:
        pd.DataFrame: The DataFrame with smoothed coordinates.
    """
    # Extract sequence indices and observation matrix
    sequences = df['Sequence'].unique()
    joint_columns = df.columns[1:13]  # Assuming joint columns are from index 1 to 12 (excluding 'Sequence' and 'Frame')
    sequence_groups = [df[df['Sequence'] == seq][joint_columns].values for seq in sequences]
    
    # Parallel processing: apply EMA smoothing to each sequence
    smoothed_sequences = Parallel(n_jobs=n_jobs)(
        delayed(apply_ema_smoothing_ndarray)(obs, alpha) for obs in sequence_groups
    )
    
    # Reconstruct the DataFrame with smoothed data
    smoothed_data = np.vstack(smoothed_sequences)
    
    # Create a new DataFrame with the original columns
    result_df = pd.DataFrame(smoothed_data, columns=joint_columns)
    result_df["Sequence"] = df['Sequence']
    result_df["Frame"] = df['Frame']
    result_df = result_df[df.columns]

    return result_df