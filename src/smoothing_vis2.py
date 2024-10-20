import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from joblib import Parallel, delayed
from pykalman import KalmanFilter
from sklearn.metrics import root_mean_squared_error as mse

from data_cleaning_tools import *

# ========== CONFIGURATION ========== #

DEFAULT_SAMPLING_RATE = 200  # Default sampling rate in Hz
TRAJECTORIES_PATH = "data/raw/tables/Trajectories.csv"

# ========== UTILITY FUNCTIONS ========== #

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load trajectory data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the trajectory data.
    """
    return pd.read_csv(filepath)

# ========== SMOOTHING FUNCTIONS ================= #

def apply_kalman_smoothing(data: np.ndarray, smoothness_factors: list, n_jobs: int = -1) -> list:
    """
    Apply Kalman smoothing in parallel for multiple smoothness factors.

    Args:
        data (np.ndarray): Observation data (positions).
        smoothness_factors (list): List of smoothness factors to use.
        n_jobs (int): Number of parallel jobs (-1 uses all cores).

    Returns:
        list: List of smoothed arrays for each smoothness factor.
    """

    return Parallel(n_jobs=n_jobs)(
        delayed(apply_kalman_filter_ndarray)(data, sf) for sf in smoothness_factors
    )

def apply_ema_smoothing(data: np.ndarray, alphas: list, n_jobs: int = -1) -> list:
    """
    Apply Kalman smoothing in parallel for multiple smoothness factors.

    Args:
        data (np.ndarray): Observation data (positions).
        smoothness_factors (list): List of smoothness factors to use.
        n_jobs (int): Number of parallel jobs (-1 uses all cores).

    Returns:
        list: List of smoothed arrays for each smoothness factor.
    """

    return Parallel(n_jobs=n_jobs)(
        delayed(apply_ema_smoothing_ndarray)(data, alpha) for alpha in alphas
    )

def apply_savgol_filter(data: np.ndarray, window_lengths: list, polyorder: int = 3, n_jobs: int = -1) -> list:
    """
    Apply Kalman smoothing in parallel for multiple smoothness factors.

    Args:
        data (np.ndarray): Observation data (positions).
        smoothness_factors (list): List of smoothness factors to use.
        n_jobs (int): Number of parallel jobs (-1 uses all cores).

    Returns:
        list: List of smoothed arrays for each smoothness factor.
    """
    window_lengths = [wl for wl in window_lengths if wl < len(data) and wl > 3]
    polyorder = min(polyorder, min(window_lengths) - 1)

    return Parallel(n_jobs=n_jobs)(
        delayed(apply_savgol_filter_ndarray)(data, window_length, min(polyorder, window_length-1)) for window_length in window_lengths
    )

def apply_woltring_spline(data: np.ndarray, smoothness_factors: list, n_jobs: int = -1) -> list:
    """
    Apply Kalman smoothing in parallel for multiple smoothness factors.

    Args:
        data (np.ndarray): Observation data (positions).
        smoothness_factors (list): List of smoothness factors to use.
        n_jobs (int): Number of parallel jobs (-1 uses all cores).

    Returns:
        list: List of smoothed arrays for each smoothness factor.
    """

    return Parallel(n_jobs=n_jobs)(
        delayed(apply_woltring_spline_ndarray)(data, sf) for sf in smoothness_factors
    )

# ========== VELOCITY AND ERROR METRICS ========== #

def calculate_velocity(positions: np.ndarray, sampling_rate: int = DEFAULT_SAMPLING_RATE) -> np.ndarray:
    """
    Calculate velocity based on positional data.

    Args:
        positions (np.ndarray): Array of joint positions.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        np.ndarray: Array of velocities for each frame.
    """
    velocity = np.linalg.norm(np.diff(positions, axis=0), axis=1) * sampling_rate
    return np.concatenate([velocity, [0]])  # Append zero for final frame consistency

def calculate_mse(true_velocities: np.ndarray, smoothed_velocities: np.ndarray) -> float:
    """
    Compute Mean Squared Error between original and smoothed velocities.

    Args:
        true_velocities (np.ndarray): Original velocity data.
        smoothed_velocities (np.ndarray): Smoothed velocity data.

    Returns:
        float: MSE value.
    """
    return mse(true_velocities, smoothed_velocities)

# ========== VISUALIZATION AND PLOTTING ========== #

class SmoothingVisualizer:
    def __init__(self, df: pd.DataFrame, sampling_rate: int = DEFAULT_SAMPLING_RATE):
        """
        Initialize the SmoothingVisualizer for creating animations of the smoothing process.

        Args:
            df (pd.DataFrame): DataFrame containing trajectory data.
            sampling_rate (int): The sampling rate of the data.
        """
        self.df = df
        self.sampling_rate = sampling_rate
        self.paused = False  # Flag to check if animation is paused

    def visualize_smoothing(self, sequence_id: int, joint: str, method: str = "kalman", frames: int = 100):
        """
        Generate and display an animation comparing original and smoothed velocities,
        along with a dynamic MSE plot.

        Args:
            sequence_id (int): Sequence index to visualize.
            joint (str): Joint name (e.g., 'wrist', 'elbow').
            method (str): Smoothing method to use (e.g., 'kalman').
            frames (int): Number of frames for the animation.
        """
        sequence_data = self._extract_sequence_data(sequence_id, joint)
        if method == "kalman":
            func = apply_kalman_smoothing
            param_range = np.linspace(0, 2, frames)
        elif method == "woltring":
            func = apply_woltring_spline
            param_range = np.linspace(0, 50, frames)
        elif method == "ema":
            func = apply_ema_smoothing
            param_range = np.linspace(1, 0, frames)
        elif method == "savgol":
            func = apply_savgol_filter
            param_range = np.round(np.linspace(3, 100, frames)).astype(int)

        velocities = self._compute_velocities(sequence_data, func, param_range)
        self._create_animation(sequence_data, velocities, joint, param_range)

    def _extract_sequence_data(self, sequence_id: int, joint: str) -> np.ndarray:
        """
        Extract joint position data for a specific sequence and joint.

        Args:
            sequence_id (int): Sequence index to extract.
            joint (str): Joint name (e.g., 'wrist').

        Returns:
            np.ndarray: Joint position data (X, Y, Z).
        """
        joint_columns = [f"{joint}:X", f"{joint}:Y", f"{joint}:Z"]
        return self.df[self.df['Sequence'] == sequence_id][joint_columns].dropna().values

    def _compute_velocities(self, original_data: np.ndarray, func, param_range) -> dict:
        """
        Compute velocities for original and smoothed data, and calculate MSE.

        Args:
            original_data (np.ndarray): Original joint position data.
            func: Smoothing function to apply.
            param_range: Range of smoothing factors.

        Returns:
            dict: Dictionary containing original velocities, smoothed velocities, and MSE values.
        """
        original_velocity = calculate_velocity(original_data, self.sampling_rate)
        smoothed_data = func(original_data, param_range)
        smoothed_velocities = [calculate_velocity(data, self.sampling_rate) for data in smoothed_data]
        mse_values = [calculate_mse(original_velocity, sv) for sv in smoothed_velocities]
        return {'original': original_velocity, 'smoothed': smoothed_velocities, 'mse': mse_values}

    def _create_animation(self, sequence_data: np.ndarray, velocities: dict, joint: str, smoothness_factors: list):
        """
        Create the smoothing animation comparing original and smoothed velocities,
        along with a dynamic MSE plot.

        Args:
            sequence_data (np.ndarray): Original joint position data.
            velocities (dict): Dictionary containing velocities and MSE.
            joint (str): Joint name (e.g., 'wrist').
            smoothness_factors (list): List of smoothness factors for smoothing.
        """
        original_velocity = velocities['original']
        smoothed_velocities = velocities['smoothed']
        mse_values = velocities['mse']

        # Create subplots: 1 for the velocity comparison, 1 for the MSE plot
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [0.3, 1, 0.5]})
        text_ax = axs[0]
        text_ax.axis('off')
        text_display = text_ax.text(0.5, 0.5, '', transform=text_ax.transAxes, fontsize=14, ha='center', va='center')

        # Precompute axis limits to avoid dynamic updates
        velocity_min = np.min([original_velocity] + smoothed_velocities) * 1.1
        velocity_max = np.max([original_velocity] + smoothed_velocities) * 1.1
        mse_min = 0
        mse_max = max(mse_values) * 1.1

        # Velocity plot
        original_line, = axs[1].plot([], [], label='Original Velocity', lw=2)
        smoothed_line, = axs[1].plot([], [], label='Smoothed Velocity', lw=2)
        axs[1].set_title(f'{joint.capitalize()} Velocity', fontsize=12)
        axs[1].set_xlim(0, len(sequence_data))
        axs[1].set_ylim(velocity_min, velocity_max)
        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel('Velocity (Euclidean)')
        axs[1].legend()

        # MSE plot
        mse_line, = axs[2].plot([], [], label='MSE', color='red', lw=2)
        axs[2].set_xlim(smoothness_factors[0], smoothness_factors[-1])
        axs[2].set_ylim(mse_min, mse_max)
        axs[2].set_xlabel('Smoothing Factor')
        axs[2].set_ylabel('MSE')
        axs[2].set_title('Mean Squared Error Over Smoothing Factors')
        axs[2].legend()

        # Data for the x-axis of the MSE plot
        mse_xdata = []
        mse_ydata = []

        def update(frame):
            if self.paused:
                return  # Skip updating if paused
            
            smoothed_velocity = smoothed_velocities[frame]
            mse_value = mse_values[frame]

            # Update velocity lines
            original_line.set_data(np.arange(len(original_velocity)), original_velocity)
            smoothed_line.set_data(np.arange(len(smoothed_velocity)), smoothed_velocity)

            # Update MSE data
            mse_xdata.append(smoothness_factors[frame])
            mse_ydata.append(mse_value)
            mse_line.set_data(mse_xdata, mse_ydata)

            # Update smoothing factor and MSE text
            text_display.set_text(f'Smoothing Factor: {smoothness_factors[frame]:.2f} | MSE: {mse_value:.5f}')
            
            return [text_display, original_line, smoothed_line, mse_line]

        # Reset the MSE plot data on each repeat
        def init():
            mse_xdata.clear()
            mse_ydata.clear()
            return [text_display, original_line, smoothed_line, mse_line]

        # Event handler to pause/resume animation
        def on_click(event):
            if self.paused:
                ani.event_source.start()  # Restart the animation
            else:
                ani.event_source.stop()  # Pause the animation
            self.paused = not self.paused  # Toggle paused state

        # Connect the click event to the figure
        fig.canvas.mpl_connect('button_press_event', on_click)

        # Create the animation with blit=True
        ani = FuncAnimation(fig, update, frames=len(smoothness_factors), blit=True, interval=100, repeat=True, init_func=init)
        plt.tight_layout()
        plt.show()


# ========== MAIN EXECUTION ========== #

def main():
    df = load_data(TRAJECTORIES_PATH)
    df_interpolated = interpolate_missing_data(df)

    # Initialize visualizer
    visualizer = SmoothingVisualizer(df_interpolated)
    
    # Define smoothness factors
    smoothness_factors = np.linspace(0, 0.5, 100)
    
    # Visualize smoothing for a specific sequence and joint
    visualizer.visualize_smoothing(sequence_id=10, joint="wrist", method="kalman", frames=100)

if __name__ == "__main__":
    main()
