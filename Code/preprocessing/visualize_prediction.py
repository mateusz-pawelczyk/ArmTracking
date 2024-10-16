import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.nn.utils.rnn import pad_sequence
from Model import MotionModel  # Ensure this imports your model definition

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and set it to evaluation mode
def load_model(input_dim=12, hidden_dim=128, output_dim=12, num_layers=2):
    model = MotionModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    except FileNotFoundError:
        raise RuntimeError("Model weights file not found.")
    
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Load normalization parameters with error handling
def load_normalization_params():
    try:
        X_mean = torch.load('X_mean.pth').to(device)
        X_std = torch.load('X_std.pth').to(device)
        Y_mean = torch.load('Y_mean.pth').to(device)
        Y_std = torch.load('Y_std.pth').to(device)
    except FileNotFoundError as e:
        raise RuntimeError("Normalization files not found. Ensure the files exist.") from e
    return X_mean, X_std, Y_mean, Y_std

X_mean, X_std, Y_mean, Y_std = load_normalization_params()

# Load and preprocess the data
def load_processed_data(file_path='csv_new/processed_Trajectories.csv'):
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns, but retain 'Sequence' for grouping
    df_cleaned = df.drop(columns=["Sequence"])
    df_cleaned["Sequence"] = df["Sequence"]
    
    return df_cleaned

full_data = load_processed_data()

# Prepare sequences by padding them to the same length
def prepare_sequences(df):
    grouped = df.groupby('Sequence')
    
    X_seqs, Y_seqs, lengths = [], [], []
    for _, group in grouped:
        X_seq = torch.tensor(group.drop(columns=['Frame', 'Sequence']).values[:-1], dtype=torch.float32)
        Y_seq = torch.tensor(group[['shoulder:X', 'shoulder:Y', 'shoulder:Z',
                                    'elbow:X', 'elbow:Y', 'elbow:Z',
                                    'wrist:X', 'wrist:Y', 'wrist:Z',
                                    'ThumbTip:X', 'ThumbTip:Y', 'ThumbTip:Z']].values[1:], dtype=torch.float32)
        
        if len(X_seq) > 0 and len(Y_seq) > 0:
            X_seqs.append(X_seq)
            Y_seqs.append(Y_seq)
            lengths.append(len(X_seq))

    # Pad sequences to the same length and return as tensor
    X_padded = pad_sequence(X_seqs, batch_first=True, padding_value=0.0)
    Y_padded = pad_sequence(Y_seqs, batch_first=True, padding_value=0.0)
    lengths_tensor = torch.tensor(lengths)
    
    return X_padded, Y_padded, lengths_tensor

X_padded, Y_padded, lengths_tensor = prepare_sequences(full_data)
print(f"X_padded shape: {X_padded.shape}, Y_padded shape: {Y_padded.shape}, Lengths shape: {lengths_tensor.shape}")

# Move padded sequences to the device
X_padded = X_padded.to(device)
Y_padded = Y_padded.to(device)
lengths_tensor = lengths_tensor.to(device)

def iterative_prediction(model, initial_input, prediction_length, X_mean, X_std, Y_mean, Y_std, device):
    input_t = initial_input  # shape (input_dim), unnormalized
    predictions = []

    for t in range(prediction_length):
        # Normalize the input
        input_normalized = (input_t - X_mean) / X_std  # shape (input_dim)

        # Reshape to (batch_size, seq_length, input_dim)
        input_normalized = input_normalized.unsqueeze(0).unsqueeze(1)  # shape (1, 1, input_dim)

        # Convert lengths to a tensor and move to device
        lengths_tensor = torch.tensor([1], dtype=torch.long).to(device)  # Since seq_length = 1

        with torch.no_grad():
            output, _ = model(input_normalized, lengths_tensor)

        # output: shape (batch_size, seq_length, output_dim)
        output_t = output[0, -1, :]  # shape (output_dim)

        # Denormalize the output
        output_denormalized = denormalize(output_t, Y_mean, Y_std)  # shape (output_dim)

        # Append to predictions
        predictions.append(output_denormalized.unsqueeze(0))  # shape (1, output_dim)

        # Set input_t for next time step
        input_t = output_denormalized

    # Stack predictions
    predictions = torch.cat(predictions, dim=0)  # shape (prediction_length, output_dim)

    return predictions  # shape (prediction_length, output_dim)


# Denormalize the data
def denormalize(data, mean, std):
    return data * std + mean

# Extract coordinates for animation
def extract_coordinates(data):
    return data[:, 0::3], data[:, 1::3], data[:, 2::3]

# Calculate wrist velocity
def calculate_wrist_velocity(xs, ys, zs, frame_interval=1):
    # Wrist is the third joint in the data (index 2)
    wrist_index = 2  # zero-based index
    positions = np.stack((xs[:, wrist_index], ys[:, wrist_index], zs[:, wrist_index]), axis=1)
    # Calculate velocity magnitude between consecutive frames
    velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1) / frame_interval
    # Append a zero at the beginning to match the number of frames
    velocities = np.insert(velocities, 0, 0)
    return velocities

# Create 3D animation for predicted and true joint movements
def create_animation(pred_xs, pred_ys, pred_zs, true_xs, true_ys, true_zs, pred_wrist_vel, true_wrist_vel):
    fig = plt.figure(figsize=(14, 6))
    
    ax_pred = fig.add_subplot(121, projection='3d')
    ax_real = fig.add_subplot(122, projection='3d')
    
    # Setup axis limits for both plots
    def set_axes_limits(ax, xs, ys, zs):
        ax.set_xlim([xs.min(), xs.max()])
        ax.set_ylim([ys.min(), ys.max()])
        ax.set_zlim([zs.min(), zs.max()])
    
    set_axes_limits(ax_pred, pred_xs, pred_ys, pred_zs)
    set_axes_limits(ax_real, true_xs, true_ys, true_zs)

    ax_pred.set_title('Predicted Movement')
    ax_real.set_title('Actual Movement')

    graph_pred = ax_pred.scatter(pred_xs[0], pred_ys[0], pred_zs[0], c="blue", marker='o')
    graph_real = ax_real.scatter(true_xs[0], true_ys[0], true_zs[0], c="red", marker='o')
    
    # Initialize text annotations for frame count and wrist velocity
    frame_text_pred = ax_pred.text2D(0.05, 0.95, '', transform=ax_pred.transAxes)
    wrist_vel_text_pred = ax_pred.text2D(0.05, 0.90, '', transform=ax_pred.transAxes)
    
    frame_text_real = ax_real.text2D(0.05, 0.95, '', transform=ax_real.transAxes)
    wrist_vel_text_real = ax_real.text2D(0.05, 0.90, '', transform=ax_real.transAxes)
    
    def update_graph(frame):
        # Update scatter plots
        graph_pred._offsets3d = (pred_xs[frame], pred_ys[frame], pred_zs[frame])
        graph_real._offsets3d = (true_xs[frame], true_ys[frame], true_zs[frame])
        
        # Update frame count text
        frame_text_pred.set_text(f'Frame: {frame}')
        frame_text_real.set_text(f'Frame: {frame}')
        
        # Update wrist velocity text
        wrist_vel_text_pred.set_text(f'Wrist Velocity: {pred_wrist_vel[frame]:.2f}')
        wrist_vel_text_real.set_text(f'Wrist Velocity: {true_wrist_vel[frame]:.2f}')
        
        return graph_pred, graph_real, frame_text_pred, wrist_vel_text_pred, frame_text_real, wrist_vel_text_real
    
    ani = animation.FuncAnimation(fig, update_graph, frames=pred_xs.shape[0], interval=50, blit=False)
    plt.show()

# Select a sequence for evaluation and perform iterative prediction
seq_idx = 10  # You can change this index to select a different sequence
seq_length = lengths_tensor[seq_idx].item()
initial_input = X_padded[seq_idx, 0, :].to(device)  # Unnormalized initial input

# Perform iterative prediction
predictions = iterative_prediction(model, initial_input, seq_length, X_mean, X_std, Y_mean, Y_std, device)

# Convert predictions to numpy arrays
Y_pred_denorm = predictions.cpu().numpy()

# Get true outputs (unnormalized)
Y_true_sequence = Y_padded[seq_idx, :seq_length, :].cpu().numpy()

# Extract coordinates for animation
pred_xs, pred_ys, pred_zs = extract_coordinates(Y_pred_denorm)
true_xs, true_ys, true_zs = extract_coordinates(Y_true_sequence)

# Calculate wrist velocity for predicted and true sequences
pred_wrist_vel = calculate_wrist_velocity(pred_xs, pred_ys, pred_zs)
true_wrist_vel = calculate_wrist_velocity(true_xs, true_ys, true_zs)

# Run the animation
create_animation(pred_xs, pred_ys, pred_zs, true_xs, true_ys, true_zs, pred_wrist_vel, true_wrist_vel)
