#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import os
import numpy as np
import torch
os.chdir('/home/tm_ba/Desktop/Bachelorarbeit_code')


# In[98]:


df = pd.read_csv("csv_new/processed_Trajectories.csv")
df


# In[99]:


df.value_counts("Sequence")


# In[100]:


# Joints of interest
joints = ['shoulder', 'elbow', 'wrist', 'ThumbTip']
coordinates = ['X', 'Y', 'Z']

# Convert 'mm' to 'm' (if needed, since robot arms often use meters)
df[[f'{joint}:{coord}' for joint in joints for coord in coordinates]]




# Merge position, velocity, and acceleration into one DataFrame
full_data = df.drop(columns=["Sequence"])
full_data["Sequence"] = df["Sequence"]

# Organize by sequence: Create sequences for the model
sequences = full_data.groupby('Sequence').apply(lambda group: group.drop(columns=['Frame', 'Sequence']).values).values

# Example: Sequences of joint positions, velocity, and acceleration for input
X = []
Y = []

# Prepare the data: Input (X) will be current positions, velocities, accelerations
# Output (Y) will be the next time step's positions

for seq in sequences:
    # Each sequence is a separate time-series for one movement
    for i in range(len(seq) - 1):  # We stop at len(seq)-1 to predict the next time step
        X.append(seq[i])  # Input: current time step (positions, velocity, acceleration)
        Y.append(seq[i+1][:len(joints)*3])  # Output: next time step's positions (x, y, z)

# Convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# X shape: (num_samples, num_features) -- for LSTMs, should reshape to (num_samples, timesteps, num_features)
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print(f"X range: {X.min()} to {X.max()}",)
print(f"Y range: {Y.min()} to {Y.max()}",)


# In[101]:


full_data.value_counts("Sequence")


# In[102]:


import torch
from torch.nn.utils.rnn import pad_sequence

# Group data by sequence
grouped = full_data.groupby('Sequence')

# Prepare input (X) and output (Y) for each sequence individually
X_seqs = []
Y_seqs = []
lengths = []  # To store original sequence lengths

for _, group in grouped:
    X_seq = torch.tensor(group.drop(columns=['Frame', 'Sequence']).values[:-1], dtype=torch.float32)  # Input sequence (positions, velocities, accelerations)
    Y_seq = torch.tensor(group[['shoulder:X', 'shoulder:Y', 'shoulder:Z',  # Output sequence (next positions)
                   'elbow:X', 'elbow:Y', 'elbow:Z',
                   'wrist:X', 'wrist:Y', 'wrist:Z',
                   'ThumbTip:X', 'ThumbTip:Y', 'ThumbTip:Z']].values[1:], dtype=torch.float32)  # Next positions
    
    # Only add non-empty sequences
    if len(X_seq) > 0 and len(Y_seq) > 0:
        X_seqs.append(X_seq)
        Y_seqs.append(Y_seq)
        # Store original length before padding
        lengths.append(len(X_seq))



# Pad sequences to the same length using PyTorch pad_sequence
X_padded = pad_sequence(X_seqs, batch_first=True, padding_value=0.0)
Y_padded = pad_sequence(Y_seqs, batch_first=True, padding_value=0.0)

# Convert original lengths to PyTorch tensor
lengths_tensor = torch.tensor(lengths)

# X_padded and Y_padded are now properly grouped by sequence, with padding if necessary
print("X_padded shape:", X_padded.shape)
print("Y_padded shape:", Y_padded.shape)
print("Lengths shape:", lengths_tensor.shape)


# ### Step 1: Data Normalization

# In[103]:


from Code.preprocessing.Model import MotionModel, AttentionLayer
from Code.preprocessing.TransformerModel import MotionTransformerModel


# In[104]:


import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MotionDataset(Dataset):
    def __init__(self, X, Y, X_mean=None, X_std=None, Y_mean=None, Y_std=None):
        self.X = X
        self.Y = Y
        self.X_mean = X_mean
        self.X_std = X_std
        self.Y_mean = Y_mean
        self.Y_std = Y_std

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        X_item = self.X[idx]
        Y_item = self.Y[idx]
        
        # Apply normalization if mean and std are provided
        if self.X_mean is not None and self.X_std is not None:
            X_item = (X_item - self.X_mean) / self.X_std
        if self.Y_mean is not None and self.Y_std is not None:
            Y_item = (Y_item - self.Y_mean) / self.Y_std
        
        return X_item, Y_item


# In[105]:


import torch
from torch.utils.data import DataLoader, random_split

# Convert data to PyTorch tensors
X = torch.tensor(X_padded, dtype=torch.float32)
Y = torch.tensor(Y_padded, dtype=torch.float32)

# Split data into training, validation, and test sets
dataset_size = X.shape[0]
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

# Create the dataset (your MotionDataset should take X and Y as arguments)
full_dataset = MotionDataset(X, Y)

# Split the lengths tensor in the same way as the dataset
lengths_tensor = torch.tensor(lengths, dtype=torch.long)

# Calculate the indices for splitting
train_idx = range(0, train_size)
val_idx = range(train_size, train_size + val_size)
test_idx = range(train_size + val_size, dataset_size)

# Split the lengths tensor using the same indices
lengths_train = lengths_tensor[train_idx]
lengths_val = lengths_tensor[val_idx]
lengths_test = lengths_tensor[test_idx]

# Split the dataset using the same proportions
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Compute mean and std from ONLY the training data
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    X_batch, Y_batch = next(iter(loader))
    X_flat = X_batch.view(-1, X_batch.shape[-1])  # Flatten the data along all but the last dimension
    Y_flat = Y_batch.view(-1, Y_batch.shape[-1])  # Flatten labels in the same way
    X_mean = X_flat.mean(dim=0)
    X_std = X_flat.std(dim=0)
    Y_mean = Y_flat.mean(dim=0)
    Y_std = Y_flat.std(dim=0)
    # Avoid division by zero for any constant feature
    X_std[X_std == 0] = 1
    Y_std[Y_std == 0] = 1
    return X_mean, X_std, Y_mean, Y_std

# Compute mean and std from the training set
X_mean, X_std, Y_mean, Y_std = compute_mean_std(train_dataset)

# Now create a normalized dataset by passing the computed mean and std
class NormalizedMotionDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, X_mean, X_std, Y_mean, Y_std):
        self.X = (X - X_mean) / X_std
        self.Y = (Y - Y_mean) / Y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Normalize each split using the mean and std from the training set
train_X, train_Y = zip(*[(X, Y) for X, Y in train_dataset])
val_X, val_Y = zip(*[(X, Y) for X, Y in val_dataset])
test_X, test_Y = zip(*[(X, Y) for X, Y in test_dataset])

train_X = torch.stack(train_X)
train_Y = torch.stack(train_Y)
val_X = torch.stack(val_X)
val_Y = torch.stack(val_Y)
test_X = torch.stack(test_X)
test_Y = torch.stack(test_Y)

# Create normalized datasets
train_dataset = NormalizedMotionDataset(train_X, train_Y, X_mean, X_std, Y_mean, Y_std)
val_dataset = NormalizedMotionDataset(val_X, val_Y, X_mean, X_std, Y_mean, Y_std)
test_dataset = NormalizedMotionDataset(test_X, test_Y, X_mean, X_std, Y_mean, Y_std)

# Now you have normalized datasets for training, validation, and testing
print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")


# ### Step 2: Model Implementation

# In[106]:


import torch.nn as nn
import torch.nn.functional as F


# In[230]:





# ### Step 3: Custom Loss Function

# In[107]:


def custom_loss(y_pred, y_true, mask):
    # mask: [batch_size, seq_len]
    mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
    mask = mask.float()
    
    # Position loss
    position_loss = ((y_true - y_pred) ** 2) * mask
    position_loss = position_loss.sum() / mask.sum()
    
    # Compute velocities
    def compute_velocity(y):
        y_shifted = torch.zeros_like(y)
        y_shifted[:, 1:, :] = y[:, :-1, :]
        velocity = (y - y_shifted) * mask
        return velocity
    
    # Compute accelerations
    def compute_acceleration(v):
        v_shifted = torch.zeros_like(v)
        v_shifted[:, 1:, :] = v[:, :-1, :]
        acceleration = (v - v_shifted) * mask
        return acceleration
    
    # Predicted velocities and accelerations
    pred_velocity = compute_velocity(y_pred)
    pred_acceleration = compute_acceleration(pred_velocity)
    
    # True velocities and accelerations
    true_velocity = compute_velocity(y_true)
    true_acceleration = compute_acceleration(true_velocity)
    
    # Velocity loss
    velocity_loss = ((true_velocity - pred_velocity) ** 2) * mask
    velocity_loss = velocity_loss.sum() / mask.sum()
    
    # Acceleration loss
    acceleration_loss = ((true_acceleration - pred_acceleration) ** 2) * mask
    acceleration_loss = acceleration_loss.sum() / mask.sum()
    
    # Total loss with weighting
    total_loss = (
        position_loss
        + 0.1 * velocity_loss
        + 0.01 * acceleration_loss
    )
    return total_loss


# ### Step 4: Model Training

# In[108]:


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # batch is a list of tuples (X_item, Y_item)
    X_batch, Y_batch = zip(*batch)
    lengths = [x.size(0) for x in X_batch]
    # Pad sequences
    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=0.0)
    Y_padded = pad_sequence(Y_batch, batch_first=True, padding_value=0.0)
    lengths = torch.tensor(lengths)
    return X_padded, Y_padded, lengths

batch_size = 16

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)


# In[110]:


input_dim = X.shape[2]
hidden_dim = 128
output_dim = Y.shape[2]
num_layers = 2

# Initialize the model
model = MotionModel(input_dim, hidden_dim, output_dim, num_layers)

# Use DataParallel for multi-GPU support
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[111]:


print(input_dim, hidden_dim, output_dim, num_layers)


# In[112]:


from tqdm import tqdm

num_epochs = 100
best_val_loss = float('inf')
patience = 10
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, Y_batch, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        lengths = lengths.to(device)  # Original lengths, not padded lengths
        
        optimizer.zero_grad()
        outputs, mask = model(X_batch, lengths)  # Pass correct lengths for masking
        
        loss = custom_loss(outputs, Y_batch, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch, lengths in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            lengths = lengths.to(device)  # Original lengths, not padded lengths
            
            outputs, mask = model(X_batch, lengths)  # Pass correct lengths for masking
            
            loss = custom_loss(outputs, Y_batch, mask)
            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(val_dataset)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save the model
        torch.save(X_mean, 'X_mean.pth')
        torch.save(X_std, 'X_std.pth')
        torch.save(Y_mean, 'Y_mean.pth')
        torch.save(Y_std, 'Y_std.pth')

        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping")
            break


# ### Step 5: Evaluation and Fine-Tuning

# In[113]:


# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

model.eval()
test_loss = 0.0
with torch.no_grad():
    for X_batch, Y_batch, lengths in test_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        lengths = lengths.to(device)
        
        outputs, mask = model(X_batch, lengths)
        
        loss = custom_loss(outputs, Y_batch, mask)
        test_loss += loss.item() * X_batch.size(0)
test_loss /= len(test_dataset)
print(f"Test Loss: {test_loss:.6f}")


# In[114]:


import random
import matplotlib.pyplot as plt

# Select a random sequence from the test set
X_sequence, Y_true_sequence, lengths = next(iter(test_loader))

# Randomly select a sequence index within the batch
sequence_idx = random.randint(0, X_sequence.size(0) - 1)  # Random index from 0 to batch size - 1
length = lengths_tensor[sequence_idx]  # Ensure the correct original length is used for this sequence
print(X_sequence.shape)

# Slice up to the true length of the sequence
X_sequence = X_sequence[sequence_idx:sequence_idx+1, :length].to(device)
Y_true_sequence = Y_true_sequence[sequence_idx:sequence_idx+1, :length].to(device)
lengths_sequence = lengths_tensor[sequence_idx:sequence_idx+1]
print(X_sequence.shape)

# Predict
model.eval()
with torch.no_grad():
    outputs, mask = model(X_sequence, lengths_sequence)  # Pass the correct sequence length for evaluation

# Denormalize the output and true values
def denormalize(data, mean, std):
    mean = mean.to(device).unsqueeze(0).unsqueeze(0)
    std = std.to(device).unsqueeze(0).unsqueeze(0)
    return data * std + mean

# Denormalize
Y_true_denorm = denormalize(Y_true_sequence, Y_mean, Y_std)
Y_pred_denorm = denormalize(outputs, Y_mean, Y_std)

# Convert to CPU and numpy arrays for plotting
Y_true_denorm = Y_true_denorm.squeeze(0).cpu().numpy()
Y_pred_denorm = Y_pred_denorm.squeeze(0).cpu().numpy()

# Plot true vs predicted positions for a specific joint and coordinate
joint_idx = 0  # First joint
coordinate_idx = 0  # X-coordinate

plt.figure(figsize=(12, 6))
plt.plot(
    Y_true_denorm[:length, joint_idx * 3 + coordinate_idx],  # Only plot non-padded values
    label='True Position',
)
plt.plot(
    Y_pred_denorm[:length, joint_idx * 3 + coordinate_idx],  # Only plot non-padded values
    label='Predicted Position',
)
plt.title('Joint Position Over Time')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.show()


# In[115]:


lengths_tensor[0]


# In[ ]:





# In[116]:


get_ipython().system('jupyter nbconvert --to script Code/preprocessing/preprocess.ipynb')


# In[ ]:




