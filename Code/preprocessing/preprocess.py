#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import os
import numpy as np
os.chdir('/home/tm_ba/Desktop/Bachelorarbeit_code')


# In[14]:


df = pd.read_csv("csv_new/processed_Trajectories.csv")
df = df.rename(columns={"Recording": "Sequence"}).drop("Sub Frame", axis=1)
df


# In[28]:


# Joints of interest
joints = ['shoulder', 'shoulderElbowSupport', 'elbow', 'wrist', 'elbowWrist', 'Triangle2', 'ThumbTip']
coordinates = ['X', 'Y', 'Z']

# Convert 'mm' to 'm' (if needed, since robot arms often use meters)
df[[f'Mathew:{joint}:{coord} (mm)' for joint in joints for coord in coordinates]] /= 1000

# Calculate velocity (difference in position between frames)
def calculate_velocity(df, joints, coordinates):
    velocity_df = pd.DataFrame()
    for joint in joints:
        for coord in coordinates:
            position_col = f'Mathew:{joint}:{coord} (mm)'
            velocity_col = f'{joint}:{coord}_velocity'
            velocity_df[velocity_col] = df.groupby('Sequence')[position_col].diff().fillna(0)  # Calculate difference between time steps
    velocity_df["Sequence"] = df["Sequence"]
    return velocity_df

# Calculate acceleration (difference in velocity between frames)
def calculate_acceleration(velocity_df, joints, coordinates):
    acceleration_df = pd.DataFrame()
    for joint in joints:
        for coord in coordinates:
            velocity_col = f'{joint}:{coord}_velocity'
            acceleration_col = f'{joint}:{coord}_acceleration'
            acceleration_df[acceleration_col] = velocity_df.groupby('Sequence')[velocity_col].diff().fillna(0)  # Difference between velocities
    acceleration_df["Sequence"] = df["Sequence"]
    return acceleration_df

# Calculate velocity and acceleration
velocity_df = calculate_velocity(df, joints, coordinates)
acceleration_df = calculate_acceleration(velocity_df, joints, coordinates)

# Merge position, velocity, and acceleration into one DataFrame
full_data = pd.concat([df, velocity_df, acceleration_df], axis=1).drop(columns=["Sequence"])
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


# In[32]:


from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Group data by sequence
grouped = full_data.groupby('Sequence')

# Prepare input (X) and output (Y) for each sequence individually
X_seqs = []
Y_seqs = []

for _, group in grouped:
    X_seq = group.drop(columns=['Frame', 'Sequence']).values[:-1]  # Input sequence (positions, velocities, accelerations)
    Y_seq = group[['Mathew:shoulder:X (mm)', 'Mathew:shoulder:Y (mm)', 'Mathew:shoulder:Z (mm)',  # Output sequence (next positions)
                   'Mathew:shoulderElbowSupport:X (mm)', 'Mathew:shoulderElbowSupport:Y (mm)', 'Mathew:shoulderElbowSupport:Z (mm)',
                   'Mathew:elbow:X (mm)', 'Mathew:elbow:Y (mm)', 'Mathew:elbow:Z (mm)',
                   'Mathew:wrist:X (mm)', 'Mathew:wrist:Y (mm)', 'Mathew:wrist:Z (mm)',
                   'Mathew:elbowWrist:X (mm)', 'Mathew:elbowWrist:Y (mm)', 'Mathew:elbowWrist:Z (mm)',
                   'Mathew:Triangle2:X (mm)', 'Mathew:Triangle2:Y (mm)', 'Mathew:Triangle2:Z (mm)',
                   'Mathew:ThumbTip:X (mm)', 'Mathew:ThumbTip:Y (mm)', 'Mathew:ThumbTip:Z (mm)']].values[1:]  # Next positions
    
    X_seqs.append(X_seq)
    Y_seqs.append(Y_seq)

# Optional: Pad sequences to the same length if needed
X_padded = pad_sequences(X_seqs, padding='post', dtype='float32')
Y_padded = pad_sequences(Y_seqs, padding='post', dtype='float32')

# X_padded and Y_padded are now properly grouped by sequence, with padding if necessary
print("X_padded shape:", X_padded.shape)
print("Y_padded shape:", Y_padded.shape)


# ### Step 1: Data Normalization

# In[51]:


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


# In[70]:


# Convert data to PyTorch tensors
X = torch.tensor(X_padded, dtype=torch.float32)
Y = torch.tensor(Y_padded, dtype=torch.float32)

# Split data into training, validation, and test sets
dataset_size = X.shape[0]
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

# Create the dataset
full_dataset = MotionDataset(X, Y)

# Split the dataset FIRST
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Compute mean and std from ONLY the training data
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    X_batch, Y_batch = next(iter(loader))
    X_flat = X_batch.view(-1, X_batch.shape[-1])
    Y_flat = Y_batch.view(-1, Y_batch.shape[-1])
    X_mean = X_flat.mean(dim=0)
    X_std = X_flat.std(dim=0)
    Y_mean = Y_flat.mean(dim=0)
    Y_std = Y_flat.std(dim=0)
    # Avoid division by zero (to handle any constant feature)
    X_std[X_std == 0] = 1
    Y_std[Y_std == 0] = 1
    return X_mean, X_std, Y_mean, Y_std

# Compute mean and std from the training set
X_mean, X_std, Y_mean, Y_std = compute_mean_std(train_dataset)

# Create normalized datasets for training, validation, and test sets
train_dataset = MotionDataset(train_dataset.dataset.X, train_dataset.dataset.Y, X_mean, X_std, Y_mean, Y_std)
val_dataset = MotionDataset(val_dataset.dataset.X, val_dataset.dataset.Y, X_mean, X_std, Y_mean, Y_std)
test_dataset = MotionDataset(test_dataset.dataset.X, test_dataset.dataset.Y, X_mean, X_std, Y_mean, Y_std)


# ### Step 2: Model Implementation

# In[54]:


import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.u = nn.Linear(hidden_dim * 2, 1, bias=False)
        
    def forward(self, inputs, mask):
        # inputs: [batch_size, seq_len, hidden_dim*2]
        # mask: [batch_size, seq_len]
        
        # Linear transformation
        u_it = torch.tanh(self.W(inputs))  # [batch_size, seq_len, hidden_dim*2]
        
        # Compute attention scores
        a_it = self.u(u_it).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask (set scores of padded elements to a large negative value)
        a_it = a_it.masked_fill(~mask, float('-1e9'))
        
        # Compute attention weights
        a_it = F.softmax(a_it, dim=1)  # [batch_size, seq_len]
        
        # Ensure no NaNs
        a_it = a_it * mask.float()  # Zero out weights where mask is False
        a_it = a_it / (a_it.sum(dim=1, keepdim=True) + 1e-9)
        
        # Compute weighted sum of inputs
        context = torch.bmm(a_it.unsqueeze(1), inputs).squeeze(1)  # [batch_size, hidden_dim*2]
        
        return context


# In[55]:


class MotionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MotionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        self.attention = AttentionLayer(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x, lengths):
        # x: [batch_size, seq_len, input_dim]
        # lengths: [batch_size]
        
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        packed_output, _ = self.lstm(packed_input)
        
        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        # output: [batch_size, seq_len, hidden_dim*2]
        
        # Create mask
        max_len = output.size(1)
        mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        # mask: [batch_size, seq_len]
        
        # Attention
        context = self.attention(output, mask)
        
        # Repeat context to match sequence length
        context = context.unsqueeze(1).expand(-1, max_len, -1)
        
        # Concatenate LSTM outputs with context
        combined = torch.cat((output, context), dim=2)
        # combined: [batch_size, seq_len, hidden_dim*4]
        
        # Output layer
        outputs = self.fc(combined)
        # outputs: [batch_size, seq_len, output_dim]
        
        return outputs, mask


# ### Step 3: Custom Loss Function

# In[56]:


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

# In[57]:


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


# In[58]:


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


# In[62]:


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
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        outputs, mask = model(X_batch, lengths)
        
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
            lengths = lengths.to(device)
            
            outputs, mask = model(X_batch, lengths)
            
            loss = custom_loss(outputs, Y_batch, mask)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_dataset)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save the model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping")
            break


# ### Step 5: Evaluation and Fine-Tuning

# In[63]:


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


# In[71]:


import matplotlib.pyplot as plt

# Select a sequence from the test set
X_sequence, Y_true_sequence, lengths = next(iter(test_loader))
sequence_idx = 0
length = lengths[sequence_idx]

X_sequence = X_sequence[sequence_idx:sequence_idx+1, :length].to(device)
Y_true_sequence = Y_true_sequence[sequence_idx:sequence_idx+1, :length].to(device)
lengths_sequence = lengths[sequence_idx:sequence_idx+1]

# Predict
model.eval()
with torch.no_grad():
    outputs, mask = model(X_sequence, lengths_sequence)
    
# Denormalize
def denormalize(data, mean, std):
    mean = mean.to(device).unsqueeze(0).unsqueeze(0)
    std = std.to(device).unsqueeze(0).unsqueeze(0)
    return data * std + mean

Y_true_denorm = denormalize(Y_true_sequence, Y_mean, Y_std)
Y_pred_denorm = denormalize(outputs, Y_mean, Y_std)

# Convert to CPU and numpy
Y_true_denorm = Y_true_denorm.squeeze(0).cpu().numpy()
Y_pred_denorm = Y_pred_denorm.squeeze(0).cpu().numpy()

# Plot true vs predicted positions for a joint
joint_idx = 0  # First joint
coordinate_idx = 0  # X-coordinate

plt.figure(figsize=(12, 6))
plt.plot(
    Y_true_denorm[:length, joint_idx * 3 + coordinate_idx],
    label='True Position',
)
plt.plot(
    Y_pred_denorm[:length, joint_idx * 3 + coordinate_idx],
    label='Predicted Position',
)
plt.title('Joint Position Over Time')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.show()


# In[69]:


get_ipython().system('jupyter nbconvert --to script Code/preprocessing/preprocess.ipynb')

