import torch.nn as nn
import torch.nn.functional as F
import torch

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
        
        # Create mask based on the correct sequence lengths (original, not padded)
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

