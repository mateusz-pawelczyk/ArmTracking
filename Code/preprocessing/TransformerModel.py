import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create constant 'pe' matrix with values dependent on
        # position and i (dimension)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the positional encodings once in log space
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class MotionTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout=0.1):
        super(MotionTransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, dim_feedforward=hidden_dim*4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: [batch_size, seq_len, input_dim]
        src = src.transpose(0, 1)  # Transpose for transformer input: [seq_len, batch_size, input_dim]

        # Input projection
        src = self.input_proj(src)  # [seq_len, batch_size, hidden_dim]

        # Positional Encoding
        src = self.pos_encoder(src)  # [seq_len, batch_size, hidden_dim]

        # Transformer Encoder
        memory = self.transformer_encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Output projection
        output = self.output_proj(memory)  # [seq_len, batch_size, output_dim]

        # Transpose back to [batch_size, seq_len, output_dim]
        output = output.transpose(0, 1)

        return output

    def generate(self, initial_input, seq_len):
        # initial_input shape: [batch_size, 1, input_dim]
        generated = [initial_input]  # List to store generated outputs

        for _ in range(seq_len - 1):
            src = torch.cat(generated, dim=1)  # Concatenate along sequence dimension
            src_mask = None  # You can define a mask if needed

            # Forward pass
            output = self.forward(src, src_mask=src_mask)  # [batch_size, current_seq_len, output_dim]

            # Get the last time step output
            next_input = output[:, -1:, :]  # [batch_size, 1, output_dim]

            # Append to the generated list
            generated.append(next_input)

        # Concatenate all generated outputs
        generated_sequence = torch.cat(generated, dim=1)  # [batch_size, seq_len, output_dim]

        return generated_sequence
