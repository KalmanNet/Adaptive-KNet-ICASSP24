import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    def __init__(self, context_dim=8, position_dim=13, d_model=32, nhead=2, num_layers=2, hidden_dim=64, output_size=None):
        super(HyperNetwork, self).__init__()

        # Positional encoding
        self.position_embedding = nn.Embedding(position_dim, d_model)

        # Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        # Feedforward layers (MLP)
        self.feedforward_layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, context, position):
        # Positional encoding
        position_embedding = self.position_embedding(position)

        # Concatenate context and positional encoding
        input_tensor = torch.cat((context, position_embedding), dim=1)

        # Multi-head attention
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        attn_output, _ = self.multi_head_attention(input_tensor, input_tensor, input_tensor)
        attn_output = attn_output.squeeze(0)  # Remove batch dimension

        # Feedforward layers
        output = self.feedforward_layers(attn_output)

        # Output layer
        output = self.output_layer(output)

        return output

# Example usage
batch_size = 8
context_dim = 8
position_dim = 13
output_size = 100  # Replace this with the total number of weights and biases for the target context modulation layers

context = torch.rand(batch_size, context_dim)
position = torch.randint(0, position_dim, (batch_size,))

hypernetwork = HyperNetwork(output_size=output_size)
output = hypernetwork(context, position)
print(output.shape)
