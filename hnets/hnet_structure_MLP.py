"""
HyperNetwork class
architecture: MLPs for different parts of the target network

hidden layer sizes are geometric mean of input and output sizes
"""

import torch
import torch.nn as nn
import math

class hnet_structure_MLP(nn.Module):
    def __init__(self, SoW_len, output_sizes):
        super(hnet_structure_MLP, self).__init__()

        # fixed division of CM layers for KalmanNet
        self.num_mlps = 2
        input_sizes = [size + SoW_len for size in [2,1]]
        self.position_embeddings = [
            "00", "01", "10", "11", # shift/gain LSTM Q/Sigma
            "0", "1" # shift/gain LSTM S
        ]
        # the number of hidden layers for each MLP (changable)
        hidden_layers = [5,5]

        self.mlps = nn.ModuleList()
        for i in range(self.num_mlps):
            layers = []
            hidden_layer_size = int(math.sqrt(input_sizes[i] * output_sizes[i]))
            layers.append(nn.Linear(input_sizes[i], hidden_layer_size))
            layers.append(nn.ReLU())

            for _ in range(hidden_layers[i] - 1):
                layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_layer_size, output_sizes[i]))

            mlp = nn.Sequential(*layers)
            self.mlps.append(mlp)

        
    def forward(self, SoW):
        # Create input tensors by concatenating position embeddings with SoW
        input_tensors = [torch.tensor([float(ch) for ch in pe] + [SoW], dtype=torch.float32) for pe in self.position_embeddings]

        # Reuse MLPs for the corresponding input tensors
        outputs = []
        for i, input_tensor in enumerate(input_tensors):
            if i < 4:
                mlp_idx = 0
            else:
                mlp_idx = 1
            outputs.append(self.mlps[mlp_idx](input_tensor))

        output_order = [
                        2, 0, # lstm_q_ih_gain, lstm_q_ih_shift
                        
                        3, 1, # lstm_sigma_ih_gain, lstm_sigma_ih_shift
                        
                        5, 4, # lstm_s_ih_gain, lstm_s_ih_shift
                        ]

        reordered_outputs = [outputs[i] for i in output_order]
        return reordered_outputs


