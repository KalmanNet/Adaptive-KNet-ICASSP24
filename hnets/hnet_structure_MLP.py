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
        self.num_mlps = 7
        input_sizes = [size + SoW_len for size in [3,2,1,1,2,2,1]]
        self.position_embeddings = [
            "000", "001", "010", "011", "100", "101", "110", "111", # LSTM Q/Sigma ih/hh shift/gain
            "00", "01", "10", "11", # LSTM S ih/hh shift/gain
            "0", "1", # FC1 shift/gain
            "0", "1", # FC2 shift/gain
            "00", "01", "10", "11", # FC 3/4 shift/gain
            "00", "01", "10", "11", # FC 5/6 shift/gain
            "0", "1" # FC7 shift/gain
        ]
        # the number of hidden layers for each MLP (changable)
        hidden_layers = [3, 2, 1, 1, 1, 1, 1] * 10

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
            if i < 8:
                mlp_idx = 0
            elif i < 12:
                mlp_idx = 1
            elif i < 14:
                mlp_idx = 2
            elif i < 16:
                mlp_idx = 3
            elif i < 20:
                mlp_idx = 4
            elif i < 24:
                mlp_idx = 5
            else:
                mlp_idx = 6
            outputs.append(self.mlps[mlp_idx](input_tensor))

        output_order = [
                        13, 12, # fc1_gain, fc1_shift
                        15, 14, # fc2_gain, fc2_shift
                        17, 16, # fc3_gain, fc3_shift
                        19, 18, # fc4_gain, fc4_shift
                        21, 20, # fc5_gain, fc5_shift
                        23, 22, # fc6_gain, fc6_shift
                        25, 24, # fc7_gain, fc7_shift
                        1, 0, # lstm_q_ih_gain, lstm_q_ih_shift
                        3, 2, # lstm_q_hh_gain, lstm_q_hh_shift
                        5, 4, # lstm_sigma_ih_gain, lstm_sigma_ih_shift
                        7, 6, # lstm_sigma_hh_gain, lstm_sigma_hh_shift
                        9, 8, # lstm_s_ih_gain, lstm_s_ih_shift
                        11, 10] # lstm_s_hh_gain, lstm_s_hh_shift

        reordered_outputs = [outputs[i] for i in output_order]
        return reordered_outputs


