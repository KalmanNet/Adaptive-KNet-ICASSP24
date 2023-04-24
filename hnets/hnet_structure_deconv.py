"""
HyperNetwork class
architecture: deconv networks for different parts of the target network

each deconv layer doubles the input size
"""

import torch
import torch.nn as nn
import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class hnet_structure_deconv(nn.Module):
    def __init__(self, SoW_len, output_sizes):
        super(hnet_structure_deconv, self).__init__()

        # fixed division of CM layers for KalmanNet
        self.num_deconvs = 7
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
        
        self.deconvs = nn.ModuleList()
        for i in range(self.num_deconvs):
            if input_sizes[i] >= output_sizes[i]:
                num_deconv_layers = 1
            else:
                num_deconv_layers = math.ceil(math.log(output_sizes[i] / input_sizes[i], 4))

            layers = []
            output_channels = input_sizes[i]
            for _ in range(num_deconv_layers):
                input_channels = output_channels
                output_channels = input_channels * 2
                deconv_layer = nn.ConvTranspose1d(input_channels, output_channels, 2, stride=2)
                layers.append(deconv_layer)
                layers.append(nn.BatchNorm1d(output_channels))
                layers.append(nn.ReLU(inplace=True))
            
            layers.append(Flatten())
            input_size_linear = input_sizes[i] * (4 ** num_deconv_layers)
            layers.append(nn.Linear(input_size_linear, output_sizes[i]))
            deconv = nn.Sequential(*layers)

            self.deconvs.append(deconv)

        
    def forward(self, SoW):
        # Create input tensors by concatenating position embeddings with SoW
        input_tensors = [torch.tensor([float(ch) for ch in pe] + [SoW], dtype=torch.float32) for pe in self.position_embeddings]

        # Reuse deconvs for the corresponding input tensors
        outputs = []
        for i, input_tensor in enumerate(input_tensors):
            if i < 8:
                deconv_idx = 0
            elif i < 12:
                deconv_idx = 1
            elif i < 14:
                deconv_idx = 2
            elif i < 16:
                deconv_idx = 3
            elif i < 20:
                deconv_idx = 4
            elif i < 24:
                deconv_idx = 5
            else:
                deconv_idx = 6
            input_tensor = input_tensor.view(1, -1, 1) # (batch size, in_channels, width)
            outputs.append(self.deconvs[deconv_idx](input_tensor))

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


