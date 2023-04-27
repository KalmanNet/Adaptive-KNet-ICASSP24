"""
HyperNetwork class
architecture: deconv - FC

each deconv layer doubles the input size
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class hnet_deconv(nn.Module):
    def __init__(self, args, SoW_len, output_size, num_deconv_layers=None):
        super(hnet_deconv, self).__init__()
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        input_size = 1 + SoW_len # 1 for shift/gain
        self.position_embeddings = ["0", "1"] # shift/gain
        
        if input_size >= output_size:
            raise ValueError('input_size must be smaller than output_size')
        
        if num_deconv_layers is None: # if not specified, calculate the number of deconv layers needed
            num_deconv_layers = math.ceil(math.log(output_size / input_size, 2)) # assume that output_size is always larger than input_size

        layers = []
        output_channels = input_size
        for _ in range(num_deconv_layers):
            input_channels = output_channels
            output_channels = input_channels * 2
            deconv_layer = nn.ConvTranspose1d(input_channels, output_channels, 1, stride=1)
            layers.append(deconv_layer)
            # layers.append(nn.BatchNorm1d(output_channels))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Dropout(p=0.5))
        
        layers.append(Flatten())
        input_size_linear = input_size * (2 ** num_deconv_layers)
        layers.append(nn.Linear(input_size_linear, output_size))
        self.deconv = nn.Sequential(*layers)

    def forward(self, SoW):
        input_tensors = [torch.tensor([float(ch) for ch in pe] + [SoW], dtype=torch.float32) for pe in self.position_embeddings]
        
        # Reuse MLPs for the corresponding input tensors
        outputs = []
        for input_tensor in input_tensors:
            input_tensor = input_tensor.view(1, -1, 1) # (batch size, in_channels, width)
            y = self.deconv(input_tensor)
            outputs.append(torch.squeeze(y,0))

        return outputs

