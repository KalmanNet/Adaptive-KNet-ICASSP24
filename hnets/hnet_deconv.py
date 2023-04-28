"""
HyperNetwork class
architecture: deconv

each deconv layer doubles the input size
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class ContextPositionEmbedding(nn.Module):
    def __init__(self, SoW_len, embedding_dim):
        super(ContextPositionEmbedding, self).__init__()
        self.context_embedding = nn.Linear(SoW_len, embedding_dim)
        self.position_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, position, context):
        # Logarithmize context
        context = torch.log10(context)
        # Embedding
        position_emb = self.position_embedding(position)
        context_emb = self.context_embedding(context.unsqueeze(0))

        # Concatenate embeddings along the channel dimension
        combined_emb = torch.cat((context_emb.unsqueeze(0), position_emb.unsqueeze(0)), dim=0).unsqueeze(0)

        return combined_emb # [batch_size=1, channel=2, embedding_dim]


class hnet_deconv(nn.Module):
    def __init__(self, args, SoW_len, output_size, embedding_dim=8, hidden_channel_dim = 32):
        super(hnet_deconv, self).__init__()
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Checking
        if SoW_len > embedding_dim:
            raise ValueError('embedding_dim must be larger than SoW_len.')
        input_size = embedding_dim * 2 # 2 for position and context
        if input_size >= output_size:
            raise ValueError('input_size must be smaller than output_size')
        self.position_embeddings = torch.tensor([0, 1]).to(torch.int) # shift/gain
        
        # Network architecture (embedding layer + deconv layers)       
        self.embedding_layer = ContextPositionEmbedding(SoW_len, embedding_dim)

        self.deconv_layers = []
        num_deconv_layers_scaleup_channel = math.floor(math.log(hidden_channel_dim / 2, 2))
        out_channel = 2 # 2 for position and context
        for _ in range(num_deconv_layers_scaleup_channel):
            in_channel = out_channel
            out_channel = in_channel * 2
            deconv_layer = nn.ConvTranspose1d( # revert to high dimension
                                    in_channels=in_channel, 
                                    out_channels=out_channel, 
                                    kernel_size=3, 
                                    stride=1,
                                    padding=1)
            self.deconv_layers.append(deconv_layer)
            self.deconv_layers.append(nn.ReLU(inplace=True))
        # Final deconv layer for scaling up the channel dimension
        deconv_layer = nn.ConvTranspose1d( # revert to high dimension
                                    in_channels=out_channel,
                                    out_channels=hidden_channel_dim,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.deconv_layers.append(deconv_layer)


        num_deconv_layers = math.floor(math.log(output_size / embedding_dim, 2))
        out_channel = hidden_channel_dim
        for _ in range(num_deconv_layers):
            in_channel = out_channel
            out_channel = math.ceil(in_channel / 2)
            deconv_layer = nn.ConvTranspose1d( # each deconv layer doubles the input size, but cut half the channel dim
                                in_channels=in_channel, 
                                out_channels=out_channel, 
                                kernel_size=4, 
                                stride=2, 
                                padding=1)
            self.deconv_layers.append(deconv_layer)
            # self.deconv_layers.append(nn.BatchNorm1d(out_channel))
            self.deconv_layers.append(nn.ReLU(inplace=True))
            # self.layers.append(nn.Dropout(p=0.5))
        
        # Final deconv layer
        input_size_final = embedding_dim * 2 ** num_deconv_layers
        kernel_size = output_size - input_size_final + 1
        stride = 1
        padding = 0
        output_padding = 0
        self.deconv_layers.append(nn.ConvTranspose1d(out_channel, 1, kernel_size, stride, padding, output_padding))
                                                               
        self.deconv = nn.Sequential(*self.deconv_layers)

    def forward(self, SoW):      
        # Reuse MLPs for the corresponding input tensors
        outputs = []
        for pe in self.position_embeddings:
            y = self.embedding_layer(pe, SoW)
            y = self.deconv(y)
            outputs.append(torch.squeeze(y))

        return outputs

    def count_weights(self, layer):
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)
    
    def print_num_weights(self):
        deconv_weights = 0
        embed_weights = 0

        for layer in self.deconv_layers:          
            deconv_weights += self.count_weights(layer)
        
        embed_weights = sum(p.numel() for p in self.embedding_layer.parameters() if p.requires_grad)

        print(f"Number of weights in deconv layers: {deconv_weights}")
        print(f"Number of weights in embedding layers: {embed_weights}")
        print(f"Total number of weights: {deconv_weights+embed_weights}")

        return deconv_weights+embed_weights
