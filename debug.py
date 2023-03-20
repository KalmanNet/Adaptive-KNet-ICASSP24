import torch
import torch.nn as nn
import ipdb

# Parameters
# m = 3
# in_mult = 5

# d_input_Q = m * in_mult # 15
# d_hidden_Q = m ** 2 # 9

# d_input_FC = m
# d_hidden_FC = in_mult
# d_output_FC = m * in_mult

## Print the shapes of the weights and biases for GRU
# gru = nn.GRU(d_input_Q, d_hidden_Q)
# print("Number of trainable parameters of gru:",sum(p.numel() for p in gru.parameters()))

# print(gru.weight_ih_l0.shape)
# print(gru.weight_hh_l0.shape)
# print(gru.bias_ih_l0.shape)
# print(gru.bias_hh_l0.shape)

# ## modify weights and biases of GRU
# gru.weight_ih_l0 = nn.Parameter(torch.ones(d_input_Q, d_hidden_Q))
# gru.weight_hh_l0 = nn.Parameter(torch.ones(d_hidden_Q, d_hidden_Q))

# ## Print the shapes of the weights and biases for FC
# FC = nn.Sequential(
#                 nn.Linear(d_input_FC, d_hidden_FC),
#                 nn.ReLU(),
#                 nn.Linear(d_hidden_FC, d_output_FC))
# print("Number of trainable parameters of FC:",sum(p.numel() for p in FC.parameters()))

# print(FC[0].weight.shape)
# print(FC[0].bias.shape)
# print(FC[2].weight.shape)
# print(FC[2].bias.shape)

# ## modify weights and biases of FC
# FC[0].weight = nn.Parameter(torch.ones(d_input_FC, d_hidden_FC))
# FC[0].bias = nn.Parameter(torch.ones(d_hidden_FC))

##########################################################################################

