"""
HyperNetwork class
architecture: FC - GRU - FC
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class HyperNetwork(nn.Module):
    def __init__(self, args, SoW_len, output_size):
        super(HyperNetwork, self).__init__()
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        input_size = 1 + SoW_len # 1 for position embedding
        self.position_embeddings = [0, 1] # shift/gain

        self.hidden_size = int(output_size / args.hnet_hidden_size_discount)

        self.fc1 = nn.Linear(input_size, self.hidden_size).to(self.device)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size).to(self.device)
        self.fc2 = nn.Linear(self.hidden_size, output_size).to(self.device)

        # Apply Xavier initialization to GRU layer
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Apply He initialization to FC layers
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.fc1.bias.data.fill_(0)
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.fc2.bias.data.fill_(0)

    def forward(self, SoW):
        SoW = torch.log10(SoW)
        input_tensors = [torch.tensor([pe] + [SoW], dtype=torch.float32) for pe in self.position_embeddings]
        
        # Reuse MLPs for the corresponding input tensors
        outputs = []
        for input_tensor in input_tensors:
            x = self.fc1(input_tensor)
            x = torch.relu(x)
            x = x.unsqueeze(0).unsqueeze(0)
            x, self.hgru = self.gru(x, self.hgru)
            x = x.squeeze(0)
            x = self.fc2(x)
            outputs.append(torch.squeeze(x,0))
        return outputs
    
    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, 1, self.fc1.out_features).zero_()
        self.hgru = hidden.data

