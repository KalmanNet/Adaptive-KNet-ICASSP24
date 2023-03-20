"""
HyperNetwork class
architecture 1: FC - GRU - FC
"""

import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    def __init__(self, args, output_size):
        super(HyperNetwork, self).__init__()
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.hidden_size = int(output_size / args.hnet_hidden_size_scale)

        self.fc1 = nn.Linear(args.hnet_input_size, self.hidden_size).to(self.device)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size).to(self.device)
        self.fc2 = nn.Linear(self.hidden_size, output_size).to(self.device)

    def forward(self, SoW):
        x = self.fc1(SoW)
        x = torch.relu(x)
        x = x.unsqueeze(0).unsqueeze(0)
        x, self.hgru = self.gru(x, self.hgru)
        x = x.squeeze(0)
        x = self.fc2(x)
        return torch.squeeze(x,0)
    
    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, 1, self.fc1.out_features).zero_()
        self.hgru = hidden.data


# if __name__ == '__main__':
#     import sys
#     sys.path.append('C://Users//xiaoy//Documents//learning//ETH_master//semester5//Thesis//codes//Hyper-KalmanNet')
#     import simulations.config as config
    
#     args = config.general_settings()
#     hnet = HyperNetwork(args, 100)
#     Q_t = torch.tensor([10]).type(torch.float)
#     R_t = torch.tensor([10]).type(torch.float)
#     hnet.init_hidden()
#     weights = hnet(Q_t, R_t)
#     print(weights.shape)
