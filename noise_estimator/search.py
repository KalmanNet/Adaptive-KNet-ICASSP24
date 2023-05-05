import torch
import torch.nn as nn
import time
import math

class Pipeline_NE:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"  

    def save(self):
        torch.save(self, self.PipelineName)

    def print_grad(self, grad):
        print('Gradient:', grad)

    def setModel(self, hnet, mnet):
        self.hnet = hnet
        self.mnet = mnet

    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N_steps = args.n_steps  # Number of Training Steps
        self.learningRate = args.lr # Learning Rate

    def unsupervised_loss(self, SysModel, x_out_test, y_true):
        loss_fn = torch.nn.MSELoss(reduction='mean')
        y_hat = torch.zeros_like(y_true)
        for t in range(y_true.shape[2]):
            y_hat[:,:,t] = torch.squeeze(SysModel.h(torch.unsqueeze(x_out_test[:,:,t],2)))
        return loss_fn(y_hat, y_true)

    # Perform gradient-based search
    def gradient_search(self, SysModel, test_input, path_results, test_init, lr=0.01, num_iterations=100):
        SoW_input = torch.randn(1, requires_grad=True)
        optimizer = torch.optim.Adam([SoW_input], lr=lr)

        hnet_model_weights = torch.load(path_results+'hnet_best-model.pt', map_location=self.device)
        self.hnet.load_state_dict(hnet_model_weights)

        for i in range(num_iterations):
            optimizer.zero_grad()

            output = model(test_input)
            loss = unsupervised_loss(SysModel,output,test_input)

            loss.backward()
            optimizer.step()

        return input_tensor