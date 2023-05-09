import torch
import torch.nn as nn
import time
import math
import random

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
        self.N_steps = args.n_steps
        self.lr = args.lr
        self.grid_size = args.grid_size_dB

    def unsupervised_loss(self, SysModel, x_out_test, y_true):
        y_hat = torch.zeros_like(y_true)
        for t in range(y_true.shape[2]):
            y_hat[:,:,t] = torch.squeeze(SysModel.h(torch.unsqueeze(x_out_test[:,:,t],2)))
        # Compute MSE loss
        loss = torch.mean((y_hat - y_true)**2)
        return loss
    
    def supervised_loss(self, x_out_test, x_true):
        # Compute MSE loss
        loss = torch.mean((x_out_test - x_true)**2)
        return loss
    
    def train_unsupervised(self, sys_model, test_input, path_results, test_init):
        # data size
        self.N_T = test_input.shape[0]
        sysmdl_n = test_input.shape[1]
        sysmdl_m = sys_model.m
        sysmdl_T_test = test_input.size()[2]
        # select 1 sequence
        self.n_t = random.sample(range(self.N_T), k=1)
        
        # Init Training tensors
        y_training = torch.zeros([1, sysmdl_n, sysmdl_T_test]).to(self.device)
        x_out_training = torch.zeros([1, sysmdl_m, sysmdl_T_test]).to(self.device)
        train_init = torch.empty([1, sysmdl_m,1]).to(self.device)

        # Training data
        y_training = test_input[self.n_t]                              
        # Init Sequence
        train_init = test_init[self.n_t]
        

        self.mnet.batch_size = 1      
        self.hnet.train() 

        for ti in range(0, self.N_steps):    
            self.optimizer.zero_grad() 
            # Init Hidden State
            if self.args.hnet_arch == "GRU":
                self.hnet.init_hidden()
            self.mnet.init_hidden()         
            self.mnet.InitSequence(train_init, sysmdl_T_test)
            # Forward Computation
            weights_cm_shift, weights_cm_gain = self.hnet() # input SoW_train

            for t in range(0, sysmdl_T_test):
                x_out_training[:, :, t] = torch.squeeze(self.mnet(torch.unsqueeze(y_training[:, :, t],2), weights_cm_gain=weights_cm_gain, weights_cm_shift=weights_cm_shift))
                

    # perform grid search
    def grid_search(self, SoW_range_dB, sys_model, test_input, path_results, test_init, test_target=None, SoW_true=None):
        
        if self.args.wandb_switch: 
            import wandb

        # Load model     
        hnet_model_weights = torch.load(path_results+'hnet_best-model.pt', map_location=self.device)
        self.hnet.load_state_dict(hnet_model_weights)

        # Create the grid of input values
        left, right = SoW_range_dB
        num_steps = int((right - left) / self.grid_size) + 1
        input_values = torch.linspace(left, right, num_steps)

        # data size
        self.N_T = test_input.shape[0]
        sysmdl_T_test = test_input.size()[2]
        sysmdl_m = sys_model.m
        # Init arrays
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        x_out = torch.zeros([self.N_T, sysmdl_m,sysmdl_T_test]).to(self.device)

        # Compute the loss for each input value
        losses = []
        self.mnet.UpdateSystemDynamics(sys_model)
        self.mnet.batch_size = self.N_T
        for input_value in input_values:            
            # Init Hidden State
            if self.args.hnet_arch == "GRU":
                        self.hnet.init_hidden()
            self.mnet.init_hidden()
            # Init Sequence
            self.mnet.InitSequence(test_init, sysmdl_T_test)               
            # SoW to linear scale
            SoW_input_linear = 10**(input_value/10)
            weights_cm_shift, weights_cm_gain = self.hnet(SoW_input_linear)
            for t in range(0, sysmdl_T_test):
                x_out[:,:, t] = torch.squeeze(self.mnet(torch.unsqueeze(test_input[:,:, t],2), weights_cm_gain=weights_cm_gain, weights_cm_shift=weights_cm_shift))
            
            # Compute loss
            loss = self.unsupervised_loss(sys_model,x_out,test_input)           
            # print("SoW:", input_value, "[dB]", "Unsupervised Loss:", 10 * torch.log10(loss), "[dB]")
            # loss = self.supervised_loss(x_out,test_target)
            # print("SoW:", input_value, "[dB]", "Supervised Loss:", 10 * torch.log10(loss), "[dB]")
            losses.append(loss.item())
        
        # Find the index of the input value with the minimum loss
        min_index = torch.argmin(torch.tensor(losses))
        # Return the input value corresponding to the minimum loss
        optimal_SoW_dB = input_values[min_index]
        optimal_SoW_linear = 10**(optimal_SoW_dB/10)

        # Print minimum SoW and its corresponding supervised loss
        if test_target is not None:
            # Compute MSE loss
            # Init Hidden State
            if self.args.hnet_arch == "GRU":
                        self.hnet.init_hidden()
            self.mnet.init_hidden()
            # Init Sequence
            self.mnet.InitSequence(test_init, sysmdl_T_test) 
            weights_cm_shift, weights_cm_gain = self.hnet(optimal_SoW_linear)
            for t in range(0, sysmdl_T_test):
                x_out[:,:, t] = torch.squeeze(self.mnet(torch.unsqueeze(test_input[:,:, t],2), weights_cm_gain=weights_cm_gain, weights_cm_shift=weights_cm_shift))
            # Compute MSE loss
            loss = self.supervised_loss(x_out,test_target)
            print("Optimal SoW:", optimal_SoW_dB, "[dB]", "Supervised Loss:", 10 * torch.log10(loss), "[dB]")

        # Compare with true SoW and supervised loss
        if test_target is not None and SoW_true is not None:
            # Init Hidden State
            if self.args.hnet_arch == "GRU":
                        self.hnet.init_hidden()
            self.mnet.init_hidden()
            # Init Sequence
            self.mnet.InitSequence(test_init, sysmdl_T_test) 
            weights_cm_shift, weights_cm_gain = self.hnet(SoW_true)
            for t in range(0, sysmdl_T_test):
                x_out[:,:, t] = torch.squeeze(self.mnet(torch.unsqueeze(test_input[:,:, t],2), weights_cm_gain=weights_cm_gain, weights_cm_shift=weights_cm_shift))
            # Compute MSE loss
            loss = self.supervised_loss(x_out,test_target)
            print("True SoW:", 10*torch.log10(SoW_true), "[dB]", "Supervised Loss:", 10 * torch.log10(loss), "[dB]")
            
            
        
        ### Optinal: record loss on wandb
        if self.args.wandb_switch:
            wandb.log({'Unsupervised_loss': 10 * torch.log10(losses[min_index])})
        ###

        return optimal_SoW_linear
    

    # Perform gradient-based search (gradient backpropagation has problem, need to be fixed)
    # def gradient_search(self, SoW_range_dB, sys_model, test_input, path_results, test_init):
        
    #     if self.args.wandb_switch: 
    #         import wandb

    #     # Load model     
    #     hnet_model_weights = torch.load(path_results+'hnet_best-model.pt', map_location=self.device)
    #     self.hnet.load_state_dict(hnet_model_weights)

    #     left, right = SoW_range_dB
    #     SoW_input = torch.FloatTensor(1).uniform_(left, right).requires_grad_(True)

    #     self.mnet.UpdateSystemDynamics(sys_model)
    #     # data size
    #     self.N_T = test_input.shape[0]
    #     sysmdl_T_test = test_input.size()[2]
    #     sysmdl_m = sys_model.m
    #     # Init arrays
    #     self.MSE_test_linear_arr = torch.zeros([self.N_T])
    #     x_out = torch.zeros([self.N_T, sysmdl_m,sysmdl_T_test]).to(self.device)

    #     for i in range(self.N_steps):
    #         self.mnet.batch_size = self.N_T
    #         # Init Hidden State
    #         if self.args.hnet_arch == "GRU":
    #                     self.hnet.init_hidden()
    #         self.mnet.init_hidden()
    #         # Init Sequence
    #         self.mnet.InitSequence(test_init, sysmdl_T_test)               
    #         print("SoW:", SoW_input, "[dB]")
    #         # SoW to linear scale
    #         SoW_input_linear = 10**(SoW_input/10)
    #         weights_cm_shift, weights_cm_gain = self.hnet(SoW_input_linear)
    #         for t in range(0, sysmdl_T_test):
    #             x_out[:,:, t] = torch.squeeze(self.mnet(torch.unsqueeze(test_input[:,:, t],2), weights_cm_gain=weights_cm_gain, weights_cm_shift=weights_cm_shift))
            
    #         # Compute loss
    #         loss = self.unsupervised_loss(sys_model,x_out,test_input)

    #         loss.backward(retain_graph=True)
    #         # Manually update the input_value with gradients and learning rate
    #         with torch.no_grad():
    #             SoW_input -= self.learningRate * SoW_input.grad

    #             # Project the input back to the allowed range
    #             SoW_input.clamp_(left, right)

    #             # Reset the gradient of input_value for the next iteration
    #             SoW_input.grad.zero_()

    #         # Print MSE loss
    #         MSE_test_dB_avg = 10 * torch.log10(loss)            
    #         print("Unsupervised Loss:", MSE_test_dB_avg, "[dB]")

    #         ### Optinal: record loss on wandb
    #         if self.args.wandb_switch:
    #             wandb.log({'Unsupervised_loss':MSE_test_dB_avg})
    #         ###

    #     return SoW_input