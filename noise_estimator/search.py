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
        self.forget_factor = args.forget_factor
        self.max_iter = args.max_iter
        self.SoW_conv_error = args.SoW_conv_error

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
    
    def exp_smoothing(self, P_old, P_new):
        return self.forget_factor * P_old + (1 - self.forget_factor) * P_new

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
    

    def innovation_based_estimation(self, SoW_old, Q_old, R_old, sys_model, test_input, path_results, test_init):
        # Load model     
        hnet_model_weights = torch.load(path_results+'hnet_best-model.pt', map_location=self.device)
        self.hnet.load_state_dict(hnet_model_weights)

        # data size
        self.N_T = test_input.shape[0]
        sysmdl_T_test = test_input.size()[2]
        sysmdl_m = sys_model.m
        sysmdl_n = test_input.size()[1]
        self.mnet.UpdateSystemDynamics(sys_model)
        self.mnet.batch_size = self.N_T
        
        # Init arrays
        x_out = torch.zeros([self.N_T, sysmdl_m,sysmdl_T_test]).to(self.device)
        KG_array = torch.zeros([sysmdl_T_test, self.N_T, sysmdl_m, sysmdl_n]).to(self.device)
        dy_array = torch.zeros([sysmdl_T_test, self.N_T, sysmdl_n]).to(self.device)
        HPH_T = torch.zeros([self.N_T*sysmdl_T_test,sysmdl_n, sysmdl_n]).to(self.device)

        # Init Hidden State
        if self.args.hnet_arch == "GRU":
                    self.hnet.init_hidden()
        self.mnet.init_hidden()
        # Init Sequence
        self.mnet.InitSequence(test_init, sysmdl_T_test)               

        weights_cm_shift, weights_cm_gain = self.hnet(SoW_old)
        for t in range(0, sysmdl_T_test):
            x_out[:,:, t] = torch.squeeze(self.mnet(torch.unsqueeze(test_input[:,:, t],2), weights_cm_gain=weights_cm_gain, weights_cm_shift=weights_cm_shift))
            KG_array[t] = torch.squeeze(self.mnet.KGain)  
            dy_array[t] = torch.squeeze(self.mnet.dy)

        ### Estimate R ##################################################################
        y_hat = torch.zeros_like(test_input)
        for t in range(sysmdl_T_test):
            y_hat[:,:,t] = torch.squeeze(sys_model.h(torch.unsqueeze(x_out[:,:,t],2)))
        residual = test_input - y_hat
        # Compute 2nd moment of residual
        residual_transposed = residual.transpose(1, 2) #[self.N_T, sysmdl_T_test, sysmdl_n]
        residual_reshaped = residual_transposed.reshape(-1, sysmdl_n) # [(self.N_T * sysmdl_T_test), sysmdl_n]
        residual_2nd_moment = torch.einsum('bi,bj->bij', residual_reshaped, residual_reshaped) #[(self.N_T * sysmdl_T_test), sysmdl_n, sysmdl_n]
        residual_2nd_moment = residual_2nd_moment.mean(dim=0)
        # Compute HPH^T
        i = 0
        for seq in range(self.N_T):
            for t in range(sysmdl_T_test):
                KG_t_seq = KG_array[t, seq, :, :] # [m, n]
                HPH_T[i] = torch.inverse(torch.eye(sysmdl_n) - sys_model.H @ KG_t_seq) @ sys_model.H @ KG_t_seq @ R_old
                i += 1
        HPH_T = HPH_T.mean(dim=0) # [n, n]
        R_est = residual_2nd_moment + HPH_T
        R_new = self.exp_smoothing(R_old, R_est)
        
        ### Estimate Q ##################################################################
        Q_t = torch.zeros(sysmdl_T_test, sysmdl_m, sysmdl_m)
        for t in range(sysmdl_T_test):
            dy = dy_array[t] # [N_T, n]
            dy_2nd_moment = torch.einsum('bi,bj->bij', dy, dy) # [N_T, n, n]
            dy_2nd_moment = dy_2nd_moment.mean(dim=0) # [n, n]
            KG_t = KG_array[t] # [N_T, m, n]
            KG = KG_t.mean(dim=0) # [m, n]
            Q_t[t] = torch.squeeze(KG @ dy_2nd_moment @ (KG.T))

        Q_est = Q_t.mean(dim=0)
        Q_new = self.exp_smoothing(Q_old, Q_est)

        return R_new, Q_new
    
    def estimate_scalar(self, Q, Q0):
        # minimize ||Q - q2*Q0||_F
        q2 = torch.trace(Q.T @ Q0) / torch.trace(Q0.T @ Q0)
        return q2
    
    def update_SoW(self, SoW_range_dB, Qk, Rk, Q0, R0):
        q2 = self.estimate_scalar(Qk, Q0)
        print("Estimated q2:", q2)
        r2 = self.estimate_scalar(Rk, R0)
        print("Estimated r2:", r2)

        min_SoW, max_SoW = SoW_range_dB
        min_SoW = 10**(min_SoW/10)
        max_SoW = 10**(max_SoW/10)
        SoW_new = q2 / r2
        if SoW_new < min_SoW:
            SoW_new = min_SoW
        elif SoW_new > max_SoW:
            SoW_new = max_SoW
        return SoW_new
    
    # def lock_SoW(self, SoW_range_dB, SoW_old, Q0, R0, sys_model, test_input, path_results, test_init):

    #     R_new, Q_new = self.innovation_based_estimation(SoW_old,Q0,R0,Q0,R0,sys_model, test_input, path_results, test_init)
    #     SoW_new = self.update_SoW(SoW_range_dB, Q_new, R_new, Q0, R0)
    #     iter = 0
    #     while abs(SoW_new - SoW_old) > self.SoW_conv_error and iter < self.max_iter:
    #         print("SoW_old:", 10*torch.log10(SoW_old), "[dB]", "SoW_new:", 10*torch.log10(SoW_new), "[dB]")
    #         SoW_old = SoW_new
    #         Q_old = Q_new
    #         R_old = R_new
    #         R_new, Q_new = self.innovation_based_estimation(SoW_old,Q_old,R_old,Q0,R0,sys_model, test_input, path_results, test_init)
    #         SoW_new = self.update_SoW(SoW_range_dB, Q_new, R_new, Q0, R0)
    #         iter += 1

    #     return SoW_new, R_new, Q_new

        


