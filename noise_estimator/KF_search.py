import torch
import torch.nn as nn
import time
import math
import random
from filters.Linear_KF import KalmanFilter

class KF_NE:

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

    def setParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.forget_factor = args.forget_factor

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

    def linear_innovation_based_estimation(self, sys_model, test_input, test_init):
        # data size
        self.N_T = test_input.shape[0]
        sysmdl_T_test = test_input.size()[2]
        sysmdl_m = sys_model.m
        sysmdl_n = test_input.size()[1]

        # Perform KF
            # allocate memory for KF output
        KF_out = torch.zeros(self.N_T, sysmdl_m, sysmdl_T_test)
        KF = KalmanFilter(sys_model, self.args)
            # Init and Forward Computation 
        KF.Init_batched_sequence(test_init, sys_model.m2x_0.view(1,sysmdl_m,sysmdl_m).expand(self.N_T,-1,-1))                        
        KF.GenerateBatch(test_input)
        KF_out = KF.x
      
        ### Estimate R ##################################################################
        y_hat = torch.zeros_like(test_input)
        for t in range(sysmdl_T_test):
            y_hat[:,:,t] = torch.squeeze(sys_model.h(torch.unsqueeze(KF_out[:,:,t],2)))
        residual = test_input - y_hat

        temp_moment = torch.zeros(self.N_T*sysmdl_T_test,sysmdl_n, sysmdl_n)
        i = 0
        for seq in range(self.N_T):
            for t in range(sysmdl_T_test):
                temp_residual = residual[seq,:,t].unsqueeze(1)
                temp_moment[i] = temp_residual * temp_residual.T

                # method 1
                HPH_T = KF.m2y_minus_R_list[t]
                HPH_T = HPH_T[seq]
                # print("HPH_T original:", HPH_T)

                # method 2
                # KG = KF.KG_list[t] # [N_T, m, n]
                # KG = KG[seq] # [m, n]
                # HPH_T = torch.inverse(torch.eye(sysmdl_n) - sys_model.H @ KG) 
                # HPH_T = HPH_T @ sys_model.H @ KG @ sys_model.R
                # print("HPH_T use KG:", HPH_T)

                temp_moment[i] = temp_moment[i] + HPH_T
                i += 1
        R_est = temp_moment.mean(dim=0)
        

        ### Estimate Q ##################################################################
        Q_t = torch.zeros(sysmdl_T_test, sysmdl_m, sysmdl_m)
        for t in range(sysmdl_T_test):
            dy = torch.squeeze(KF.dy_list[t]) # [N_T, n]
            dy_2nd_moment = torch.einsum('bi,bj->bij', dy, dy) # [N_T, n, n]
            dy_2nd_moment = dy_2nd_moment.mean(dim=0) # [n, n]
            KG = KF.KG_list[t] # [N_T, m, n]
            KG = KG.mean(dim=0) # [m, n]
            Q_t[t] = torch.squeeze(KG @ dy_2nd_moment @ KG.T)

        Q_est = Q_t.mean(dim=0)

        return R_est, Q_est

    def estimate_scalar(self, Q, Q0):
        # minimize ||Q - q2*Q0||_F
        q2 = torch.trace(Q.T @ Q0) / torch.trace(Q0.T @ Q0)
        return q2


