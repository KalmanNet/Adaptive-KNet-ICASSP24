"""
This file contains the class Pipeline_EKF, 
which is used to train and test KalmanNet.
"""

import torch
import torch.nn as nn
import random
import time
from Plot import Plot_KF
import math

class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch # Number of Samples in Batch
        self.learningRate = args.lr # Learning Rate
        self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay
        self.alpha = args.alpha # Composition loss factor
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.weights.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, \
        MaskOnState=False, randomInit=False,cv_init=None,train_init=None,\
        train_lengthMask=None,cv_lengthMask=None):

        ### Optional: start training from previous checkpoint
        # model_weights = torch.load(path_results+'knet_best-model.pt', map_location=self.device) 
        # self.model.load_state_dict(model_weights)

        if self.args.wandb_switch: 
            import wandb

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        
        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B
            # Init Hidden State
            self.model.init_hidden()

            # Init Training Batch tensors
            y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            if self.args.randomLength:
                MSE_train_linear_LOSS = torch.zeros([self.N_B])
                MSE_cv_linear_LOSS = torch.zeros([self.N_CV])

            # Randomly select N_B training sequences
            assert self.N_B <= self.N_E # N_B must be smaller than N_E
            n_e = random.sample(range(self.N_E), k=self.N_B)
            ii = 0
            for index in n_e:
                if self.args.randomLength:
                    y_training_batch[ii,:,train_lengthMask[index,:]] = train_input[index,:,train_lengthMask[index,:]]
                    train_target_batch[ii,:,train_lengthMask[index,:]] = train_target[index,:,train_lengthMask[index,:]]
                else:
                    y_training_batch[ii,:,:] = train_input[index]
                    train_target_batch[ii,:,:] = train_target[index]
                ii += 1
            
            # Init Sequence
            if(randomInit):
                train_init_batch = torch.empty([self.N_B, SysModel.m,1]).to(self.device)
                ii = 0
                for index in n_e:
                    train_init_batch[ii,:,0] = torch.squeeze(train_init[index])
                    ii += 1
                self.model.InitSequence(train_init_batch, SysModel.T)
            else:
                self.model.InitSequence(\
                SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_B,1,1), SysModel.T)
            
            # Forward Computation
            for t in range(0, SysModel.T):
                x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t],2)))
            
            # Compute Training Loss
            MSE_trainbatch_linear_LOSS = 0
            if (self.args.CompositionLoss):
                y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T])
                for t in range(SysModel.T):
                    y_hat[:,:,t] = torch.squeeze(SysModel.h(torch.unsqueeze(x_out_training_batch[:,:,t],2)))

                if(MaskOnState):### FIXME: composition loss, y_hat may have different mask with x
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(x_out_training_batch[jj,mask,train_lengthMask[index]], train_target_batch[jj,mask,train_lengthMask[index]])+(1-self.alpha)*self.loss_fn(y_hat[jj,mask,train_lengthMask[index]], y_training_batch[jj,mask,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:                     
                        MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(x_out_training_batch[:,mask,:], train_target_batch[:,mask,:])+(1-self.alpha)*self.loss_fn(y_hat[:,mask,:], y_training_batch[:,mask,:])
                else:# no mask on state
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(x_out_training_batch[jj,:,train_lengthMask[index]], train_target_batch[jj,:,train_lengthMask[index]])+(1-self.alpha)*self.loss_fn(y_hat[jj,:,train_lengthMask[index]], y_training_batch[jj,:,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:                
                        MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(x_out_training_batch, train_target_batch)+(1-self.alpha)*self.loss_fn(y_hat, y_training_batch)
            
            else:# no composition loss
                if(MaskOnState):
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.loss_fn(x_out_training_batch[jj,mask,train_lengthMask[index]], train_target_batch[jj,mask,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch[:,mask,:], train_target_batch[:,mask,:])
                else: # no mask on state
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.loss_fn(x_out_training_batch[jj,:,train_lengthMask[index]], train_target_batch[jj,:,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else: 
                        MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)

            # dB Loss
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            MSE_trainbatch_linear_LOSS.backward(retain_graph=True)

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            self.model.batch_size = self.N_CV
            # Init Hidden State
            self.model.init_hidden()
            with torch.no_grad():

                SysModel.T_test = cv_input.size()[-1] # T_test is the maximum length of the CV sequences

                x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)
                
                # Init Sequence
                if(randomInit):
                    if(cv_init==None):
                        self.model.InitSequence(\
                        SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_CV,1,1), SysModel.T_test)
                    else:
                        self.model.InitSequence(cv_init, SysModel.T_test)                       
                else:
                    self.model.InitSequence(\
                        SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_CV,1,1), SysModel.T_test)

                for t in range(0, SysModel.T_test):
                    x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input[:, :, t],2)))
                
                # Compute CV Loss
                MSE_cvbatch_linear_LOSS = 0
                if(MaskOnState):
                    if self.args.randomLength:
                        for index in range(self.N_CV):
                            MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,mask,cv_lengthMask[index]], cv_target[index,mask,cv_lengthMask[index]])
                        MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
                    else:          
                        MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch[:,mask,:], cv_target[:,mask,:])
                else:
                    if self.args.randomLength:
                        for index in range(self.N_CV):
                            MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,:,cv_lengthMask[index]], cv_target[index,:,cv_lengthMask[index]])
                        MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
                    else:
                        MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

                # dB Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                
                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    # Save the model weights to a file
                    torch.save(self.model.state_dict(), path_results + 'knet_best-model.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")
                      
            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")
            
            ### Optinal: record loss on wandb
            if self.args.wandb_switch:
                wandb.log({
                    "train_loss": self.MSE_train_dB_epoch[ti],
                    "val_loss": self.MSE_cv_dB_epoch[ti]})
            ###
        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False,\
     randomInit=False,test_init=None,load_model=False,load_model_path=None,\
        test_lengthMask=None):
        if self.args.wandb_switch: 
            import wandb
        # Load model weights
        if load_model:
            model_weights = torch.load(load_model_path, map_location=self.device) 
        else:
            model_weights = torch.load(path_results+'knet_best-model.pt', map_location=self.device) 
        # Set the loaded weights to the model
        # FIXME: if not NNTrain before, the model is not defined
        self.model.load_state_dict(model_weights)

        self.N_T = test_input.shape[0]
        SysModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, SysModel.m,SysModel.T_test]).to(self.device)

        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        # Init Hidden State
        self.model.init_hidden()
        torch.no_grad()

        start = time.time()

        if (randomInit):
            self.model.InitSequence(test_init, SysModel.T_test)               
        else:
            self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_T,1,1), SysModel.T_test)         
        
        for t in range(0, SysModel.T_test):
            x_out_test[:,:, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:,:, t],2)))
        
        end = time.time()
        t = end - start

        # MSE loss
        for j in range(self.N_T):# cannot use batch due to different length and std computation  
            if(MaskOnState):
                if self.args.randomLength:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,mask,test_lengthMask[j]], test_target[j,mask,test_lengthMask[j]]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,mask,:], test_target[j,mask,:]).item()
            else:
                if self.args.randomLength:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,:,:], test_target[j,:,:]).item()
        
        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        ### Optinal: record loss on wandb
        if self.args.wandb_switch:
            wandb.log({f'averaged test loss':self.MSE_test_dB_avg})
        ###

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]
    
    def NNTrain_mixdatasets(self, SoW_train_range, SysModel, cv_input_tuple, cv_target_tuple, train_input_tuple, train_target_tuple, path_results, \
        cv_init, train_init, MaskOnState=False, train_lengthMask=None,cv_lengthMask=None):

        ### Optional: start training from previous checkpoint
        # model_weights = torch.load(path_results+'knet_best-model.pt', map_location=self.device) 
        # self.model.load_state_dict(model_weights)

        if self.args.wandb_switch: 
            import wandb
        
        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        # Init MSE Loss
        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])
        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])

        # dataset size
        for i in SoW_train_range[:-1]:# except the last one
            assert(train_target_tuple[i][0].shape[1]==train_target_tuple[i+1][0].shape[1])
            assert(train_input_tuple[i][0].shape[1]==train_input_tuple[i+1][0].shape[1])
            assert(train_input_tuple[i][0].shape[2]==train_input_tuple[i+1][0].shape[2])
            # check all datasets have the same m, n, T   
        sysmdl_m = train_target_tuple[0][0].shape[1] # state x dimension
        sysmdl_n = train_input_tuple[0][0].shape[1] # input y dimension
        sysmdl_T = train_input_tuple[0][0].shape[2] # sequence length 
        
        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        ##############
        ### Epochs ###
        ##############

        for ti in range(0, self.N_steps):
            # each turn, go through all datasets
            #################
            ### Training  ###
            #################    
            self.optimizer.zero_grad()        
            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B
            MSE_trainbatch_linear_LOSS = torch.zeros([len(train_target_tuple)]) # loss for each dataset
            
            for i in SoW_train_range: # dataset i 
        
               # Init Training Batch tensors
                y_training_batch = torch.zeros([self.N_B, sysmdl_n, sysmdl_T]).to(self.device)
                train_target_batch = torch.zeros([self.N_B, sysmdl_m, sysmdl_T]).to(self.device)
                x_out_training_batch = torch.zeros([self.N_B, sysmdl_m, sysmdl_T]).to(self.device)
                if self.args.randomLength:
                    MSE_train_linear_LOSS = torch.zeros([self.N_B])
                # Init Sequence
                train_init_batch = torch.empty([self.N_B, sysmdl_m,1]).to(self.device)
                # Init Hidden State
                self.model.init_hidden()  
                # SoW: make sure SoWs are consistent
                assert torch.allclose(cv_input_tuple[i][1], cv_target_tuple[i][1]) 
                assert torch.allclose(train_input_tuple[i][1], train_target_tuple[i][1]) 
                self.model.UpdateSystemDynamics(SysModel[i])
                # req grad
                train_input_tuple[i][1].requires_grad = True # SoW_train
                train_input_tuple[i][0].requires_grad = True # input y
                train_target_tuple[i][0].requires_grad = True # target x
                train_init[i].requires_grad = True # init x0
                # data size
                self.N_E = len(train_input_tuple[i][0]) # Number of Training Sequences
                # mask on state
                if MaskOnState:
                    mask = torch.tensor([True,False,False])
                    if sysmdl_m == 2: 
                        mask = torch.tensor([True,False])
                # Randomly select N_B training sequences
                assert self.N_B <= self.N_E # N_B must be smaller than N_E
                n_e = random.sample(range(self.N_E), k=self.N_B)
                dataset_index = 0
                for index in n_e:
                    # Training Batch
                    if self.args.randomLength:
                        y_training_batch[dataset_index,:,train_lengthMask[i][index,:]] = train_input_tuple[i][0][index,:,train_lengthMask[index,:]]
                        train_target_batch[dataset_index,:,train_lengthMask[i][index,:]] = train_target_tuple[i][0][index,:,train_lengthMask[index,:]]
                    else:
                        y_training_batch[dataset_index,:,:] = train_input_tuple[i][0][index]
                        train_target_batch[dataset_index,:,:] = train_target_tuple[i][0][index]                                 
                    # Init Sequence
                    train_init_batch[dataset_index,:,0] = torch.squeeze(train_init[i][index])                  
                    dataset_index += 1
                self.model.InitSequence(train_init_batch, sysmdl_T)
                
                # Forward Computation
                for t in range(0, sysmdl_T):
                    x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t],2)))
                
                # Compute Training Loss
                if (self.args.CompositionLoss):
                    y_hat = torch.zeros([self.N_B, sysmdl_n, sysmdl_T])
                    for t in range(sysmdl_T):
                        y_hat[:,:,t] = torch.squeeze(SysModel[i].h(torch.unsqueeze(x_out_training_batch[:,:,t],2)))

                    if(MaskOnState):### FIXME: composition loss, y_hat may have different mask with x
                        if self.args.randomLength:
                            jj = 0
                            for index in n_e:# mask out the padded part when computing loss
                                MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(x_out_training_batch[jj,mask,train_lengthMask[index]], train_target_batch[jj,mask,train_lengthMask[index]])+(1-self.alpha)*self.loss_fn(y_hat[jj,mask,train_lengthMask[index]], y_training_batch[jj,mask,train_lengthMask[index]])
                                jj += 1
                            MSE_trainbatch_linear_LOSS[i] = torch.mean(MSE_train_linear_LOSS)
                        else:                     
                            MSE_trainbatch_linear_LOSS[i] = self.alpha * self.loss_fn(x_out_training_batch[:,mask,:], train_target_batch[:,mask,:])+(1-self.alpha)*self.loss_fn(y_hat[:,mask,:], y_training_batch[:,mask,:])
                    else:# no mask on state
                        if self.args.randomLength:
                            jj = 0
                            for index in n_e:# mask out the padded part when computing loss
                                MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(x_out_training_batch[jj,:,train_lengthMask[index]], train_target_batch[jj,:,train_lengthMask[index]])+(1-self.alpha)*self.loss_fn(y_hat[jj,:,train_lengthMask[index]], y_training_batch[jj,:,train_lengthMask[index]])
                                jj += 1
                            MSE_trainbatch_linear_LOSS[i] = torch.mean(MSE_train_linear_LOSS)
                        else:                
                            MSE_trainbatch_linear_LOSS[i] = self.alpha * self.loss_fn(x_out_training_batch, train_target_batch)+(1-self.alpha)*self.loss_fn(y_hat, y_training_batch)
                
                else:# no composition loss
                    if(MaskOnState):
                        if self.args.randomLength:
                            jj = 0
                            for index in n_e:# mask out the padded part when computing loss
                                MSE_train_linear_LOSS[jj] = self.loss_fn(x_out_training_batch[jj,mask,train_lengthMask[index]], train_target_batch[jj,mask,train_lengthMask[index]])
                                jj += 1
                            MSE_trainbatch_linear_LOSS[i] = torch.mean(MSE_train_linear_LOSS)
                        else:
                            MSE_trainbatch_linear_LOSS[i] = self.loss_fn(x_out_training_batch[:,mask,:], train_target_batch[:,mask,:])
                    else: # no mask on state
                        if self.args.randomLength:
                            jj = 0
                            for index in n_e:# mask out the padded part when computing loss
                                MSE_train_linear_LOSS[jj] = self.loss_fn(x_out_training_batch[jj,:,train_lengthMask[index]], train_target_batch[jj,:,train_lengthMask[index]])
                                jj += 1
                            MSE_trainbatch_linear_LOSS[i] = torch.mean(MSE_train_linear_LOSS)
                        else: 
                            MSE_trainbatch_linear_LOSS[i] = self.loss_fn(x_out_training_batch, train_target_batch)

            # averaged Loss over all datasets           
            MSE_trainbatch_linear_LOSS_average = MSE_trainbatch_linear_LOSS.sum() / len(SoW_train_range)                         
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS_average.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################
            MSE_trainbatch_linear_LOSS_average.backward(retain_graph=True)
            self.optimizer.step()

            #################################
            ### Validation Sequence Batch ###
            #################################
            # Cross Validation Mode
            self.model.eval()
            # data size
            self.N_CV = len(cv_input_tuple[i][0])
            sysmdl_T_test = cv_input_tuple[i][0].shape[2] 
            # loss for each dataset
            MSE_cvbatch_linear_LOSS = torch.zeros([len(cv_target_tuple)])                     
            # Update Batch Size
            self.model.batch_size = self.N_CV 

            with torch.no_grad():
                for i in SoW_train_range: # dataset i 
                    if self.args.randomLength:
                        MSE_cv_linear_LOSS = torch.zeros([self.N_CV])
                    # Init Output
                    x_out_cv_batch = torch.empty([self.N_CV, sysmdl_m, sysmdl_T_test]).to(self.device)

                    # Init Hidden State
                    self.model.init_hidden()              
                    
                    # Init Sequence                    
                    self.model.InitSequence(cv_init[i], sysmdl_T_test)                       
                    
                    for t in range(0, sysmdl_T_test):
                        x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input_tuple[i][0][:, :, t],2)))
                    
                    # Compute CV Loss
                    if(MaskOnState):
                        if self.args.randomLength:
                            for index in range(self.N_CV):
                                MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,mask,cv_lengthMask[i][index]], cv_target_tuple[i][0][index,mask,cv_lengthMask[index]])
                            MSE_cvbatch_linear_LOSS[i] = torch.mean(MSE_cv_linear_LOSS)
                        else:          
                            MSE_cvbatch_linear_LOSS[i] = self.loss_fn(x_out_cv_batch[:,mask,:], cv_target_tuple[i][0][:,mask,:])
                    else:
                        if self.args.randomLength:
                            for index in range(self.N_CV):
                                MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,:,cv_lengthMask[i][index]], cv_target_tuple[i][0][index,:,cv_lengthMask[index]])
                            MSE_cvbatch_linear_LOSS[i] = torch.mean(MSE_cv_linear_LOSS)
                        else:
                            MSE_cvbatch_linear_LOSS[i] = self.loss_fn(x_out_cv_batch, cv_target_tuple[i][0])

                # Print loss for each dataset in train range    
                for i in SoW_train_range:
                    MSE_cvbatch_dB_LOSS_i = 10 * math.log10(MSE_cvbatch_linear_LOSS[i].item())
                    print(f"MSE Validation on dataset {i}:", MSE_cvbatch_dB_LOSS_i,"[dB]")
                
                # averaged dB Loss
                MSE_cvbatch_linear_LOSS = MSE_cvbatch_linear_LOSS.sum() / len(SoW_train_range)
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                # save model with best averaged loss on all datasets
                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    
                    torch.save(self.model.state_dict(), path_results + 'knet_best-model.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training Average:", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")
                      
            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")
            
            ### Optinal: record loss on wandb
            if self.args.wandb_switch:
                wandb.log({
                    "train_loss": self.MSE_train_dB_epoch[ti],
                    "val_loss": self.MSE_cv_dB_epoch[ti]})
            ###
        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]


    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot_KF(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)