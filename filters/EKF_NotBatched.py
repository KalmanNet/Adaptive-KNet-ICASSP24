"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch

from simulations.lorenz_attractor.parameters import Origin_getJacobian as getJacobian

class ExtendedKalmanFilter:

    def __init__(self, SystemModel, args):
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.f = SystemModel.f
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q.to(self.device)

        self.h = SystemModel.h
        self.n = SystemModel.n

        # Has to be transofrmed because of EKF non-linearity
        self.R = SystemModel.R.to(self.device)

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # Pre allocate KG array
        self.KG_array = torch.zeros((self.T_test,self.m,self.n))
   
    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior)).to(self.device)
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(self.m1x_posterior,self.f), getJacobian(self.m1x_prior, self.h))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.squeeze(self.h(self.m1x_prior))
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

        #Save KalmanGain
        self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    def UpdateJacobians(self, F, H):
        self.F = F.to(self.device)
        self.F_T = torch.transpose(F,0,1)
        self.H = H.to(self.device)
        self.H_T = torch.transpose(H,0,1)
        #print(self.H,self.F,'\n')
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, T):
        y = y.to(self.device)
        # Pre allocate an array for predicted state and variance
        self.x = torch.zeros(size=[self.m, T]).to(self.device)
        self.sigma = torch.zeros(size=[self.m, self.m, T]).to(self.device)
        # Pre allocate KG array
        self.KG_array = torch.zeros((T,self.m,self.n)).to(self.device)
        self.i = 0 # Index for KG_array alocation

        self.m1x_posterior = torch.squeeze(self.m1x_0).to(self.device)
        self.m2x_posterior = torch.squeeze(self.m2x_0).to(self.device)

        for t in range(0, T):
            yt = torch.squeeze(y[:, t])
            xt,sigmat = self.Update(yt)
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)