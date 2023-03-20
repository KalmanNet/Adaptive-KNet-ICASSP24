import torch
import torch.nn as nn
from datetime import datetime

from simulations.Linear_sysmdl import SystemModel
from simulations.utils import DataGen
import simulations.config as config
from simulations.linear_canonical.parameters import F, H, Q_structure, R_structure,\
   m, m1_0

from filters.KalmanFilter_test import KFTest

from hnets.hnet import HyperNetwork
from mnets.KNet_mnet import KalmanNetNN

from pipelines.Pipeline_hknet import Pipeline_hknet

print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

#########################
### Parameter Setting ###
#########################
args = config.general_settings()

### dataset parameters ##################################################
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
# init condition
args.randomInit_train = False
args.randomInit_cv = False
args.randomInit_test = False
if args.randomInit_train or args.randomInit_cv or args.randomInit_test:
   # you can modify initial variance
   args.variance = 1
   args.distribution = 'normal' # 'uniform' or 'normal'
   m2_0 = args.variance * torch.eye(m)
else: 
   # deterministic initial condition
   m2_0 = 0 * torch.eye(m) 
# sequence length
args.T = 100
args.T_test = 100
args.randomLength = False
if args.randomLength:# you can modify T_max and T_min 
   args.T_max = 1000
   args.T_min = 100
   # set T and T_test to T_max for convenience of batch calculation
   args.T = args.T_max 
   args.T_test = args.T_max
else:
   train_lengthMask = None
   cv_lengthMask = None
   test_lengthMask = None

### training parameters ##################################################
args.use_cuda = False # use GPU or not

mixed_dataset = True # use mixed dataset or one-by-one
if mixed_dataset:
   args.n_steps = 10 # switch dataset every *** steps
   n_turns = 500 # number of turns, each turn len(SoW) datasets are used 
else:
   args.n_steps = 2500 # for each dataset

args.n_batch = 30
args.lr = 1e-4
args.wd = 1e-3

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

### True model ##################################################
# SoW
SoW = torch.tensor([[0,0,0,0], [0,0,1,1]]).float()

# noise
r2 = torch.tensor([1, 1e-3]).float()
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
print("1/r2 [dB] and 1/q2 [dB] of dataset 1: ", 10 * torch.log10(1/r2[0]), 10 * torch.log10(1/q2[0]))
print("1/r2 [dB] and 1/q2 [dB] of dataset 2: ", 10 * torch.log10(1/r2[1]), 10 * torch.log10(1/q2[1]))

# model
sys_model = []
for i in range(len(SoW)):
   sys_model_i = SystemModel(F, q2[i]*Q_structure, H, r2[i]*R_structure, args.T, args.T_test, SoW[i])
   sys_model_i.InitSequence(m1_0, m2_0)
   sys_model.append(sys_model_i)

### paths ##################################################
path_results = 'simulations/linear_canonical/results/'
dataFolderName = 'data/Linear_canonical' + '/'
dataFileName = ('2x2_rq020_T100.pt', '2x2_rq3050_T100.pt')

###################################
### Data Loader (Generate Data) ###
###################################
# print("Start Data Gen")
# for i in range(len(SoW)):
#    DataGen(args, sys_model[i], dataFolderName + dataFileName[i])
print("Data Load")
train_input_list = []
train_target_list = []
cv_input_list = []
cv_target_list = []
test_input_list = []
test_target_list = []
train_init_list = []
cv_init_list = []
test_init_list = []
if args.randomLength:
   train_lengthMask_list = []
   cv_lengthMask_list = []
   test_lengthMask_list = []

for i in range(len(SoW)):
   if args.randomLength:
      [train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init, train_lengthMask,cv_lengthMask,test_lengthMask] = torch.load(dataFolderName + dataFileName[i], map_location=device)
      train_lengthMask_list.append(train_lengthMask)
      cv_lengthMask_list.append(cv_lengthMask)
      test_lengthMask_list.append(test_lengthMask)
   else:
      [train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init] = torch.load(dataFolderName + dataFileName[i], map_location=device)
   train_input_list.append((train_input, SoW[i]))
   train_target_list.append((train_target, SoW[i]))
   cv_input_list.append((cv_input, SoW[i]))
   cv_target_list.append((cv_target, SoW[i]))
   test_input_list.append((test_input, SoW[i]))
   test_target_list.append((test_target, SoW[i]))
   train_init_list.append(train_init)
   cv_init_list.append(cv_init)
   test_init_list.append(test_init)

########################################
### Evaluate Observation Noise Floor ###
########################################
for i in range(len(SoW)):
   test_input = test_input_list[i][0]
   test_target = test_target_list[i][0]
   loss_obs = nn.MSELoss(reduction='mean')
   MSE_obs_linear_arr = torch.empty(args.N_T)# MSE [Linear]  
   for seq in range(args.N_T):
      MSE_obs_linear_arr[seq] = loss_obs(test_input[seq], test_target[seq]).item()   
   MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
   MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

   # Standard deviation
   MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

   # Confidence interval
   obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

   print(f"Observation Noise Floor for dataset {i} - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
   print(f"Observation Noise Floor for dataset {i} - STD:", obs_std_dB, "[dB]")

##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter True")
for i in range(len(SoW)):
   test_input = test_input_list[i][0]
   test_target = test_target_list[i][0]
   test_init = test_init_list[i][0]
   if args.randomLength:
      test_lengthMask = test_lengthMask_list[i][0]
   else:
      test_lengthMask = None
   
   print(f"Dataset {i}")
   if args.randomInit_test:
      [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model[i], test_input, test_target, randomInit = True, test_init=test_init, test_lengthMask=test_lengthMask)
   else: 
      [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model[i], test_input, test_target, test_lengthMask=test_lengthMask)


##################################
### Hyper - KalmanNet Pipeline ###
##################################
# Build Neural Networks
print("Build HNet and KNet")
KalmanNet_model = KalmanNetNN()
weight_size = KalmanNet_model.NNBuild(sys_model[0], args)
print("Number of parameters for KalmanNet:", weight_size)
HyperNet_model = HyperNetwork(args, weight_size)

## Set up pipeline
hknet_pipeline = Pipeline_hknet(strTime, "pipelines", "hknet")
hknet_pipeline.setModel(HyperNet_model, KalmanNet_model)
hknet_pipeline.setTrainingParams(args)

## Train Neural Networks
if mixed_dataset:
   for turns in range(n_turns):
      for i in range(len(SoW)):
         if args.randomLength:
            hknet_pipeline.NNTrain(sys_model[i], cv_input_list[i], cv_target_list[i], train_input_list[i], train_target_list[i], path_results, cv_init_list[i],train_init_list[i],train_lengthMask=train_lengthMask_list[i],cv_lengthMask=cv_lengthMask_list[i])
         else:
            hknet_pipeline.NNTrain(sys_model[i], cv_input_list[i], cv_target_list[i], train_input_list[i], train_target_list[i], path_results,cv_init_list[i],train_init_list[i])
else:
   for i in range(len(SoW)):  
      if args.randomLength:
         hknet_pipeline.NNTrain(sys_model[i], cv_input_list[i], cv_target_list[i], train_input_list[i], train_target_list[i], path_results, cv_init_list[i],train_init_list[i],train_lengthMask=train_lengthMask_list[i],cv_lengthMask=cv_lengthMask_list[i])
      else:
         hknet_pipeline.NNTrain(sys_model[i], cv_input_list[i], cv_target_list[i], train_input_list[i], train_target_list[i], path_results,cv_init_list[i],train_init_list[i])

## Test Neural Networks for each dataset
for i in range(len(SoW)):
   if args.randomLength:
      hknet_pipeline.NNTest(sys_model[i], test_input_list[i], test_target_list[i], path_results,test_init_list[i],test_lengthMask=test_lengthMask_list[i])
   else:    
      hknet_pipeline.NNTest(sys_model[i], test_input_list[i], test_target_list[i], path_results,test_init_list[i])

## Save pipeline
hknet_pipeline.save()