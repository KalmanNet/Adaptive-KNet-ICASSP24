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
args.use_cuda = False # use GPU or not
if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
      torch.set_default_tensor_type(torch.cuda.FloatTensor)
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

### dataset parameters ##################################################
F = F.to(device)
H = H.to(device)
Q_structure = Q_structure.to(device)
R_structure = R_structure.to(device)
m1_0 = m1_0.to(device)

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
   args.init_distri = 'normal' # 'uniform' or 'normal'
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
args.wandb_switch = False
if args.wandb_switch:
   import wandb
   wandb.init(project="HKNet_Linear")
args.n_steps = 2
args.n_batch = 100 # will be multiplied by num of datasets
args.lr = 1e-5
args.wd = 1e-3

### True model ##################################################
# SoW
SoW = torch.tensor([[0,0,1,1], [0,0,1,4], [0,0,1,7], [0,0,1,10], [0,0,1,1.5], [0,0,1,5.5], [0,0,1,9]])
SoW_train_range = [0,1,2,3] # first *** number of datasets are used for training
SoW_test_range = [0,1,2,3,4,5,6] # last *** number of datasets are used for testing
# noise
r2 = SoW[:, 2]
q2 = SoW[:, 3]
for i in range(len(SoW)):
   print(f"SoW of dataset {i}: ", SoW[i])
   print(f"r2 [linear] and q2 [linear] of dataset  {i}: ", r2[i], q2[i])

# model
sys_model = []
for i in range(len(SoW)):
   sys_model_i = SystemModel(F, q2[i]*Q_structure, H, r2[i]*R_structure, args.T, args.T_test, SoW[i])
   sys_model_i.InitSequence(m1_0, m2_0)
   sys_model.append(sys_model_i)

### paths ##################################################
path_results = 'simulations/linear_canonical/results/'
dataFolderName = 'data/linear_canonical/r2=1' + '/'
dataFileName = []
for i in range(len(SoW)):
   dataFileName.append('r2=' + str(r2[i].item())+"_" +"q2="+ str(q2[i].item())+ '.pt')
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
## Build Neural Networks
print("Build HNet and KNet")
KalmanNet_model = KalmanNetNN()
weight_size = KalmanNet_model.NNBuild(sys_model[0], args)
print("Number of parameters for KalmanNet:", weight_size)
HyperNet_model = HyperNetwork(args, weight_size)
weight_size_hnet = sum(p.numel() for p in HyperNet_model.parameters() if p.requires_grad)
print("Number of parameters for HyperNet:", weight_size_hnet)
print("Total number of parameters:", weight_size + weight_size_hnet)
## Set up pipeline
hknet_pipeline = Pipeline_hknet(strTime, "pipelines", "hknet")
hknet_pipeline.setModel(HyperNet_model, KalmanNet_model)
hknet_pipeline.setTrainingParams(args)
## Optinal: record parameters to wandb
if args.wandb_switch:
   wandb.log({
   "total_params": weight_size + weight_size_hnet,
   "batch_size": args.n_batch,
   "learning_rate": args.lr,  
   "weight_decay": args.wd})
## Train Neural Networks
if args.randomLength:
   hknet_pipeline.NNTrain_mixdatasets(SoW_train_range, sys_model, cv_input_list, cv_target_list, train_input_list, train_target_list, path_results, cv_init_list,train_init_list,train_lengthMask=train_lengthMask_list,cv_lengthMask=cv_lengthMask_list)
else:
   hknet_pipeline.NNTrain_mixdatasets(SoW_train_range, sys_model, cv_input_list, cv_target_list, train_input_list, train_target_list, path_results,cv_init_list,train_init_list)

## Test Neural Networks for each dataset
if args.randomLength:
   hknet_pipeline.NNTest_alldatasets(SoW_test_range, sys_model, test_input_list, test_target_list, path_results,test_init_list,test_lengthMask=test_lengthMask_list)
else:    
   hknet_pipeline.NNTest_alldatasets(SoW_test_range, sys_model, test_input_list, test_target_list, path_results,test_init_list)

# ## Save pipeline
# hknet_pipeline.save()
## Close wandb run
if args.wandb_switch: 
   wandb.finish() 