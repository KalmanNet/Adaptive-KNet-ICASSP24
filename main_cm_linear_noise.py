import torch
import torch.nn as nn
from datetime import datetime

from simulations.Linear_sysmdl import SystemModel
from simulations.utils import DataGen
import simulations.config as config
from simulations.linear_canonical.parameters import F, H, Q_structure_nonid, R_structure_nonid,\
   m, m1_0

from filters.KalmanFilter_test import KFTest

from hnets.hnet import HyperNetwork
from hnets.hnet_deconv import hnet_deconv
from mnets.KNet_mnet import KalmanNetNN as KNet_mnet

from pipelines.Pipeline_cm import Pipeline_cm
from pipelines.Pipeline_EKF import Pipeline_EKF

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
Q_structure = Q_structure_nonid.to(device)
R_structure = R_structure_nonid.to(device)
m1_0 = m1_0.to(device)

args.N_E = 1000
args.N_CV = 100
args.N_T = 1000
# deterministic initial condition
m2_0 = 0 * torch.eye(m)
# sequence length
args.T = 100
args.T_test = 100
train_lengthMask = None
cv_lengthMask = None
test_lengthMask = None
# determine noise distribution normal/exp (DEFAULT: "normal")
args.meas_noise_distri = "normal"

### training parameters ##################################################
args.wandb_switch = False
if args.wandb_switch:
   import wandb
   wandb.init(project="HKNet_Linear")
args.knet_trainable = True
# training parameters for KNet
args.in_mult_KNet = 1
args.out_mult_KNet = 1
args.n_steps = 5000
args.n_batch = 100 
args.lr = 1e-4
args.wd = 1e-3
# training parameters for Hypernet
args.hnet_arch = "GRU" # "deconv" or "GRU
if args.hnet_arch == "GRU": # settings for GRU hnet
   args.hnet_hidden_size_discount = 100
elif args.hnet_arch == "deconv": # settings for deconv hnet
   # 2x2 system
   embedding_dim = 4
   hidden_channel_dim = 32
else:
   raise Exception("args.hnet_arch not recognized")
n_steps = 10000
n_batch = 100 # will be multiplied by num of datasets
lr = 1e-3
wd = 1e-3

### True model ##################################################
# SoW (state of world) = [distribution of R, distribution of Q, r2, q2]
SoW = torch.tensor([[10,-10], [1,-10], [-10,-10],[-20,-10]])
SoW_train_range = list(range(len(SoW))) # first *** number of datasets are used for training
print("SoW_train_range: ", SoW_train_range)
SoW_test_range = list(range(len(SoW))) # last *** number of datasets are used for testing
# noise
r2_dB = SoW[:, 0]
q2_dB = SoW[:, 1]

r2 = 10 ** (r2_dB/10)
q2 = 10 ** (q2_dB/10)
for i in range(len(SoW)):
   print(f"SoW of dataset {i}: ", SoW[i])
   print(f"r2 [linear] and q2 [linear] of dataset  {i}: ", r2[i], q2[i])

# model
sys_model = []
for i in range(len(SoW)):
   sys_model_i = SystemModel(F, q2[i]*Q_structure, H, r2[i]*R_structure, args.T, args.T_test, q2[i], r2[i])
   sys_model_i.InitSequence(m1_0, m2_0)
   sys_model.append(sys_model_i)

### paths ##################################################
path_results = 'simulations/linear_canonical/results/5x5/'
dataFolderName = 'data/linear_canonical/5x5/non-diag' + '/'
dataFileName = []
for i in range(len(SoW)):
   dataFileName.append('r2=' + str(r2_dB[i].item())+"dB"+"_" +"q2="+ str(q2_dB[i].item())+"dB" + '.pt')
###################################
### Data Loader (Generate Data) ###
###################################
print("Start Data Gen")
for i in range(len(SoW)):
   DataGen(args, sys_model[i], dataFolderName + dataFileName[i])
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

for i in range(len(SoW)):  
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

##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter True")
for i in range(len(SoW)):
   test_input = test_input_list[i][0]
   test_target = test_target_list[i][0]
   test_init = test_init_list[i][0]  
   test_lengthMask = None 
   print(f"Dataset {i}") 
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model[i], test_input, test_target, test_lengthMask=test_lengthMask)


##################################
### Hyper - KalmanNet Pipeline ###
##################################
### train and test KalmanNet on dataset i
i = 0
print(f"KalmanNet pipeline start, train on dataset {i}")
KalmanNet_model = KNet_mnet()
KalmanNet_model.NNBuild(sys_model[i], args)
print("Number of trainable parameters for KalmanNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
## Train Neural Network
KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model[i])
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(args)
KalmanNet_Pipeline.NNTrain(sys_model[i], cv_input_list[i][0], cv_target_list[i][0], train_input_list[i][0], train_target_list[i][0], path_results)
for i in range(len(SoW)):
   test_input = test_input_list[i][0]
   test_target = test_target_list[i][0]
   test_init = test_init_list[i][0]  
   test_lengthMask = None 
   print(f"Dataset {i}") 
   KalmanNet_Pipeline.NNTest(sys_model[i], test_input_list[i][0], test_target_list[i][0], path_results)

### frozen KNet weights, train hypernet to generate CM weights on multiple datasets
# load frozen weights
frozen_weights = torch.load(path_results + 'knet_best-model.pt', map_location=device) 
### frozen KNet weights, train hypernet to generate CM weights on multiple datasets
args.knet_trainable = False # frozen KNet weights
args.use_context_mod = True # use CM
## training parameters for Hypernet
args.n_steps = n_steps
args.n_batch = n_batch # will be multiplied by num of datasets
args.lr = lr
args.wd = wd
## Build Neural Networks
print("Build HNet and KNet")
KalmanNet_model = KNet_mnet()
cm_weight_size = KalmanNet_model.NNBuild(sys_model[0], args, frozen_weights=frozen_weights)
print("Number of CM parameters:", cm_weight_size)

# Split into gain and shift
cm_weight_size = torch.tensor([cm_weight_size / 2]).int().item()

if args.hnet_arch == "deconv":
   HyperNet_model = hnet_deconv(args, 1, cm_weight_size, embedding_dim=embedding_dim, hidden_channel_dim = hidden_channel_dim)
   weight_size_hnet = HyperNet_model.print_num_weights()
elif args.hnet_arch == "GRU":
   HyperNet_model = HyperNetwork(args, 1, cm_weight_size)
   weight_size_hnet = sum(p.numel() for p in HyperNet_model.parameters() if p.requires_grad)
   print("Number of parameters for HyperNet:", weight_size_hnet)
else:
   raise ValueError("Unknown hnet_arch")

## Set up pipeline
hknet_pipeline = Pipeline_cm(strTime, "pipelines", "hknet")
hknet_pipeline.setModel(HyperNet_model, KalmanNet_model)
hknet_pipeline.setTrainingParams(args)
## Optinal: record parameters to wandb
if args.wandb_switch:
   wandb.log({
   "total_params": cm_weight_size + weight_size_hnet,
   "batch_size": args.n_batch,
   "learning_rate": args.lr,  
   "weight_decay": args.wd})
## Train Neural Networks
hknet_pipeline.NNTrain_mixdatasets(SoW_train_range, sys_model, cv_input_list, cv_target_list, train_input_list, train_target_list, path_results,cv_init_list,train_init_list)

## Test Neural Networks for each dataset  
hknet_pipeline.NNTest_alldatasets(SoW_test_range, sys_model, test_input_list, test_target_list, path_results,test_init_list)



## Close wandb run
if args.wandb_switch: 
   wandb.finish() 