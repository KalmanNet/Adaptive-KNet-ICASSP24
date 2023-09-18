import torch
from datetime import datetime

from simulations.Linear_sysmdl import SystemModel
import simulations.config as config
import simulations.utils as utils
from simulations.Linear_CA.parameters import F_gen,F_CV,H_identity,H_onlyPos,\
   Q_gen,Q_CV,R_3,R_2,R_onlyPos,\
   m,m_cv

from filters.KalmanFilter_test import KFTest

from hnets.hnet import HyperNetwork
from hnets.hnet_deconv import hnet_deconv
from mnets.KNet_mnet_allCM import KalmanNetNN as KNet_mnet

from pipelines.Pipeline_cm import Pipeline_cm
from pipelines.Pipeline_EKF import Pipeline_EKF

from Plot import Plot_RTS as Plot

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

### Dataset parameters ##################################################
args.N_E = 1000
args.N_CV = 100
args.N_T = 200

args.T = 100
args.T_test = 100

m1x_0 = torch.zeros(m) # Initial State
m1x_0_cv = torch.zeros(m_cv) # Initial State for CV
m2x_0 = 0 * torch.eye(m) # Initial Covariance for feeding to smoothers and RTSNet
m2x_0_cv = 0 * torch.eye(m_cv) # Initial Covariance for CV

### training parameters ##################################################
args.wandb_switch = False
if args.wandb_switch:
   import wandb
   wandb.init(project="HKNet_Linear")

### PVA or P
Loss_On_AllState = True # if false: only calculate test loss on position
Train_Loss_On_AllState = True # if false: only calculate training loss on position
CV_model = False # if true: use CV model, else: use CA model

# training parameters for KNet
args.knet_trainable = True
# depending on m and n, scale the input and output dimension multiplier on the FC layers and LSTM layers of KNet 
args.in_mult_KNet = 40
# args.out_mult_KNet = 40
args.n_steps = 50000
args.n_batch = 32
args.lr = 1e-4
args.wd = 1e-4

# training parameters for Hypernet
args.hnet_arch = "GRU" # "deconv" or "GRU
if args.hnet_arch == "GRU": # settings for GRU hnet
   args.hnet_hidden_size_discount = 10

elif args.hnet_arch == "deconv": # settings for deconv hnet
   embedding_dim = 4
   hidden_channel_dim = 32

else:
   raise Exception("args.hnet_arch not recognized")
n_steps = 50000
n_batch_list = [32]  # will be multiplied by num of datasets
lr = 1e-3
wd = 1e-3

### True model ##################################################
# SoW
SoW = torch.tensor([[10,1], [1,1], [0.1,1],[0.01,1]]) # different q2/r2 ratios
SoW_train_range = list(range(len(SoW))) # these datasets are used for training
n_batch_list = n_batch_list * len(SoW_train_range)
print("SoW_train_range: ", SoW_train_range)
SoW_test_range = list(range(len(SoW))) # these datasets are used for testing
# noise
r2 = SoW[:, 0]
q2 = SoW[:, 1]

# change SoW to q2/r2 ratio
SoW = q2/r2

for i in range(len(SoW)):
   print(f"SoW of dataset {i}: ", SoW[i])
   print(f"r2 [linear] and q2 [linear] of dataset  {i}: ", r2[i], q2[i])

# Generation model, i.e. GT model may use Constant Acceleration Model (CA) or Constant Velocity Model (CV)
# model
sys_model = []
for i in range(len(SoW)):
   sys_model_i = SystemModel(F_gen, q2[i]*Q_gen, H_onlyPos, r2[i]*R_onlyPos, args.T, args.T_test, q2[i], r2[i])
   sys_model_i.InitSequence(m1x_0, m2x_0)# x0 and P0
   sys_model.append(sys_model_i)

# Feed model (to KF, RTS and RTSNet) may use Constant Velocity Model (CV)
sys_model_feed = []
if CV_model:
   H_onlyPos = torch.tensor([[1, 0]]).float()
   sys_model_feed_i = SystemModel(F_CV, q2[i]*Q_CV, H_onlyPos, r2[i]*R_onlyPos, args.T, args.T_test, q2[i], r2[i])
   sys_model_feed_i.InitSequence(m1x_0_cv, m2x_0_cv)# x0 and P0
   sys_model_feed.append(sys_model_feed_i)


### paths ##################################################
path_results = 'simulations/Linear_CA/results/'
DatafolderName = 'data/Linear_CA' + '/'
dataFileName = []
rounding_digits = 4 # round to # digits after decimal point
for i in range(len(SoW)):
   r2_rounded = round(r2[i].item() * 10**rounding_digits) / 10**rounding_digits
   q2_rounded = round(q2[i].item() * 10**rounding_digits) / 10**rounding_digits
   dataFileName.append('r2=' + str(r2_rounded)+"_" +"q2="+ str(q2_rounded)+ '.pt')


###################################
### Data Loader (Generate Data) ###
###################################
# print("Start Data Gen")
# for i in range(len(SoW)):
#    utils.DataGen(args, sys_model[i], DatafolderName + dataFileName[i])

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
   [train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init] = torch.load(DatafolderName + dataFileName[i], map_location=device)
   
   if CV_model:# set state as (p,v) instead of (p,v,a)
      train_target = train_target[:,0:m_cv,:]
      train_init = train_init[:,0:m_cv]
      cv_target = cv_target[:,0:m_cv,:]
      cv_init = cv_init[:,0:m_cv]
      test_target = test_target[:,0:m_cv,:]
      test_init = test_init[:,0:m_cv]
   
   train_input_list.append((train_input, SoW[i]))
   train_target_list.append((train_target, SoW[i]))
   cv_input_list.append((cv_input, SoW[i]))
   cv_target_list.append((cv_target, SoW[i]))
   test_input_list.append((test_input, SoW[i]))
   test_target_list.append((test_target, SoW[i]))
   train_init_list.append(train_init)
   cv_init_list.append(cv_init)
   test_init_list.append(test_init)


print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter True")
KF_out_list = []
for i in range(len(SoW)):
   test_input = test_input_list[i][0]
   test_target = test_target_list[i][0]
   test_init = test_init_list[i] 
   print(f"Dataset {i}") 
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model[i], test_input, test_target, allStates=Loss_On_AllState)
   KF_out_list.append(KF_out)


##########################
### KalmanNet Pipeline ###
##########################
# Build Neural Network
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
# KalmanNet_Pipeline.NNTrain(sys_model[i], cv_input_list[i][0], cv_target_list[i][0], train_input_list[i][0], train_target_list[i][0], path_results)
for i in range(len(SoW)):
   print(f"Dataset {i}") 
   KalmanNet_Pipeline.NNTest(sys_model[i], test_input_list[i][0], test_target_list[i][0], path_results)

### frozen KNet weights, train hypernet to generate CM weights on multiple datasets
# load frozen weights
frozen_weights = torch.load(path_results + 'knet_best-model.pt', map_location=device) 
### frozen KNet weights, train hypernet to generate CM weights on multiple datasets
args.knet_trainable = False # frozen KNet weights
args.use_context_mod = True # use CM
args.mixed_dataset = True # use mixed dataset training
## training parameters for Hypernet
args.n_steps = n_steps
args.n_batch_list = n_batch_list # will be multiplied by num of datasets
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
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, x_out_test] = hknet_pipeline.NNTest_alldatasets(SoW_test_range, sys_model, test_input_list, test_target_list, path_results,test_init_list)



## Close wandb run
if args.wandb_switch: 
   wandb.finish() 

####################
### Plot results ###
####################
i = 0 # dataset index
PlotfolderName = "Figures/Linear_CA/"
PlotfileName0 = "TrainPVA_position.png"
PlotfileName1 = "TrainPVA_velocity.png"
PlotfileName2 = "TrainPVA_acceleration.png"

Plot = Plot(PlotfolderName, PlotfileName0)
print(f"Plot dataset {i}")
Net_out = x_out_test[args.N_T*i:args.N_T*(i+1),:, :]
Plot.plotTraj_CA( test_target_list[i][0], KF_out_list[i], Net_out, dim=0, file_name=PlotfolderName+PlotfileName0, model_name="KalmanNet")#Position
Plot.plotTraj_CA( test_target_list[i][0], KF_out_list[i], Net_out, dim=1, file_name=PlotfolderName+PlotfileName1, model_name="KalmanNet")#Velocity
Plot.plotTraj_CA( test_target_list[i][0], KF_out_list[i], Net_out, dim=2, file_name=PlotfolderName+PlotfileName2, model_name="KalmanNet")#Acceleration