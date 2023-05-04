import torch

def unsupervised_loss(SysModel,pre_trained_model, y_true):
    y_hat = torch.zeros([, sysmdl_n, sysmdl_T])
    for t in range(sysmdl_T):
        y_hat[:,:,t] = torch.squeeze(SysModel[i].h(torch.unsqueeze(x_out_training_batch[:,:,t],2)))
    return 