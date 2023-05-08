"""
This file contains the parameters for the simulations with linear canonical model
* Linear State Space Models with Full Information
    # v = 0, -10, -20 dB
    # scaling model dim to 5x5, 10x10, 20x20, etc
    # scalable trajectory length T
    # random initial state
* Linear SS Models with Partial Information
    # observation model mismatch
    # evolution model mismatch
"""

import torch

m = 2 # state dimension = 2, 5, 10, etc.
n = 2 # observation dimension = 2, 5, 10, etc.

##################################
### Initial state and variance ###
##################################
m1_0 = torch.zeros(m, 1) # initial state mean

#########################################################
### state evolution matrix F and observation matrix H ###
#########################################################
# F in canonical form
F = torch.eye(m)
F[0] = torch.ones(1,m) 

if m == 2:
    # H = I
    H = torch.eye(2)
else:
    # H in reverse canonical form
    H = torch.zeros(n,n)
    H[0] = torch.ones(1,n)
    for i in range(n):
        H[i,n-1-i] = 1

#######################
### Rotated F and H ###
#######################
F_rotated = torch.zeros_like(F)
H_rotated = torch.zeros_like(H)
if(m==2):
    alpha_degree = 10 # rotation angle in degree
    rotate_alpha = torch.tensor([alpha_degree/180*torch.pi])
    cos_alpha = torch.cos(rotate_alpha)
    sin_alpha = torch.sin(rotate_alpha)
    rotate_matrix = torch.tensor([[cos_alpha, -sin_alpha],
                                [sin_alpha, cos_alpha]])

    F_rotated = torch.mm(F,rotate_matrix) 
    H_rotated = torch.mm(H,rotate_matrix) 

###############################################
### process noise Q and observation noise R ###
###############################################
# Noise variance takes the form of a diagonal matrix
Q_structure = torch.eye(m)
R_structure = torch.eye(n)
# Non-diagonal noise
def normalize(matrix):
    frobenius_norm = torch.sqrt(torch.sum(matrix**2))
    return matrix / frobenius_norm

if m==2 and n==2:
  Q_structure_nonid = torch.tensor([[5, 2], [2, 5]])/torch.sqrt(torch.tensor(29))
  R_structure_nonid = torch.tensor([[3, 2], [2, 3]])/torch.sqrt(torch.tensor(13))
elif m==5 and n==5:
  A = torch.tensor([
    [8.0, 2.0, 3.0, 1.0, 1.0],
    [2.0, 6.0, 1.0, 1.0, 0.0],
    [3.0, 1.0, 7.0, 2.0, 1.0],
    [1.0, 1.0, 2.0, 5.0, 1.0],
    [1.0, 0.0, 1.0, 1.0, 4.0]
  ])
  A = A.matmul(A.t())  # Make it symmetric positive definite
  Q_structure_nonid = normalize(A)

  B = torch.tensor([
      [10.0, 3.0, 2.0, 1.0, 2.0],
      [3.0, 7.0, 1.0, 0.0, 1.0],
      [2.0, 1.0, 8.0, 2.0, 1.0],
      [1.0, 0.0, 2.0, 6.0, 2.0],
      [2.0, 1.0, 1.0, 2.0, 5.0]
  ])
  B = B.matmul(B.t())  # Make it symmetric positive definite
  R_structure_nonid = normalize(B)