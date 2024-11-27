# -*- coding: utf-8 -*-
from efficient_kan import KAN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import numpy as np
from jax import random, jit
import jax.numpy as jnp
import sklearn
from sklearn.preprocessing import MinMaxScaler
import itertools
from functools import partial
from tqdm import trange, tqdm
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time
from torch.func import jacrev, hessian
start_time=time.time()
import math
from torch import vmap



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(0)
device = torch.device("cuda")
# def calculate_V(T,r,v,M,K):
#     delta_T=T/M
#     S_max=3*K
#     delta_S= S_max/M

#     def get_call_matrix(M):
#         f_matrx = np.matrix(np.array([0.0] * (M + 1) * (M + 1)).reshape((M + 1, M + 1)))
#         f_matrx[:, 0] = 0.0
#         for i in range(M + 1):
#             f_matrx[M, i] = float(max(delta_S * i - K, 0))
#         f_matrx[:, M] = float(S_max - K)
#         print("f_matrix shape : ", f_matrx.shape)
#         return f_matrx
#     def calculate_coeff(j):
#         vj2 = (v * j)**2
#         aj = 0.5 * delta_T * (r * j - vj2)
#         bj = 1 + delta_T * (vj2 + r)
#         cj = -0.5 * delta_T * (r * j + vj2)
#         return aj, bj, cj

#     def get_coeff_matrix(M):
#         matrx = np.matrix(np.array([0.0]*(M-1)*(M-1)).reshape((M-1, M-1)))
#         a1, b1, c1 = calculate_coeff(1)
#         am_1, bm_1, cm_1 = calculate_coeff(M - 1)
#         matrx[0,0] = b1
#         matrx[0,1] = c1
#         matrx[M-2, M-3] = am_1
#         matrx[M-2, M-2] = bm_1
#         for i in range(2, M-1):
#             a, b, c = calculate_coeff(i)
#             matrx[i-1, i-2] = a
#             matrx[i-1, i-1] = b
#             matrx[i-1, i] = c
#         print("coeff matrix shape : ",  matrx.shape)
#         return matrx


#     f_matrx = get_call_matrix(M)
#     matrx = get_coeff_matrix(M)
#     inverse_m = matrx.I
#     for i in range(M, 0, -1):
#         Fi = f_matrx[i, 1:M]
#         Fi_1 = inverse_m * Fi.reshape((M-1, 1))
#         Fi_1 = list(np.array(Fi_1.reshape(1, M-1))[0])
#         f_matrx[i-1, 1:M]=Fi_1
#     return f_matrx, delta_T,delta_S



class PI_DeepONet(nn.Module):
    def __init__(self,model1,model2,model4,model5):
        super(PI_DeepONet, self).__init__()
        # Network initialization and evaluation functions
        self.model1 = model1
        self.model2 = model2
        # self.model3 = model3
        self.model4 = model4
        self.model5 = model5


        self. bc_losses = []
        self. pde_losses = []
    def reshape(self,X):
        reshaped_X = X.reshape(-1,)
        return reshaped_X
    
    def brunk_net(self,u1,u2,u_s1,u_s2):
        BC1=self.model1(u1)
        BC2=self.model2(u2)
        # BC3 = self.model3(u3)
        B=BC1*BC2
        # loss=torch.mean((BC1.flatten() -u_s1) ** 2+(BC2.flatten() -u_s2) ** 2+(BC3.flatten() -u_s3) ** 2)
        return B


    def helper(self,X, Y):
        reshaped_X=self.reshape(X)
        reshaped_Y=self.reshape(Y)
        stacked_tensor = torch.stack([reshaped_X, reshaped_Y])
        permuted_tensor = stacked_tensor.permute(1, 0)
        return permuted_tensor

    # Define DeepONet architecture
    def operator_net(self,u1,u2,u_s1,u_s2,x,t):
        n=len(x)
        B1=self.brunk_net(u1,u2,u_s1,u_s2)
        B1= (B1.view(1, -1)).repeat(n, 1).float()
        B = self.model4(B1)
        y = self.helper(x, t)
        T = self.model5(y)
        outputs =torch.sum(B * T, dim=1)
        return outputs

    # Define PDE residual
    def residual_net(self,u1,u2,u_s1,u_s2,x,t):
        s=self.operator_net(u1,u2,u_s1,u_s2,x,t)
        s_x =jacrev(self.operator_net,argnums=4)(u1,u2,u_s1,u_s2,x,t).sum(dim=0).to(device)
        s_xx =(hessian(self.operator_net,argnums=4)(u1,u2,u_s1,u_s2,x,t).sum(dim=0)).sum(dim=0).to(device)
        s_t =jacrev(self.operator_net,argnums=5)(u1,u2,u_s1,u_s2,x,t).sum(dim=0).to(device)
        member1 = torch.tensor(0.5, device='cuda')
        member2 = torch.tensor(0.01, device='cuda')
        member3 = torch.tensor(0.3, device='cuda')
        res =s_t-(member1)*(member2**2)*(x**2)*s_xx-member3*x*s_x+member3*s
        return res
        # r =0.025610
        # v=0.165856529

    # Define boundary loss
    def loss_bcs(self,u1,u2,u_s1,u_s2,x,t, output):
        # Compute forward pass
        s_pred= self.operator_net(u1,u2,u_s1,u_s2,x,t)
       
        
        # Compute loss
        loss = torch.mean((output.flatten() - s_pred)**2)
        return loss


    # Define residual loss
    def loss_res(self,u1,u2,u_s1,u_s2,x,t,output):
        # Compute forward pass
        pred = self.residual_net(u1,u2,u_s1,u_s2,x,t)
        loss = torch.mean((output.flatten() - pred)**2)
        return loss



    def train(self,u1,u2,u_s1, u_s2,dataloader1,dataloader2):
        params1 = tuple(model1.parameters())
        params2 = tuple(model2.parameters())
        # params3 = tuple(model3.parameters())
        params4 = tuple(model4.parameters())
        params5 = tuple(model5.parameters())
        params = params1 + params2+ params4+ params5
        # params = (model1.parameters(), model2.parameters())
        # Initialize optimizer

        self.optimizer = torch.optim.LBFGS(params, lr=0.0001,history_size=10, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-64, tolerance_change=1e-64)
    
        pbar = tqdm(range(60), desc='description')
    
       
        for _ in pbar:
           
            
            for (x_i, t_i,outputs_i),(x_b, t_b, outputs_b) in zip(dataloader1, dataloader2):
                def closure():
                    global pde_loss, bc_loss
                    self.optimizer.zero_grad()
                    bc_loss= self.loss_bcs(u1,u2,u_s1,u_s2,x_i, t_i,outputs_i)
                    pde_loss=self.loss_res(u1,u2,u_s1,u_s2,x_b,t_b,outputs_b)
                    # _,brunk_net_loss= model.brunk_net(u1, u2,u_s1, u_s2)
                    loss =pde_loss+2*bc_loss
                    loss.backward()
                    return loss

            # # if _ % 5 == 0 and _ < 50:
            #     model1.update_grid_from_samples(u1)
            #     model2.update_grid_from_samples(u2)
                # model4.update_grid_from_samples(u_b1)
                # model5.update_grid_from_samples(u_b2)
                self.optimizer.step(closure)
       

            if _ % 1 == 0:
                pbar.set_description("pde loss: %.2e | bc loss: %.2e" % (
                pde_loss.detach().cpu().numpy(), bc_loss.detach().cpu().numpy()))

            # self.pde_losses.append(pde_loss.detach().cpu().numpy())
            # self.bc_losses.append(bc_loss.detach().cpu().numpy())








# Deinfe initial and boundary conditions for advection equation
def f1(x,t,k):
  return np.where(t==0,np.maximum(x-2.411,0),0)
def f2(x,k):
  return np.where(x ==4.822, x-2.411, 10)
def f3(x):
  return np.where(x==0,0,0)




def min_max_normalize(x, min_val, max_val):
    normalized_x = (x - min_val) / (max_val - min_val)
    return normalized_x
# Geneate training data corresponding to one input sample
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)
def generate_one_training_data(key,P,Q,K,M,r,v,T):
    subkeys = random.split(key, 10)
    subkeys1 = random.split(key, 2)
    # idx = random.randint(subkeys[8], (100, 2), 0, max(M, M))
    # call,delta_T,delta_S= calculate_V(T, r, v, M, K)
    # call = np.asarray(call)
    # s_bcs4 = call[idx[:, 1], idx[:, 0]]
    # s_bc4 = s_bcs4.reshape(-1, 1)
    # x_bc4 = (idx[:, 0] * delta_S).reshape(-1, 1)
    # t_bc4 = (T - idx[:, 1] * delta_T).reshape(-1, 1)

    
    gp_params = (1.0,0.2)
    jitter = 1e-10
    X = np.linspace(0,7.233,P//3)[:,None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter*np.eye(P//3))
    gp_sample = np.dot(L, random.normal(subkeys1[0], (P//3,)))
    f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Create grid
    x= np.linspace(0,7.233,P//3)
    t = np.linspace(0,365,P//3)
    x_bc4= f_fn(x)
    x_bc4 = x.reshape(-1, 1)
    t_bc4= f_fn(t)
    t_bc4=t_bc4.reshape(-1, 1)
    x_bc4 = jnp.array(x_bc4)
    t_bc4=jnp.array(t_bc4)
    # print(type(x_bc4))
    # print(t_bc4.shape)
    np_K=K*(np.ones((P // 3, 1)))

    x_bc1 = random.uniform(subkeys[2], shape=(P // 3, 1), minval=0, maxval=4.822)
    x_bc2 = 4.822* (np.ones((P // 3, 1)))
    x_bc3 = np.zeros((P // 3, 1))
    # x_bc4= random.uniform(subkeys[7], shape=(P // 3, 1), minval=0, maxval=3* K)
    x_bcs = np.vstack([x_bc1, x_bc2,x_bc3])
    x_bcs_min_value = np.min(x_bcs)
    x_bcs_max_value = np.max(x_bcs)
    x_bcs=min_max_normalize(x_bcs, x_bcs_min_value, x_bcs_max_value)
    x_bcs= x_bcs.__array__()
    x_i = torch.tensor(x_bcs)

    t_bc1 = np.zeros((P // 3, 1))
    t_bc2 = random.uniform(subkeys[3], shape=(P // 3, 1), minval=0, maxval=1)
    t_bc3 = random.uniform(subkeys[4], shape=(P // 3, 1), minval=0, maxval=1)
    # t_bc4 = random.uniform(subkeys[8], shape=(P // 3, 1), minval=0, maxval=365)
    t_bcs = np.vstack([t_bc1, t_bc2,t_bc3])
    t_bcs_min_value = np.min(t_bcs)
    t_bcs_max_value = np.max(t_bcs)
    t_bcs= min_max_normalize(t_bcs, t_bcs_min_value,t_bcs_max_value)
    # t_bcs = np.vstack([t_bcs,t_bc4])
    t_bcs = t_bcs.__array__()
    t_i = torch.tensor(t_bcs)


    s_bc1=f1(x_bc1,t_bc1,np_K)
    s_bc1 =np.array(list(s_bc1))
    s_bc1 =s_bc1.reshape(-1,1)
    s_bc2 = f2(x_bc2,np_K)
    s_bc2 = np.array(list(s_bc2))
    s_bc2 = s_bc2.reshape(-1, 1)
    s_bc3 = f3(x_bc3)
    s_bc3 = np.array(list(s_bc3))
    s_bc3 = s_bc3.reshape(-1, 1)
    # s_train= np.vstack([s_bc1, s_bc2,s_bc3,s_bc4])
    s_train= np.vstack([s_bc1, s_bc2,s_bc3])
    s_bcs_min_value = np.min(s_train)
    s_bcs_max_value = np.max(s_train)
    s_train= min_max_normalize(s_train,s_bcs_min_value, s_bcs_max_value)
    s_train= s_train.__array__()
    print(s_bc1.shape)
    # print(s_bc2.shape)
    # print(s_bc3.shape)
    
    # print(s_train.shape)
    outputs_i= torch.tensor(s_train)

    x_bb = random.uniform(subkeys[5], shape=(Q, 1), minval=0, maxval=4.822)
    t_bb = random.uniform(subkeys[6], shape=(Q, 1), minval=0, maxval=1)
    x_b = min_max_normalize(x_bb,x_bcs_min_value, x_bcs_max_value)
    t_b= min_max_normalize(t_bb,t_bcs_min_value,t_bcs_max_value)
    x_b = x_b.__array__()
    x_b= torch.tensor(x_b)
    t_b = t_b.__array__()
    t_b = torch.tensor(t_b)
    outputs_b = torch.zeros((Q, 1))


    s_bc4=f1(x_bc4,t_bc1,np_K)
    s_bc4 =np.array(list(s_bc4))
    s_bc4 =s_bc4.reshape(-1,1)
    
    x_bc11=min_max_normalize(x_bc4, x_bcs_min_value, x_bcs_max_value)
    x_bc11=x_bc11.__array__()
    x_bc11= torch.tensor(x_bc11)
    t_bc11 = min_max_normalize(t_bc1, t_bcs_min_value, t_bcs_max_value)
    t_bc11 = t_bc11.__array__()
    t_bc11 = torch.tensor(t_bc11)
    s_bc11 = min_max_normalize(s_bc4, s_bcs_min_value, s_bcs_max_value)
    s_bc11= s_bc11.__array__()
    s_bc11 = torch.tensor(s_bc11)
    u_1= torch.cat((x_bc11,t_bc11,s_bc11), dim=1)  # shape: (4, 2)
    u_s1=s_bc11

    s_bc5 = f2(x_bc2,np_K)
    s_bc5 = np.array(list(s_bc5))
    s_bc5 = s_bc5.reshape(-1, 1)

    x_bc22 = min_max_normalize(x_bc2, x_bcs_min_value, x_bcs_max_value)
    x_bc22 = x_bc22.__array__()
    x_bc22 = torch.tensor(x_bc22)
    t_bc22 = min_max_normalize(t_bc4, t_bcs_min_value, t_bcs_max_value)
    t_bc22 = t_bc22.__array__()
    t_bc22 = torch.tensor(t_bc22)
    s_bc22 = min_max_normalize(s_bc5, s_bcs_min_value, s_bcs_max_value)
    s_bc22 = s_bc22.__array__()
    s_bc22 = torch.tensor(s_bc22)
    u_2 = torch.cat((x_bc22, t_bc22,s_bc22), dim=1)
    u_s2=s_bc22


    # x_bc33 = min_max_normalize(x_bc3, x_bcs_min_value, x_bcs_max_value)
    # x_bc33 = x_bc33.__array__()
    # x_bc33 = torch.tensor(x_bc33)
    # t_bc33= min_max_normalize(t_bc3, t_bcs_min_value, t_bcs_max_value)
    # t_bc33 = t_bc33.__array__()
    # t_bc33= torch.tensor(t_bc33)
    # s_bc33 = min_max_normalize(s_bc3, s_bcs_min_value, s_bcs_max_value)
    # s_bc33 = s_bc33.__array__()
    # s_bc33 = torch.tensor(s_bc33)
    # u_3 = torch.cat((x_bc33, t_bc33), dim=1)
    # u_s3=s_bc33

    # x_bc4= min_max_normalize(x_bc4,x_bcs_min_value, x_bcs_max_value)
    # x_bc4 = x_bc4.__array__()
    # x_bc4= torch.tensor(x_bc4)
    # t_bc4 = t_bc4.__array__()
    # t_bc4= torch.tensor(t_bc4)
    # s_bc4= min_max_normalize(s_bc4,s_bcs_min_value, s_bcs_max_value)
    # s_bc4= s_bc4.__array__()
    # s_bc4 = torch.tensor(s_bc4)

    x_l=[2.426, 2.403, 2.407, 2.393, 2.4, 2.388, 2.362, 2.315, 2.313, 2.308, 2.3,
          2.318, 2.33, 2.304, 2.291, 2.27, 2.28, 2.291, 2.263, 2.297, 2.306, 2.308,
          2.299, 2.3, 2.35, 2.356, 2.324, 2.331, 2.308, 2.3, 2.281, 2.275, 2.269, 
          2.268, 2.263, 2.268, 2.275, 2.225, 2.259, 2.274, 2.255, 2.269, 2.296,
          2.34, 2.35, 2.33, 2.298, 2.281, 2.284, 2.253, 2.281, 2.355, 2.389, 2.398,
          2.399, 2.395, 2.44, 2.466, 2.464, 2.418, 2.438, 2.417, 2.44, 2.444, 2.442, 
          2.478, 2.466, 2.46, 2.46, 2.479, 2.481, 2.464]
    t_l=[121, 120, 119, 118, 117, 114, 113, 112, 111, 110, 107, 
          106, 105, 104, 103, 100, 99, 98, 97, 96, 93, 92, 91, 90,
          89, 85, 84, 83, 82, 79, 78, 77, 76, 75, 72, 71, 70, 69, 68,
          65, 64, 63, 62, 61, 58, 57, 56, 55, 54, 51, 50, 49, 48, 37,
          36, 35, 34, 33, 30, 29, 28, 27, 26, 23, 22, 21, 20, 19, 16, 15, 14, 13]
    t_l=[x / 365 for x in t_l]
    outputs_bt=[0.2097, 0.1910, 0.1925, 0.1810, 0.1846, 0.1720, 0.1531, 0.1231, 0.1208,
        0.1168, 0.1090, 0.1179, 0.1236, 0.1079, 0.0999, 0.0863, 0.0902, 0.0946,
        0.0800, 0.0955, 0.0969, 0.0969, 0.0913, 0.0907, 0.1162, 0.1149, 0.0964,
        0.0990, 0.0863, 0.0793, 0.0697, 0.0661, 0.0626, 0.0613, 0.0565, 0.0576,
        0.0596, 0.0397, 0.0514, 0.0546, 0.0464, 0.0508, 0.0609, 0.0797, 0.0813,
        0.0708, 0.0560, 0.0484, 0.0486, 0.0348, 0.0439, 0.0741, 0.0897, 0.0817,
        0.0811, 0.0780, 0.1001, 0.1135, 0.1085, 0.0825, 0.0917, 0.0797, 0.0903,
        0.0888, 0.0865, 0.1048, 0.0969, 0.0924, 0.0887, 0.0976, 0.0975, 0.0871]
    xl=np.array(x_l).reshape(72,1)
    tl=np.array(t_l).reshape(72,1)
    outputs_bt=np.array(outputs_bt).reshape(72,1)
    
    x_lc=np.vstack([x_bc1, x_bc2,x_bc3,xl])
    t_lc=np.vstack([t_bc1, t_bc2,t_bc3,tl])
    x_lb=np.vstack([x_bb,xl])
    t_lb=np.vstack([t_bb,tl])
    outputs_tc=np.vstack([s_bc1, s_bc2,s_bc3,outputs_bt])
    outputs_tb=np.vstack([outputs_b,outputs_bt])




    x_lc= min_max_normalize(x_lc, x_bcs_min_value, x_bcs_max_value)
    t_lc= min_max_normalize(t_lc, t_bcs_min_value, t_bcs_max_value)
    x_lb= min_max_normalize(x_lb, x_bcs_min_value, x_bcs_max_value)
    t_lb= min_max_normalize(t_lb, t_bcs_min_value, t_bcs_max_value)
    outputs_tc= min_max_normalize(outputs_tc, s_bcs_min_value, s_bcs_max_value)
    outputs_tb= min_max_normalize(outputs_tb, s_bcs_min_value, s_bcs_max_value)
    x_lc=torch.tensor(x_lc)
    t_lc=torch.tensor(t_lc)
    x_lb=torch.tensor(x_lb)
    t_lb=torch.tensor(t_lb)
    outputs_tc=torch.tensor(outputs_tc)
    outputs_tb=torch.tensor(outputs_tb)




    





    return u_1,u_2,u_s1,u_s2,x_i,t_i,outputs_i,x_b,t_b,outputs_b,x_lc,t_lc,outputs_tc,x_lb,t_lb,outputs_tb,\
           s_bcs_min_value, s_bcs_max_value,x_bcs_min_value, x_bcs_max_value,t_bcs_min_value, t_bcs_max_value



key = random.PRNGKey(0)

K=2.411
P =120 # number of output sensors, 100 for each side
Q =120  # number of collocation points for each input sample
M = 5000
r =0.025610
v=0.165856529
T=1
u_1,u_2,u_s1,u_s2,x_i, t_i,outputs_i, x_b, t_b, outputs_b,x_lc,t_lc,outputs_tc,x_lb,t_lb,outputs_tb,\
         s_bcs_min_value, s_bcs_max_value,x_bcs_min_value, x_bcs_max_value,t_bcs_min_value, t_bcs_max_value\
            =generate_one_training_data(key,P,Q,K,M,r,v,T)
u_1=u_1.float().to(device)
u_s1=u_s1.float().to(device).reshape(-1,)
# print(u_1)
u_2=u_2.float().to(device)
u_s2=u_s2.float().to(device).reshape(-1,)
# u_3=u_3.float().to(device)
# u_s3=u_s3.float().to(device).reshape(-1,)
x_i=x_i.float()
t_i=t_i.float()
outputs_i=outputs_i.float()
x_b=x_b.float()
t_b=t_b.float()
outputs_b=outputs_b.float()
# x_bc4=x_bc4.float()
# t_bc4=t_bc4.float()
# s_bc4=s_bc4.float()
x_i = x_i.reshape(-1,).to(device)
t_i = t_i.reshape(-1,).to(device)
outputs_i = outputs_i.reshape(-1,).to(device)
x_b = x_b.reshape(-1,).to(device)
t_b = t_b.reshape(-1,).to(device)
outputs_b = outputs_b.reshape(-1,).to(device)
# x_bc4 = x_bc4.reshape(-1,).to(device)
# t_bc4 = t_bc4.reshape(-1,).to(device)
# s_bc4 = s_bc4.reshape(-1,).to(device)
print("x_i shape:", x_i.shape)
print("t_i shape:", t_i.shape)
print("outputs_i shape:", outputs_i.shape)
dataset1 = TensorDataset(x_i,t_i,outputs_i)
dataset2 = TensorDataset(x_b,t_b,outputs_b)
# dataset3 = TensorDataset(x_bc4,t_bc4,s_bc4)
batch_size1= 100
batch_size2= 100
dataloader1 = DataLoader(dataset1, batch_size=batch_size1, shuffle=True)
dataloader2 = DataLoader(dataset2, batch_size=batch_size2, shuffle=True)
# dataloader3 = DataLoader(dataset3, batch_size=batch_size2, shuffle=True)
x_lc= x_lc.reshape(-1,)
t_lc= t_lc.reshape(-1,)
outputs_tc= outputs_tc.reshape(-1,)
x_lc=x_lc.float().to(device)
t_lc=t_lc.float().to(device)
outputs_tc=outputs_tc.float().to(device)
batch_size3= 32
dataset3 = TensorDataset(x_lc,t_lc,outputs_tc)
dataloader3 = DataLoader(dataset3, batch_size=batch_size3, shuffle=True)
x_lb= x_lb.reshape(-1,)
t_lb= t_lb.reshape(-1,)
outputs_tb= outputs_tb.reshape(-1,)
x_lb=x_lb.float().to(device)
t_lb=t_lb.float().to(device)
outputs_tb=outputs_tb.float().to(device)
batch_size3= 32
dataset4= TensorDataset(x_lb,t_lb,outputs_tb)
dataloader4 = DataLoader(dataset4, batch_size=batch_size3, shuffle=True)
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(0)


model1 =KAN([3,2,1], base_activation=nn.Identity)
model2 = KAN([3,2,1], base_activation=nn.Identity)
# model3 = KAN([2,1], base_activation=nn.Identity)
model4 = KAN([40,2,1], base_activation=nn.Identity)
model5 = KAN([2,10,1], base_activation=nn.Identity)

# model1 =BayesianNetwork()
# model2 =BayesianNetwork()
# model4 =BayesianNetwork1()
# model5 =BayesianNetwork()

model= PI_DeepONet(model1,model2,model4,model5)
model.to(device)
model.train(u_1,u_2,u_s1,u_s2,dataloader3,dataloader4)
data=pd.read_csv('data.csv')
x_test=data.iloc[:,1]
t_test=data.iloc[:,2]/365
x_test=torch.tensor(x_test).float()
x_test=x_test.unsqueeze(1).to(device)
t_test=torch.tensor(t_test).float()
t_test=t_test.unsqueeze(1).to(device)

x_test=min_max_normalize(x_test,x_bcs_min_value, x_bcs_max_value)
t_test=min_max_normalize(t_test,t_bcs_min_value,t_bcs_max_value)

s_pred = model.operator_net(u_1,u_2,u_s1, u_s2, x_test,t_test)
s_pred=s_pred*s_bcs_max_value
s_true=data.iloc[:,3]
s_true=torch.tensor(s_true).to(device)
error_s =(s_pred- s_true)/s_true
print('s_pred:\n',s_pred)
print('s_true:\n',s_true)
print('error_s:\n',error_s)
def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)
def rmse(y_pred, y_true):
    return torch.sqrt(mse(y_pred, y_true))
def mape(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
mse_val = mse(s_pred, s_true)
rmse_val = rmse(s_pred, s_true)
mape_val = mape(s_pred, s_true)

print(f"MSE: {mse_val.item()}")
print(f"RMSE: {rmse_val.item()}")
print(f"MAPE: {mape_val.item()}%")
end_time=time.time()
rap_time=end_time-start_time
print('run-time:{}'.format(rap_time))
