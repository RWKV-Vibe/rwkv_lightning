import torch
from torch.nn import functional as F
MyStatic = torch.jit.script 
from .ops.ops_loader import rwkv_mm_sparsity
from .ops.mm_int8_kernel import linear_w8a8

############################################### FP16 ###########################################
@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev[1] - x
    x_prev[1] = x
    k = x + xx * x_k
    k = torch.relu(F.linear(k, K_)) ** 2
    kv = rwkv_mm_sparsity(k, V_)
    # kv = k @ V_
    # kv = SPMV_OP(k, V_)
    return kv

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(0), x[:-1,:])) - x
    x_prev[1] = x[-1,:]
    k = x + xx * x_k
    k = torch.relu(F.linear(k, K_)) ** 2
    # print("Sparsity:", (k == 0).float().mean().item())
    return k @ V_ # F.linear(k, V_)

@MyStatic
def RWKV_x070_CMix_seq_batch(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[1] = x[:,-1,:]
    k = x + xx * x_k
    k = torch.relu(F.linear(k, K_)) ** 2
    return k @ V_ # F.linear(k, V_)

############################################### W8A16 ###########################################

############################################### W8A8 ###########################################
@MyStatic
def RWKV_x070_CMix_one_i8(x, x_prev, x_k, K_w, K_scale, V_w, V_scale):
    xx = x_prev[1] - x
    x_prev[1] = x
    k = x + xx * x_k
    k = torch.relu(linear_w8a8(k, K_w, K_scale)) ** 2
    return linear_w8a8(k, V_w, V_scale)

@MyStatic
def RWKV_x070_CMix_seq_i8(x, x_prev, x_k, K_w, K_scale, V_w, V_scale):
    xx = torch.cat((x_prev[1].unsqueeze(0), x[:-1,:])) - x
    x_prev[1] = x[-1,:]
    k = x + xx * x_k
    k = torch.relu(linear_w8a8(k, K_w, K_scale)) ** 2
    return linear_w8a8(k, V_w, V_scale)

@MyStatic
def RWKV_x070_CMix_seq_batch_i8(x, x_prev, x_k, K_w, K_scale, V_w, V_scale):
    xx = torch.cat((x_prev[1].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[1] = x[:,-1,:]
    k = x + xx * x_k
    k = torch.relu(linear_w8a8(k, K_w, K_scale)) ** 2
    return linear_w8a8(k, V_w, V_scale)
