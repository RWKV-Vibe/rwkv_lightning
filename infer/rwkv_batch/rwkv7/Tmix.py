import torch
from torch.nn import functional as F
MyStatic = torch.jit.script 
from .ops.ops_loader import RWKV7_ONE_OP, RWKV7_SEQ_OP, RWKV7_BATCH_OP

############################################### FP16 ###########################################
@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    xx = x_prev[0] - x
    x_prev[0] = x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))

    xx = RWKV7_ONE_OP(state, r, w, k, v, -kk, kka, elapsed_t) # !!! using CUDA to modify state in-place !!! (faster too)

    xx = F.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return F.linear((xx * g), O_), v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    T = x.shape[0]
    xx = torch.cat((x_prev[0].unsqueeze(0), x[:-1,:])) - x
    x_prev[0] = x[-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))

    xx = RWKV7_SEQ_OP(state, r, w, k, v, -kk, kka, elapsed_t)

    xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return F.linear((xx * g), O_), v_first

@MyStatic
def RWKV_x070_TMix_seq_batch(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b, elapsed_t):
    B,T,C = x.shape
    xx = torch.cat((x_prev[0].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[0] = x[:,-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = F.linear(xr, R_)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    k = F.linear(xk, K_)
    v = F.linear(xv, V_)
    a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)

    kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    k = k * (1 + (a-1) * k_a)
    kka = kk * a

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))

    # if T == 1:
    #     vk = v.view(B,H,N,1) @ k.view(B,H,1,N)
    #     ab = (-kk).view(B,H,N,1) @ (kk*a).view(B,H,1,N)
    #     state = state * w.view(B,H,1,N) + state @ ab + vk
    #     xx = (state.to(dtype=x.dtype) @ r.view(B,H,N,1)).view(B*T,H*N)
    # else:
    xx = RWKV7_BATCH_OP(state, r, w, k, v, -kk, kka, elapsed_t).view(B*T,H*N)

    xx = F.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,T,H*N)
    xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
    return F.linear((xx * g), O_), v_first

############################################### W8A16 ###########################################