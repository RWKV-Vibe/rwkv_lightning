########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

from typing import List
import torch
from torch.nn import functional as F


torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
# MyModule = nn.Module
# MyFunction = torch.compile()
# MyStatic = torch.compile()
MyDisable = torch.compiler.disable
# def __nop(ob): return ob
# MyFunction = __nop
# MyStatic = __nop
# MyDisable = __nop

HEAD_SIZE = 64
DTYPE = torch.half

from .Tmix import RWKV_x070_TMix_one, RWKV_x070_TMix_seq, RWKV_x070_TMix_seq_batch
from .Cmix import RWKV_x070_CMix_one, RWKV_x070_CMix_seq, RWKV_x070_CMix_seq_batch

class RWKV_x070(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.head_size = 64
        self.eval()
        
        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        args.n_embd = self.n_head * self.head_size

        assert HEAD_SIZE == self.head_size
        assert self.head_size == args.head_size

        keys = list(z.keys())
        max_layer = -1
        for k in keys:
            kk = k.split('.')
            # if kk[0] == 'blocks' and int(kk[1]) >= 10:
            #     continue
            if 'att.g1' in k or 'att.g2' in k or 'att.a1' in k or 'att.a2' in k or 'att.w1' in k or 'att.w2' in k or 'att.v1' in k or 'att.v2' in k or 'ffn.value.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE, device="cuda")
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
            z[k] = z[k].contiguous()
            if kk[0] == 'blocks':
                max_layer = max(max_layer, int(kk[1]))
        args.n_layer = max_layer + 1
        print(args)
        self.n_layer, self.n_embd = args.n_layer, args.n_embd

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

    def generate_zero_state(self, bsz):
        args = self.args
        state = [None, None, None]
        if bsz >= 1:
            state[0] = torch.zeros((args.n_layer, 2, bsz, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, bsz, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=DTYPE, requires_grad=False, device="cuda")
            state[2] = torch.zeros((bsz,), dtype=torch.int32, requires_grad=False, device="cuda")
        else:
            state[0] = torch.zeros((args.n_layer, 2, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=DTYPE, requires_grad=False, device="cuda")
            state[2] = torch.zeros((), dtype=torch.int32, requires_grad=False, device="cuda")
        return state

    def forward(self, idx, state, full_output=False): # will modify state in-place
        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                x = self.z['emb.weight'][idx[0]]
                return self.forward_one(x, state)
        elif type(idx) is torch.Tensor:
            return self.forward_one(idx, state)
        else:
            x = self.z['emb.weight'][idx]
            return self.forward_one(x, state)
        
    def forward_batch(self, tokens, state, full_output=False): # will modify state in-place
        assert type(tokens) is list
        lengths = [len(x) for x in tokens]
        if len(set(lengths)) == 1 and full_output == False:
            return self.forward_batch_same_length(tokens, state, full_output)

        bsz = len(tokens)
        pos = [0] * bsz

        if full_output == False:
            out = torch.empty((bsz, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda")
        else:
            out = [torch.empty((0, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda") for _ in range(bsz)]
        while True:
            active = [i for i in range(bsz) if pos[i] < lengths[i]]
            if not active:
                break
            step = min(lengths[i] - pos[i] for i in active)
            batch_tokens = [tokens[i][pos[i]:pos[i]+step] for i in active]
            batch_state = [state[0][:,:,active],state[1][:,active], state[2][active]] # state[0]=[Layer][2][Bsz][C]    state[1]=[Layer][Bsz][H][N][N]
            new_out = self.forward_batch_same_length(batch_tokens, batch_state, full_output)
            for k, i in enumerate(active):
                if full_output == False:
                    out[i] = new_out[k]
                else:
                    out[i] = torch.cat([out[i], new_out[k]], dim=0)
                state[0][:,:,i] = batch_state[0][:,:,k]
                state[1][:,i] = batch_state[1][:,k]
                state[2][i] = batch_state[2][k]
                pos[i] += step
        return out

    def forward_batch_same_length(self, tokens, state, full_output=False):
        assert type(tokens) is list
        assert len(set([len(x) for x in tokens])) == 1, 'here all sequences must have the same length'
        return self.forward_seq_batch(tokens, state, full_output)

    @MyFunction
    def forward_one(self, x:torch.Tensor, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_one(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            state[2] += 1
            return x
        
    @MyFunction
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            state[2] += len(idx)
            return x
        
    @MyFunction
    def forward_seq_batch(self, idxs:List[List[int]], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][torch.tensor(idxs, device=z['emb.weight'].device)]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_seq_batch(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[:,-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            state[2] += len(idxs[0])
            return x
    @MyFunction
    def forward_seq_batch_chunk(self, idxs: List[List[int]], state: List[torch.Tensor], chunk_len: int = 64, full_output: bool = False):
        with torch.no_grad():
            z = self.z
            device = z['emb.weight'].device
            
            # 转换为 Tensor，形状为 [Batch, Total_Seq_Len]
            full_idxs = torch.tensor(idxs, device=device)
            batch_size, total_len = full_idxs.size()
            
            all_outputs = []

            # 按 chunk_len 进行循环处理
            for start in range(0, total_len, chunk_len):
                end = min(start + chunk_len, total_len)
                # 截取当前的 chunk
                chunk_idxs = full_idxs[:, start:end]
                
                x = z['emb.weight'][chunk_idxs] 
                v_first = torch.empty_like(x)

                for i in range(self.n_layer):
                    bbb = f'blocks.{i}.'
                    att = f'blocks.{i}.att.'
                    ffn = f'blocks.{i}.ffn.'

                    # TimeMix
                    xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                    
                    xx, v_first = RWKV_x070_TMix_seq_batch(
                        i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                        z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                        z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                        z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                        z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                        z[att+'ln_x.weight'], z[att+'ln_x.bias'], state[2]
                    )
                    x = x + xx

                    # ChannelMix
                    xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                    xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                    x = x + xx

                state[2] += (end - start)
                
                if full_output:
                    all_outputs.append(x)
                else:
                    if end == total_len:
                        all_outputs.append(x[:, -1, :])

            x = torch.cat(all_outputs, dim=1) if full_output else all_outputs[0]
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = F.linear(x, z['head.weight'])
            
            return x


