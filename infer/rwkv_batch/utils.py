########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

import torch
import random

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
# MyModule = nn.Module
# def __nop(ob): return ob
# MyFunction = __nop
# MyStatic = __nop

def sample_logits(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
    from infer.rwkv_batch.sampler import sample

    logits_reshaped = logits.unsqueeze(0).float()
    contiguous = getattr(logits_reshaped, "contiguous", None)
    if callable(contiguous):
        logits_reshaped = contiguous()
    if temperature <= 0:
        temperature = 1.0
        top_k = 1
        top_p = 1.0

    rand_states = sample.setup_rand(random.randint(0, 2**63 - 1), 1)
    sampled = sample.batch_sampling_temperature_topk_topp(
        logits_reshaped,
        rand_states,
        temperature,
        top_k,
        top_p,
    )
    if hasattr(sampled, "view"):
        return sampled.view(-1)[0].item()
    if hasattr(sampled, "tolist"):
        values = sampled.tolist()
        return values[0] if isinstance(values, (list, tuple)) else values
    if hasattr(sampled, "item"):
        return sampled.item()
    return sampled

@torch.no_grad() # !!! will modify logits inplace !!!
@MyStatic
def sampler_gumbel_batch(logits: torch.Tensor, temp: float = 1.0, eps: float = 6.2e-5) -> torch.Tensor:
    if temp > 0:
        if temp != 1: logits.mul_(1.0 / temp)
        noise = torch.empty_like(logits).exponential_().clamp_(min=eps).log_().neg_()
        logits.add_(noise)
    return torch.argmax(logits, dim=-1, keepdim=True)

@MyStatic # !!! will modify logits inplace !!!
def sampler_simple_batch(logits: torch.Tensor, noise: float = 0.0, temp: float = 1.0):
    assert temp > 0, "use noise=0 for greedy decoding"
    with torch.no_grad():
        if temp != 1.0:
            logits.mul_(1.0 / temp)
        if noise != 0.0:
            logits.add_(torch.empty_like(logits).uniform_(0.0, noise))
        return torch.argmax(logits, dim=-1, keepdim=True)

@MyStatic # !!! will modify logits inplace !!!
def sampler_simple(logits: torch.Tensor, noise: float = 0.0, temp: float = 1.0):
    assert temp > 0, "use noise=0 for greedy decoding"
    with torch.no_grad():
        if temp != 1.0:
            logits.mul_(1.0 / temp)
        if noise != 0.0:
            logits.add_(torch.empty_like(logits).uniform_(0.0, noise))
        return torch.argmax(logits, dim=-1, keepdim=False)

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

class TRIE_TOKENIZER():
    def __init__(self, file_name):
        self.idx2token = {0:"<|endoftext|>".encode("utf-8")} # add <|endoftext|>
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            if k != 0: # ignore <|endoftext|>
                self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens, utf8_errors="strict"):
        return self.decodeBytes(tokens).decode('utf-8', errors=utf8_errors)

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()
