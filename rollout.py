import math, torch, types, copy, re
import numpy as np
from torch.nn import functional as F

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
#
# model download: https://huggingface.co/BlinkDL/rwkv7-g1
#
args.MODEL_NAME = "/home/alic-li/ComfyUI/models/RWKV/RWKV_v7_G1a_0.4B_Translate_ctx4096_20250915_latest"

prompt = 'User: Evaluate $(1+2i)6-3i$.\n\nAssistant: <think'
BATCH_SIZE = 320
GENERATION_LENGTH = 4000
BATCH_SIZE = 64
GENERATION_LENGTH = 1000

# we use simple sampling = greedy(logits + noise)
DECODE_NOISE = 1.0
DECODE_TEMP = 0.5
LOG_FILE = open("rollout.log", "w")

########################################################################################################

from rwkv_batch.rwkv7 import RWKV_x070
from rwkv_batch.utils import TRIE_TOKENIZER, sampler_simple_batch

# init model
print('loading...', args.MODEL_NAME)
model = RWKV_x070(args)
tokenizer = TRIE_TOKENIZER("rwkv_batch/rwkv_vocab_v20230424.txt")

# init state
state = model.generate_zero_state(BATCH_SIZE)
out = model.forward_batch([tokenizer.encode(prompt) for _ in range(BATCH_SIZE)], state)

all_out = []

print(f'rollout {GENERATION_LENGTH} tokens...', end=' ')

for i in range(GENERATION_LENGTH):
    token = sampler_simple_batch(out, noise=DECODE_NOISE, temp=DECODE_TEMP).tolist()
    all_out.append(token)
    out = model.forward_batch(token, state)
    if i % 10 == 0:
        print(i, end=' ', flush=True)
print('\n' + '#'*120 + '\n')

all_out = np.transpose(np.array(all_out), axes=(1,0,2)).squeeze(-1)

for n in range(BATCH_SIZE):
    tokens = all_out[n]
    
    eod = np.flatnonzero(tokens == 0)
    if eod.size:
        tokens = tokens[:eod[0]] # get tokens before eod (token 0)

    out_str = tokenizer.decode(tokens, utf8_errors="ignore").strip()
    LOG_FILE.write(out_str + '\n' + '#'*120 + '\n')
    if eod.size:
        print(out_str.splitlines()[-1].strip())
    else:
        print(f'(unfinished within {GENERATION_LENGTH} tokens)')
    print('#'*120)

LOG_FILE.close()
