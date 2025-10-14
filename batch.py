########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################
import types
from torch.nn import functional as F
import numpy as np
########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
#
# model download: https://huggingface.co/BlinkDL/rwkv7-g1
#
args.MODEL_NAME = "/home/alic-li/ComfyUI/models/RWKV/RWKV_v7_G1a_0.4B_Translate_ctx4096_20250915_latest"

print(f'\nUsing CUDA fp16. Loading {args.MODEL_NAME} ...\n')

from rwkv_batch.rwkv7 import RWKV_x070
model = RWKV_x070(args)

from rwkv_batch.utils import TRIE_TOKENIZER, sampler_simple_batch
tokenizer = TRIE_TOKENIZER("rwkv_batch/rwkv_vocab_v20230424.txt")

prompts = ["也许", "我看到", "他们发现", "我认为", "哈哈", "这是一个有趣的", "List of Emojis:"]
BATCH_SIZE = len(prompts)

state = model.generate_zero_state(BATCH_SIZE)
out = model.forward_batch([tokenizer.encode(prompt) for prompt in prompts], state)

tokens = []
GENERATE_LENGTH = 10
for i in range(GENERATE_LENGTH):
    new_tokens = sampler_simple_batch(out, temp=1).tolist()
    tokens.append(new_tokens)
    out = model.forward_batch(new_tokens, state)

tokens = np.transpose(np.array(tokens), axes=(1,0,2)).squeeze(-1)

print('\n')
for n in range(BATCH_SIZE):
    print(prompts[n], end='')
    print(tokenizer.decode(tokens[n], utf8_errors="ignore"))
    print('#'*80)
