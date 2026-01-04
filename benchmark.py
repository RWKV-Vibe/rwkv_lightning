########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time, random, json, math, gc
from tqdm import tqdm
from torch.nn import functional as F
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

SHOW_SPEED_PERCENTILE = 50

########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
#
# model download: https://huggingface.co/BlinkDL/rwkv7-g1
#
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.1b-20250728-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.4b-20250905-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-1.5b-20250429-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-2.9b-20250519-ctx4096"
args.MODEL_NAME = "/mnt/3f7ab3b2-e663-407a-831c-ee4789165577/rwkv_translate/rwkv7-g1b-1.5b-20251202-ctx8192"

print(f'\nUsing CUDA fp16. Loading {args.MODEL_NAME} ...\n')

from rwkv_batch.rwkv7 import RWKV_x070
model = RWKV_x070(args)

PARAM_BYTES = 2
active_params = 0
for k,v in model.z.items():
    if 'emb' not in k:
        active_params += v.numel()
active_GB = active_params/1e9*PARAM_BYTES
print(f'\nActive params = {round(active_params/1e9,2)} B = {round(active_GB,2)} GB (gigabytes)')

from rwkv_batch.utils import TRIE_TOKENIZER, sampler_simple, sampler_simple_batch
tokenizer = TRIE_TOKENIZER("rwkv_batch/rwkv_vocab_v20230424.txt")

########################################################################################################

def xprint(s):
    c0, c1 = 3, 80-len(s)-3
    print(f"\n{'#'*c0} {s} {'#'*c1}\n")

# xprint("Basic")

# prompt = "The Eiffel tower is in the city of"
# print(prompt)

# init_out = model.forward(tokenizer.encode(prompt), model.generate_zero_state(0))
# probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
# _, indices = torch.topk(probs, 5) # print top-5 possibilities
# for i in range(len(indices)):
#     token_id = indices[i].item()
#     token = tokenizer.decode([token_id])
#     token_prob = probs[token_id].item()
#     print(repr(token), f'[probability {token_prob:.2%}]')

# ########################################################################################################

# xprint("Batch")

# prompts = ["The apple can be", "The cat can't be", "Q: 1+1=?\nA: 1+1=2."]
# tokens = [tokenizer.encode(prompt) for prompt in prompts]

# print(tokens)
# for prompt in prompts:
#     print(prompt)
#     init_out = model.forward(tokenizer.encode(prompt), model.generate_zero_state(0))
#     probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
#     _, indices = torch.topk(probs, 5) # print top-5 possibilities
#     for i in range(len(indices)):
#         token_id = indices[i].item()
#         token = tokenizer.decode([token_id])
#         token_prob = probs[token_id].item()
#         print(repr(token), f'[probability {token_prob:.2%}]')

# init_outs = model.forward_batch(tokens, model.generate_zero_state(len(prompts)))
# for n in range(len(prompts)):
#     print(prompts[n])
#     init_out = init_outs[n]
#     probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
#     _, indices = torch.topk(probs, 5) # print top-5 possibilities
#     for i in range(len(indices)):
#         token_id = indices[i].item()
#         token = tokenizer.decode([token_id], utf8_errors="replace")
#         token_prob = probs[token_id].item()
#         print(repr(token), f'[probability {token_prob:.2%}]')
#     if n != len(prompts)-1:
#         print()

########################################################################################################

xprint("Decode")

prompt = "User: simulate SpaceX mars landing using python\n\nAssistant: <think"
LENGTH_PER_TRIAL = 256
TEMPERATURE = 1.0
TOP_P = 0.0
print(prompt, end="")

all_tokens = []
out_last = 0
state = model.generate_zero_state(0)
out = model.forward(tokenizer.encode(prompt), state)

times = []
all_times = []
t000 = time.perf_counter()
for i in range(LENGTH_PER_TRIAL):
    t00 = time.perf_counter()
    token = sampler_simple(out, noise=0).item()
    all_tokens += [token]
    try:
        tmp = tokenizer.decode(all_tokens[out_last:], utf8_errors="strict")
        print(tmp, end="", flush=True) # only print when we have a valid utf-8 string
        out_last = i+1
    except:
        pass
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.forward(token, state)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    all_times.append(t1 - t00)
times = np.percentile(times, SHOW_SPEED_PERCENTILE)
all_times = np.percentile(all_times, SHOW_SPEED_PERCENTILE)
print(f'\n\nToken/s = {round(1/times,2)} (forward), {round(1/all_times,2)} (full) || Bandwidth = {round(active_GB/times,2)} GB/s || {round(time.perf_counter()-t000,3)}s')

# exit(0)

# #######################################################################################################

xprint("Decode (CUDAGraph)")


prompt = "User: simulate SpaceX mars landing using python\n\nAssistant: <think"
LENGTH_PER_TRIAL = 256
TEMPERATURE = 1.0
TOP_P = 0.0
print(prompt, end="")

all_tokens = []
out_last = 0
state = model.generate_zero_state(0)
out = model.forward(tokenizer.encode(prompt), state)
token = sampler_simple(out, noise=0).item()

x = model.z['emb.weight'][token]

static_input = torch.empty_like(x, device="cuda")
static_state = [None, None, None]
static_state[0] = torch.empty_like(state[0], device="cuda")
static_state[1] = torch.empty_like(state[1], device="cuda")
static_state[2] = torch.empty_like(state[2], device="cuda")
static_output = torch.empty_like(out, device="cuda")

static_output = model.forward(static_input, static_state)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model.forward(static_input, static_state)

static_input.copy_(x)
static_state[0].copy_(state[0])
static_state[1].copy_(state[1])
static_state[2].copy_(state[2])
static_output.copy_(out)

times = []
all_times = []
t000 = time.perf_counter()
for i in range(0, LENGTH_PER_TRIAL):
    t00 = time.perf_counter()
    token = sampler_simple(static_output, noise=0).item()
    all_tokens += [token]
    try:
        tmp = tokenizer.decode(all_tokens[out_last:], utf8_errors="strict")
        print(tmp, end="", flush=True) # only print when we have a valid utf-8 string
        out_last = i+1
    except:
        pass

    x = model.z['emb.weight'][token]
    static_input.copy_(x)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    all_times.append(t1 - t00)
times = np.percentile(times, SHOW_SPEED_PERCENTILE)
all_times = np.percentile(all_times, SHOW_SPEED_PERCENTILE)
print(f'\n\nToken/s = {round(1/times,2)} (forward), {round(1/all_times,2)} (full) || Bandwidth = {round(active_GB/times,2)} GB/s || {round(time.perf_counter()-t000,3)}s')

#######################################################################################################

xprint("Decode (batch)")

# for BSZ in [2**n for n in range(1,8)] + [128 + n for n in range(8, 512+8, 8)]:
for BSZ in [512, 768, 920, 1024, 1248, 1536, 1792, 2048, 2304, 2560]:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    state = model.generate_zero_state(BSZ)

    time.sleep(1)
    if BSZ == 2:
        prompts = ["The apple can be", "The cat can't be"]
    else:
        prompts = ["The apple can be" for _ in range(BSZ)]
    nnn = len(prompts)
    tokens = [tokenizer.encode(prompt) for prompt in prompts]
    LENGTH_PER_TRIAL = 32
    # TEMPERATURE = 1.0
    # TOP_P = 0.0

    if BSZ == 2:
        print('wait', end='')
    all_tokens = []
    out = model.forward_batch(tokens, state)

    times = []
    all_times = []
    t000 = time.perf_counter()
    for i in range(LENGTH_PER_TRIAL):
        t00 = time.perf_counter()
        token = sampler_simple_batch(out, noise=0).tolist()
        all_tokens += [token]
        if BSZ == 2:
            print('.', end='', flush=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.forward_batch(token, state)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        all_times.append(t1 - t00)

    times = np.percentile(times, SHOW_SPEED_PERCENTILE)
    all_times = np.percentile(all_times, SHOW_SPEED_PERCENTILE)

    del state
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    if BSZ == 2:
        print('\n')
        for n in range(nnn):
            print(prompts[n], end='')
            aaa_tokens = []
            for i in range(LENGTH_PER_TRIAL):
                aaa_tokens += all_tokens[i][n]
            print(tokenizer.decode(aaa_tokens, utf8_errors="ignore"))
            print('#'*80)

    print(f'Bsz {BSZ} || Token/s = {round(nnn/times,2)} (forward), {round(nnn/all_times,2)} (full) || {round(time.perf_counter()-t000,3)}s')

######################################################################################################

xprint("Prefill")

base_memory = torch.cuda.memory_allocated()
print(f'Base memory after model loading: {base_memory/1e9:.2f} GB')

raw = open("test_batch_scripts/calibration_data_v5_rc.txt").read()
tokens = tokenizer.encode(raw)

for stage in range(8, 12+1):
    CTX_LEN = 2**stage
    loss = 0
    a = 0
    cnt = 0
    
    torch.cuda.reset_peak_memory_stats()
    
    times = []
    while a+CTX_LEN < len(tokens):
        src = tokens[a:a+CTX_LEN]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prob = model.forward(src[:-1], model.generate_zero_state(0), full_output=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
            
        prob = F.softmax(prob.float(), dim=-1)
        for j in range(CTX_LEN-1):
            loss -= math.log(prob[j][src[j+1]])
            cnt += 1
        a += CTX_LEN

    final_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    memory_delta_peak = peak_memory - base_memory
    memory_delta_final = final_memory - base_memory
    
    times = np.percentile(times, SHOW_SPEED_PERCENTILE)
    print(f'CTX_LEN {CTX_LEN} : avg loss {round(loss/cnt,4)} || prefill {round((CTX_LEN-1)/times)} token/s = {round((CTX_LEN-1)/times * active_params * 2/1e12, 2)} TFLOPS || Memory: +peak {memory_delta_peak/1e9:.2f}GB, +final {memory_delta_final/1e9:.2f}GB')


xprint("Batch Prefill")

raw = open("test_batch_scripts/calibration_data_v5_rc.txt").read()
tokens = tokenizer.encode(raw)
base_memory = torch.cuda.memory_allocated()
print(f'Base memory after model loading: {base_memory/1e9:.2f} GB')

for batch_size in [8, 16, 32, 64, 128, 256]:
    CTX_LEN = 4096
    total_loss = 0
    total_tokens = 0  
    
    src = tokens[:CTX_LEN] 
    batch_tokens = [src for _ in range(batch_size)]  
    torch.cuda.reset_peak_memory_stats()
    
    times = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = model.forward_seq_batch_trunk(batch_tokens, model.generate_zero_state(batch_size), trunk_len=64)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    
    probs = F.softmax(probs.float(), dim=-1)
    # for b in range(batch_size):
    #     for j in range(CTX_LEN-1):
    #         total_loss -= math.log(probs[b][j][batch_tokens[b][j+1]].item())
    #         total_tokens += 1

    final_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    memory_delta_peak = peak_memory - base_memory
    memory_delta_final = final_memory - base_memory
    
    avg_time = np.mean(times)
    processed_tokens = batch_size * (CTX_LEN - 1)
    tokens_per_sec = processed_tokens / avg_time
    
    # avg_loss = total_loss / total_tokens 
    
    print(f'Batch Size {batch_size}, CTX_LEN {CTX_LEN} : batch prefill {round(tokens_per_sec)} token/s || Memory: +peak {memory_delta_peak/1e9:.5f}GB, +final {memory_delta_final/1e9:.5f}GB')
    
    del probs
    torch.cuda.empty_cache()
    gc.collect()

exit(0)