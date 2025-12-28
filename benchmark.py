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
args.MODEL_NAME = "/mnt/sda1/rwkv_weights/rwkv7-g0a3-7.2b-20251029-ctx8192"

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

def test_sampling_function():
    print("开始测试Sampling函数...")
    from rwkv_batch.rwkv7 import Sampling
    batch_size = 2
    vocab_size = 65536  
    
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device='cuda')
    print(f"✅ 创建logits张量: shape={logits.shape}, device={logits.device}")
    
    # penalties (B, V) - 惩罚矩阵
    penalties = torch.zeros(batch_size, vocab_size, dtype=torch.float32, device='cuda')
    print(f"✅ 创建penalties张量: shape={penalties.shape}, device={penalties.device}")
    
    states = torch.ops.rwkv7_state_fwd_fp16.setup_rand(42, batch_size)
    print(f"✅ 创建随机状态: shape={states.shape}, device={states.device}")
    
    test_cases = [
        {
            "name": "基本采样 (top_k=-1, top_p=0.0)",
            "presence_penalty": 0.0,
            "repetition_penalty": 0.0,
            "penalty_decay": 0.0,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 0.0
        },
        {
            "name": "温度采样",
            "presence_penalty": 0.0,
            "repetition_penalty": 0.0,
            "penalty_decay": 0.0,
            "temperature": 0.8,
            "top_k": -1,
            "top_p": 0.0
        },
        {
            "name": "Top-K采样",
            "presence_penalty": 0.0,
            "repetition_penalty": 0.0,
            "penalty_decay": 0.0,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.0
        },
        {
            "name": "Top-P采样",
            "presence_penalty": 0.0,
            "repetition_penalty": 0.0,
            "penalty_decay": 0.0,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 0.9
        },
        {
            "name": "Top-K + Top-P采样",
            "presence_penalty": 0.0,
            "repetition_penalty": 0.0,
            "penalty_decay": 0.0,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9
        },
        {
            "name": "带重复惩罚",
            "presence_penalty": 0.5,
            "repetition_penalty": 0.8,
            "penalty_decay": 0.95,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 0.0
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1}: {case['name']} ---")
        try:
            output = Sampling(
                logits=logits,
                penalties=penalties,
                states=states,
                presence_penalty=case['presence_penalty'],
                repetition_penalty=case['repetition_penalty'],
                penalty_decay=case['penalty_decay'],
                temperature=case['temperature'],
                top_k=case['top_k'],
                top_p=case['top_p']
            )
            
            print(f"✅ 执行成功")
            print(f"  - 输入logits形状: {logits.shape}")
            print(f"  - 输出token形状: {output.shape}")
            print(f"  - 输出token: {output.tolist()}")
            print(f"  - 参数: temp={case['temperature']}, top_k={case['top_k']}, top_p={case['top_p']}")
            
            assert output.shape == (batch_size,), f"期望输出形状为({batch_size},)，实际为{output.shape}"
            assert output.dtype == torch.int32, f"期望输出类型为int32，实际为{output.dtype}"
            assert output.device.type == 'cuda', f"输出应在CUDA上，实际在{output.device}"
            
            invalid_tokens = (output < 0) | (output >= vocab_size)
            if invalid_tokens.any():
                print(f"⚠️  警告: 发现无效token: {output[invalid_tokens].tolist()}")
            else:
                print(f"  - 所有输出token都在有效范围内 [0, {vocab_size-1}]")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print(f"\n--- 完整性测试 ---")
    
    print("测试单批次采样...")
    single_logits = torch.randn(1, vocab_size, dtype=torch.float32, device='cuda')
    single_penalties = torch.zeros(1, vocab_size, dtype=torch.float32, device='cuda')
    single_states = torch.ops.rwkv7_state_fwd_fp16.setup_rand(43, 1)
    
    single_output = Sampling(
        logits=single_logits,
        penalties=single_penalties,
        states=single_states,
        temperature=1.0,
        top_k=-1,
        top_p=0.0
    )
    
    print(f"✅ 单批次测试成功: {single_output.item()}")
    
    print("\n🎉 所有测试完成!")

def test_sampling_with_realistic_logits():

    print("\n=== 测试真实场景的logits ===")
    
    from rwkv_batch.rwkv7 import Sampling
    
    batch_size = 1
    vocab_size = 65536 
    
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device='cuda')
    preferred_tokens = [100, 200, 300, 400, 500]
    logits[0, preferred_tokens] += 5.0  
    
    penalties = torch.zeros(batch_size, vocab_size, dtype=torch.float32, device='cuda')
    states = torch.ops.rwkv7_state_fwd_fp16.setup_rand(44, batch_size)
    
    print("测试不同温度设置下的采样结果:")
    for temp in [0.1, 0.5, 1.0, 2.0]:
        output = Sampling(
            logits=logits,
            penalties=penalties,
            states=states,
            temperature=temp,
            top_k=50,
            top_p=0.9
        )
        print(f"  温度 {temp}: token {output.item()}")
    
    print("✅ 真实场景测试完成")


print("CUDA设备信息:")
print(f"  - 设备数量: {torch.cuda.device_count()}")
print(f"  - 当前设备: {torch.cuda.current_device()}")
print(f"  - 设备名称: {torch.cuda.get_device_name()}")

# 运行测试
test_sampling_function()
test_sampling_with_realistic_logits()

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
# token = model.forward(tokenizer.encode(prompt), state, with_sampling=True)

x = model.z['emb.weight'][token]

static_input = torch.empty_like(x, device="cuda")
static_state = copy.deepcopy(state)
static_output = model.forward(static_input, static_state, with_sampling=True)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model.forward(static_input, static_state, with_sampling=True)

static_input.copy_(x)
for i in range(len(state)):
    static_state[i].copy_(state[i])
# static_output.copy_(out)
static_output[0] = token

times = []
all_times = []
t000 = time.perf_counter()
for i in range(0, LENGTH_PER_TRIAL):
    t00 = time.perf_counter()
    # token = sampler_simple(static_output, noise=0).item()
    token = static_output.item()
    all_tokens += [token]
    try:
        tmp = tokenizer.decode(all_tokens[out_last:], utf8_errors="strict")
        print(tmp, end="", flush=True) # only print when we have a valid utf-8 string
        out_last = i+1
    except:
        pass

    static_input.copy_(model.z['emb.weight'][token])

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

exit(0)
#######################################################################################################

xprint("Decode (batch)")

# for BSZ in [2**n for n in range(1,8)] + [128 + n for n in range(8, 512+8, 8)]:
for BSZ in [128, 256, 384]:
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
