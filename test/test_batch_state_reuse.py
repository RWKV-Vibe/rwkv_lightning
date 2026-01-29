import os, types, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import time
import torch
from infer.rwkv_batch.rwkv7 import RWKV_x070
from infer.rwkv_batch.utils import TRIE_TOKENIZER, sampler_simple

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
args.MODEL_NAME = "/mnt/pc411_data/rwkv_translate/rwkv7-g1b-1.5b-20251202-ctx8192"

print(f'\nLoading {args.MODEL_NAME} ...\n')
GEN_LENGTH = 50 
model = RWKV_x070(args)
tokenizer = TRIE_TOKENIZER("infer/rwkv_batch/rwkv_vocab_v20230424.txt")

def clone_batch_state_from_index(state, next_content_idx, batch_size):
    if state is None:
        raise ValueError("state is None")
    current_batch = state[2].numel()
    if current_batch < 1:
        raise ValueError("state batch size must be >= 1 for slicing")
    if next_content_idx < 0 or next_content_idx >= current_batch:
        raise ValueError(
            f"next_content_idx {next_content_idx} out of range for state batch size {current_batch}"
        )
    if state[0].dim() < 4 or state[1].dim() < 5:
        raise ValueError("state does not have batch dimension for slicing")
    base0 = state[0][:, :, next_content_idx:next_content_idx + 1, :]
    base1 = state[1][:, next_content_idx:next_content_idx + 1, :, :, :]
    base2 = state[2][next_content_idx:next_content_idx + 1]
    new0 = base0.repeat(1, 1, batch_size, 1).contiguous()
    new1 = base1.repeat(1, batch_size, 1, 1, 1).contiguous()
    new2 = base2.repeat(batch_size).contiguous()
    return [new0, new1, new2]

def run_batch_inference(prompts, state, label="Batch Infer"):
    batch_size = len(prompts)
    print(f"\n--- [{label}] 正在处理 {batch_size} 个并发序列 ---")
    
    tokens_batch = [tokenizer.encode(p) for p in prompts]
    
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    out = model.forward_batch(tokens_batch, state)
    
    all_generated_tokens = [[] for _ in range(batch_size)]
    active_mask = [True] * batch_size # 记录哪些 batch 还没遇到停止符
    
    for i in range(GEN_LENGTH):
        current_tokens = []
        for b in range(batch_size):
            token = sampler_simple(out[b], noise=0).item()
            current_tokens.append(token)
            
            if active_mask[b]:
                if token == 0: 
                    active_mask[b] = False
                else:
                    all_generated_tokens[b].append(token)
        
        if not any(active_mask):
            break
            
        out = model.forward_batch([[t] for t in current_tokens], state)

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    for idx, (prompt, gen) in enumerate(zip(prompts, all_generated_tokens)):
        decoded_text = tokenizer.decode(gen)
        print(f"\n[Batch {idx}] Prompt: {repr(prompt)}")
        print(f"[Batch {idx}] Response: {decoded_text}")
    
    print(f"\n[{label}] 总耗时: {t_end - t_start:.3f}s")
    return state


# 1. 初始化 Batch State (Batch Size = 2)
batch_prompts_1 = [
    "User: 你现在是一只可爱的猫娘，说话要带喵。懂了吗？\n\nAssistant: <think>\n</think>\n",
    "User: 你现在是一个严谨的数学家，说话要专业。懂了吗？\n\nAssistant: <think>\n</think>\n"
]
state = model.generate_zero_state(len(batch_prompts_1))

print(">>>> 第一轮：设定角色 (State 将被更新)")
state = run_batch_inference(batch_prompts_1, state, label="Turn 1")
state_expended = clone_batch_state_from_index(state, 0, 3)
print(state_expended[0].flatten()[:10])
print(state_expended[1].flatten()[:10])
print(state_expended[2].flatten()[:10])

# 2. 第二轮：复用 State 进行对话
batch_prompts_2 = [
    "\n\nUser: 请证明 1+1 为什么等于 2。\n\nAssistant: <think>\n</think>\n",
    "\n\nUser: 请证明 1+1 为什么等于 2。\n\nAssistant: <think>\n</think>\n",
    "\n\nUser: 请证明 1+1 为什么等于 2。\n\nAssistant: <think>\n</think>\n"
]

print("\n" + "="*50)
print(">>>> 第二轮：复用 State 测试")
print(f"当前 State 携带的 Token 计数: {state[2].cpu().numpy()}")
print("="*50)

state = run_batch_inference(batch_prompts_2, state_expended, label="Turn 2 (Reused)")