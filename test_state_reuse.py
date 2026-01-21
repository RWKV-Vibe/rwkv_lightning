import os, types
import time
import torch
import numpy as np
import argparse
from rwkv_batch.rwkv7 import RWKV_x070
from rwkv_batch.utils import TRIE_TOKENIZER, sampler_simple
args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
args.MODEL_NAME = "/mnt/3f7ab3b2-e663-407a-831c-ee4789165577/rwkv_translate/rwkv7-g1b-1.5b-20251202-ctx8192"

print(f'\nUsing CUDA fp16. Loading {args.MODEL_NAME} ...\n')
LENGTH_PER_TRIAL = 100
from rwkv_batch.rwkv7 import RWKV_x070
model = RWKV_x070(args)
tokenizer = TRIE_TOKENIZER("rwkv_batch/rwkv_vocab_v20230424.txt")
print("[INFO] Model loaded.\n")

# === 辅助函数：执行推理 ===
def run_inference(prompt, state, label="Infer"):
    print(f"\n--- [{label}] Prompt: {repr(prompt)} ---")
    print(f"[{label}] Output: ", end="", flush=True)

    # Encode & Forward (Prefill)
    input_tokens = tokenizer.encode(prompt)
    print(f"[{label}] Input tokens: {input_tokens}")
    
    # 记录时间
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    # Prefill
    out = model.forward(input_tokens, state)
    
    generated_tokens = []
    out_last = 0
    
    # Decode Loop
    for i in range(LENGTH_PER_TRIAL):
        token = sampler_simple(out, noise=0).item()
        
        # Stop tokens check (Standard RWKV stop tokens)
        if token in [0]: 
            break
            
        generated_tokens.append(token)
        
        # Print
        try:
            tmp = tokenizer.decode(generated_tokens[out_last:], utf8_errors="strict")
            print(tmp, end="", flush=True)
            out_last = i + 1
        except:
            pass
            
        # Forward (Decode)
        out = model.forward(token, state)
    
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    print(f"\n[{label}] Time cost: {t_end - t_start:.3f}s")
    
    return state

print("\n[TEST] Initializing State...")
state = model.generate_zero_state(0) 

# 2. 第一轮对话：设定人设
prompt_1 = "User: 现在你将模仿一只猫娘，与我对话每一句话后面都要加上“喵”\n\nAssistant: <think>\n</think>\n"
state = run_inference(prompt_1, state, label="Turn 1")

print("\n" + "="*50)
print(f"[TEST] State has been updated in-place. State[2] (Token Count): {state[2].item()}")
print("="*50)


prompt_2 = "\n\nUser: 我可以吃些什么嘛？我可以摸摸你的耳朵嘛？\n\nAssistant: <think>\n</think>\n"
state = run_inference(prompt_2, state, label="Turn 2 (State Reused)")

print("\n" + "="*50)
print(f"[TEST] Final Token Count: {state[2].item()}")
print("="*50)

# 4. (可选) 对比测试：如果没有 State 会怎样
# print("\n[TEST] Control Group: No State Reuse")
# empty_state = model.generate_zero_state(0)
# run_inference(prompt_2, empty_state, label="Turn 2 (No State)")