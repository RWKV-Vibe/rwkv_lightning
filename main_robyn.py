import argparse
import torch
import types
import json
import gc, re
import asyncio
from robyn import Robyn, Response, StreamingResponse, ALLOW_CORS
from pydantic import BaseModel
from rwkv_batch.rwkv7 import RWKV_x070
from rwkv_batch.utils import TRIE_TOKENIZER, sampler_simple_batch, sampler_simple
from rwkv_batch.sampler import sample
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True, help="RWKV model path")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--password", type=str, default=None, help="API password for authentication")
args_cli = parser.parse_args()
ROCm_Flag = torch.version.hip is not None

print(f"\n[INFO] Loading RWKV-7 model from {args_cli.model_path}\n")

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
if args_cli.model_path.endswith(".pth"):
    args.MODEL_NAME = re.sub(r'\.pth$', '', args_cli.model_path)
else:
    args.MODEL_NAME = args_cli.model_path

model = RWKV_x070(args)
tokenizer = TRIE_TOKENIZER("rwkv_batch/rwkv_vocab_v20230424.txt")

print(f"[INFO] Model loaded successfully.\n")


app = Robyn(__file__)
@app.after_request()
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response
@app.options("/")
@app.options("/v1/chat/completions")
@app.options("/v2/chat/completions")
@app.options("/v3/chat/completions")
@app.options("/translate/v1/batch-translate")
@app.options("/FIM/v1/batch-FIM")
async def handle_options():
    return Response(
        status_code=204,
        description="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"  
        }
    )


model_lock = Lock()
executor = ThreadPoolExecutor(max_workers=128, thread_name_prefix="model_inference")

class ChatRequest(BaseModel):
    model: str = "rwkv7"
    contents: list[str] = []              # 输入句子列表
    prefix: list[str] = []
    suffix: list[str] = [] 
    max_tokens: int = 50
    stop_tokens: list[int] = [0, 261, 24281]
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 0.3
    noise: float = 1.5
    stream: bool = False
    pad_zero:bool = True
    alpha_presence: float = 0.5
    alpha_frequency: float = 0.5
    alpha_decay: float = 0.996
    enable_think: bool = False
    chunk_size: int = 32
    password: str = None

def torch_top_k_top_p(logits, top_k, top_p):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1)) 
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, -float('Inf'))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :1] = False  
        
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('Inf'))
    
    probabilities = torch.softmax(logits, dim=-1)
    sampled_tokens = torch.multinomial(probabilities, 1).squeeze(-1)
    
    return sampled_tokens


def batch_generate(prompts, max_length=512, temperature=1.0, top_k=50, top_p=0.6, alpha_presence=1.0, alpha_frequency=0.1, alpha_decay=0.996, stop_tokens=[0, 261, 24281]):
    B = len(prompts)
    state = model.generate_zero_state(B)
    encoded_prompts = [tokenizer.encode(p) for p in prompts]
    out = model.forward_batch(encoded_prompts, state).float()

    finished = [False] * B
    generated_tokens = [[] for _ in range(B)]

    for step in range(max_length):
        sample_rand_states = sample.setup_rand(0, B)
        penalties = torch.zeros(B, 65536).to(0)
        new_tokens = sample.batch_sampling_repetition_temperature_topk_topp(out, penalties, sample_rand_states,
                                                                     alpha_presence, alpha_frequency, alpha_decay,
                                                                     temperature, top_k, top_p).tolist()
        # new_tokens = sampler_simple_batch(out, noise=noise, temp=temperature).tolist()
        new_tokens = [[token] for token in new_tokens]
        out = model.forward_batch(new_tokens, state).float()

        for i in range(B):
            tok = new_tokens[i][0] if isinstance(new_tokens[i], list) else new_tokens[i]
            if finished[i]:
                continue
            if tok in stop_tokens:
                finished[i] = True
                continue
            generated_tokens[i].append(tok)

        if all(finished):
            break
    del state
    gc.collect()

    decoded = []
    for i in range(B):
        text = tokenizer.decode(generated_tokens[i], utf8_errors="ignore")
        decoded.append(text)
    return decoded

async def batch_infer_stream(prompts, max_length=512, temperature=1.0, top_k=50, top_p=0.6, alpha_presence=1.0, alpha_frequency=0.1, alpha_decay=0.996, stop_tokens=[0, 261, 24281], chunk_size=32):
    B = len(prompts)
    state = model.generate_zero_state(B)
    encoded_prompts = [tokenizer.encode(p) for p in prompts]
    out = model.forward_batch(encoded_prompts, state).float()

    finished = [False] * B
    generated_tokens = [[] for _ in range(B)]
    token_buffers = [[] for _ in range(B)] 

    try:
        while not all(finished) and max_length > 0:
            sample_rand_states = sample.setup_rand(0, B)
            penalties = torch.zeros(B, 65536).to(0)
            new_tokens = sample.batch_sampling_repetition_temperature_topk_topp(out, penalties, sample_rand_states,
                                                                         alpha_presence, alpha_frequency, alpha_decay,
                                                                         temperature, top_k, top_p).tolist()
            # new_tokens = sampler_simple_batch(out, noise=noise, temp=temperature).tolist()
            new_tokens = [[token] for token in new_tokens]
            out = model.forward_batch(new_tokens, state).float()
            max_length -= 1

            contents_to_send = [""] * B
            
            for i in range(B):
                if finished[i]:
                    continue
                    
                tok = new_tokens[i][0] if isinstance(new_tokens[i], list) else new_tokens[i]
                
                if tok in stop_tokens:
                    finished[i] = True
                    if token_buffers[i]:
                        contents_to_send[i] = tokenizer.decode(token_buffers[i], utf8_errors="ignore")
                        token_buffers[i].clear()
                    continue
                
                token_buffers[i].append(tok)
                generated_tokens[i].append(tok)
                
                if len(token_buffers[i]) >= chunk_size:
                    contents_to_send[i] = tokenizer.decode(token_buffers[i], utf8_errors="ignore")
                    token_buffers[i].clear()
            
            if any(contents_to_send):
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": i, "delta": {"content": contents_to_send[i]}} 
                        for i in range(B) if contents_to_send[i]
                    ]
                }
                if chunk["choices"]:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            
            await asyncio.sleep(0)
        
        remaining_contents = [""] * B
        for i in range(B):
            if token_buffers[i]:
                remaining_contents[i] = tokenizer.decode(token_buffers[i], utf8_errors="ignore")
                token_buffers[i].clear()
        
        if any(remaining_contents):
            chunk = {
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": i, "delta": {"content": remaining_contents[i]}}
                    for i in range(B) if remaining_contents[i]
                ]
            }
            if chunk["choices"]:
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
    finally:
        del state
        gc.collect()

    yield "data: [DONE]\n\n"

async def graph_generate(
    inputs,
    stop_tokens,
    max_generate_tokens,
    temperature=1.0,
    top_k=50,
    top_p=0.3,
    alpha_presence=0.5,
    alpha_frequency=0.5,
    alpha_decay=0.996,
):
    # Only BSZ=1 
    prompt = inputs[0]
    encoded_prompt = tokenizer.encode(prompt)
    state = model.generate_zero_state(0)
    
    out = model.forward(encoded_prompt, state)
    
    # 初始采样 (获取 prefill 后的第一个 token)
    # 用 sampler_simple 取值用于构建 graph 反正只是 prepare CUDA Graph 又没事哈哈，后续循环会用完整采样逻辑
    token = sampler_simple(out, noise=0, temp=temperature).item()
    
    x_emb = model.z['emb.weight'][token]
    
    static_input = torch.empty_like(x_emb, device="cuda")
    static_state = [None, None, None]
    static_state[0] = torch.empty_like(state[0], device="cuda")
    static_state[1] = torch.empty_like(state[1], device="cuda")
    static_state[2] = torch.empty_like(state[2], device="cuda")
    static_output = torch.empty_like(out, device="cuda")

    # Warmup run 
    static_output = model.forward(static_input, static_state)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = model.forward(static_input, static_state)

    # Copy to static_state (Context Transfer)
    static_input.copy_(x_emb)
    static_state[0].copy_(state[0])
    static_state[1].copy_(state[1])
    static_state[2].copy_(state[2])
    static_output.copy_(out)
    
    generated_tokens = [token]
    
    for _ in range(max_generate_tokens):
            
        x_emb = model.z['emb.weight'][token]
        static_input.copy_(x_emb)
        
        # Replay Graph
        g.replay()
        
        # sampling (必须在 Graph 外进行, 否则会被这该死的随机数生成器炸了的)        
        # 增加 batch 维度适配采样函数 [Vocab] -> [1, Vocab]
        logits_reshaped = static_output.unsqueeze(0).float()
        
        sample_rand_states = sample.setup_rand(0, 1)
        penalties = torch.zeros(1, 65536).to(0)
        new_tokens = sample.batch_sampling_repetition_temperature_topk_topp(logits_reshaped, penalties, sample_rand_states,
                                                                     alpha_presence, alpha_frequency, alpha_decay,
                                                                     temperature, top_k, top_p).tolist()
        token = new_tokens[0]
        generated_tokens.append(token)
        if token in stop_tokens:
            break

    del state
    del static_state
    del static_input
    del static_output
    del g
    gc.collect()

    decoded = tokenizer.decode(generated_tokens, utf8_errors="ignore")
    return [decoded]


async def graph_infer_stream(
    inputs,
    stop_tokens,
    max_generate_tokens,
    temperature=1.0,
    top_k=50,
    top_p=0.3,
    alpha_presence=0.5,
    alpha_frequency=0.5,
    alpha_decay=0.996,
    chunk_size=32,
):
    prompt = inputs[0]
    encoded_prompt = tokenizer.encode(prompt)
    
    state = model.generate_zero_state(0)
    
    out = model.forward(encoded_prompt, state)
    
    # 初始采样 (获取 prefill 后的第一个 token)
    # 用 sampler_simple 取值用于构建 graph 反正只是 prepare CUDA Graph 又没事哈哈，后续循环会用完整采样逻辑
    token = sampler_simple(out, noise=0, temp=temperature).item()
    
    x_emb = model.z['emb.weight'][token]
    
    static_input = torch.empty_like(x_emb, device="cuda")
    static_state = [None, None, None]
    static_state[0] = torch.empty_like(state[0], device="cuda")
    static_state[1] = torch.empty_like(state[1], device="cuda")
    static_state[2] = torch.empty_like(state[2], device="cuda")
    static_output = torch.empty_like(out, device="cuda")

    static_output = model.forward(static_input, static_state)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = model.forward(static_input, static_state)
        
    static_input.copy_(x_emb)
    static_state[0].copy_(state[0])
    static_state[1].copy_(state[1])
    static_state[2].copy_(state[2])
    static_output.copy_(out)
    
    token_buffer = [token]  # 只在buffer中维护需要输出的token
    
    try:
        # 检查初始token是否为停止token
        if token not in stop_tokens:
            if len(token_buffer) >= chunk_size:
                content = tokenizer.decode(token_buffer, utf8_errors="ignore")
                token_buffer.clear()
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": content}}]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        for _ in range(max_generate_tokens):
            x_emb = model.z['emb.weight'][token]
            static_input.copy_(x_emb)
            
            # cuda graph Replay
            g.replay()

            # sampling (必须在 Graph 外进行, 否则会被这该死的随机数生成器炸了的)        
            # 增加 batch 维度适配采样函数 [Vocab] -> [1, Vocab]
            logits_reshaped = static_output.unsqueeze(0).float()
            
            sample_rand_states = sample.setup_rand(0, 1)
            penalties = torch.zeros(1, 65536).to(0)
            new_tokens = sample.batch_sampling_repetition_temperature_topk_topp(logits_reshaped, penalties, sample_rand_states,
                                                                         alpha_presence, alpha_frequency, alpha_decay,
                                                                         temperature, top_k, top_p).tolist()
            token = new_tokens[0]  
            if token in stop_tokens:
                break

            # 添加到buffer中准备输出
            token_buffer.append(token)
            
            # 发送 Chunk
            if len(token_buffer) >= chunk_size:
                content = tokenizer.decode(token_buffer, utf8_errors="ignore")
                token_buffer.clear()
                if content:  # 只有在内容非空时才发送
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": content}}]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            
            await asyncio.sleep(0) # 让出控制权
            
        # 发送剩余 buffer
        if token_buffer:
            content = tokenizer.decode(token_buffer, utf8_errors="ignore")
            if content:  # 只有在内容非空时才发送
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": content}}]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
    finally:
        del state
        del static_state
        del static_input
        del static_output
        del g
        gc.collect()

    yield "data: [DONE]\n\n"

def _continuous_batching_stream_sync(
    model,
    tokenizer,
    inputs,
    stop_tokens,
    max_generate_tokens,
    batch_size,
    output_queue,  # 用于传递实时输出的队列
    pad_zero=True,
    temperature=1,
    top_k=50,
    top_p=0.3,
    alpha_presence=0.5,
    alpha_frequency=0.5,
    alpha_decay=0.996,
    chunk_size=32,
):
    """
    同步版本：执行模型推理，通过队列实时输出chunks
    这个函数在后台线程中运行，不会阻塞事件循环
    """
    STOP_TOKENS = stop_tokens
    MAX_GENERATE_TOKENS = max_generate_tokens
    BATCH_SIZE = batch_size
    PAD_ZERO = pad_zero
    CHUNK_SIZE = chunk_size
    
    device = model.z["head.weight"].device
    alpha_presence_val = torch.tensor(alpha_presence, dtype=torch.float32, device=device)
    
    if temperature == 0:
        temperature = 1.0
        top_k = 1
    
    # 准备输入
    encoded_inputs = []
    for prompt in inputs:
        input_token = tokenizer.encode(prompt)
        if PAD_ZERO:
            input_token = [0] + input_token
        encoded_inputs.append((prompt, input_token))
    input_queue = deque(encoded_inputs)
    
    # 初始化状态
    states = model.generate_zero_state(BATCH_SIZE)
    task_pool = []
    token_buffers = {}
    
    prompt_idx = 0
    for i in range(BATCH_SIZE):
        prompt, input_token = input_queue.popleft()
        task_pool.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "input_token": input_token,
            "state_pos": i,
            "generated_tokens": [],
            "new_token": None,
        })
        token_buffers[prompt_idx] = []
        prompt_idx += 1
    
    occurrence = torch.zeros((BATCH_SIZE, args.vocab_size), dtype=torch.float32, device=device)
    no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
    alpha_presence_vector = torch.zeros((BATCH_SIZE, args.vocab_size), dtype=torch.float32, device=device)
    
    try:
        while True:
            contents_to_send = {}
            accomplished_task_indices = []
            state_slots_to_remove = set()
            
            # 检查任务状态
            for task_idx, task in enumerate(task_pool):
                if len(task["input_token"]) == 0:
                    if task["new_token"] is None:
                        continue
                    
                    new_token = task["new_token"]
                    prompt_id = task["prompt_idx"]
                    
                    is_finished = (new_token in STOP_TOKENS or 
                                 len(task["generated_tokens"]) >= MAX_GENERATE_TOKENS)
                    
                    if not is_finished:
                        task["generated_tokens"].append(new_token)
                        token_buffers[prompt_id].append(new_token)
                        
                        if len(token_buffers[prompt_id]) >= CHUNK_SIZE:
                            text_chunk = tokenizer.decode(token_buffers[prompt_id], utf8_errors="ignore")
                            contents_to_send[prompt_id] = text_chunk
                            token_buffers[prompt_id].clear()
                    
                    if is_finished:
                        if token_buffers[prompt_id]:
                            text_chunk = tokenizer.decode(token_buffers[prompt_id], utf8_errors="ignore")
                            contents_to_send[prompt_id] = contents_to_send.get(prompt_id, "") + text_chunk
                            token_buffers[prompt_id].clear()
                        
                        del token_buffers[prompt_id]
                        
                        if len(input_queue) > 0:
                            prompt, input_token = input_queue.popleft()
                            new_prompt_idx = prompt_idx
                            task_pool[task_idx] = {
                                "prompt_idx": new_prompt_idx,
                                "prompt": prompt,
                                "input_token": input_token,
                                "state_pos": task["state_pos"],
                                "generated_tokens": [],
                                "new_token": None,
                            }
                            token_buffers[new_prompt_idx] = []
                            prompt_idx += 1
                            
                            state_pos = task["state_pos"]
                            states[0][:, :, state_pos, :] = 0
                            states[1][:, state_pos, :, :] = 0
                            occurrence[state_pos, :] = 0
                            alpha_presence_vector[state_pos, :] = 0
                        else:
                            accomplished_task_indices.append(task_idx)
                            state_slots_to_remove.add(task["state_pos"])
                    else:
                        task["input_token"].append(new_token)
                        www = 0.0 if new_token in no_penalty_token_ids else 1.0
                        occurrence[task["state_pos"], new_token] += www
                        alpha_presence_vector[task["state_pos"], new_token] = alpha_presence_val
            
            # 实时发送chunks
            if contents_to_send:
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": pid, "delta": {"content": content}}
                        for pid, content in contents_to_send.items() if content
                    ]
                }
                if chunk["choices"]:
                    output_queue.put(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")
            
            # 压缩状态张量
            if accomplished_task_indices:
                sorted_slots = sorted(list(state_slots_to_remove), reverse=True)
                
                for slot in sorted_slots:
                    states[0] = torch.cat([states[0][:, :, :slot, :], states[0][:, :, slot+1:, :]], dim=2)
                    states[1] = torch.cat([states[1][:, :slot, :, :], states[1][:, slot+1:, :, :]], dim=1)
                    occurrence = torch.cat([occurrence[:slot, :], occurrence[slot+1:, :]], dim=0)
                    alpha_presence_vector = torch.cat([alpha_presence_vector[:slot, :], 
                                                       alpha_presence_vector[slot+1:, :]], dim=0)
                
                for task_idx in sorted(accomplished_task_indices, reverse=True):
                    del task_pool[task_idx]
                
                remaining_slots = sorted([t["state_pos"] for t in task_pool])
                pos_map = {old_pos: new_pos for new_pos, old_pos in enumerate(remaining_slots)}
                for task in task_pool:
                    task["state_pos"] = pos_map[task["state_pos"]]
            
            if len(task_pool) == 0:
                break
            
            # 准备下一批tokens
            current_batch_size = len(task_pool)
            next_tokens = [None] * current_batch_size
            for task in task_pool:
                next_tokens[task["state_pos"]] = [task["input_token"].pop(0)]
            
            # 模型前向传播
            out = model.forward_batch(next_tokens, states)

            if alpha_presence != 0 or alpha_frequency != 0:
                mask = (occurrence > 0).float()
                out -= (mask * alpha_presence + occurrence * alpha_frequency)
            
            # 应用惩罚和采样
            occurrence *= alpha_decay
            out -= alpha_presence_vector + occurrence * alpha_frequency
            
            if temperature != 1.0:
                out /= temperature
            
            if ROCm_Flag:
                new_tokens = torch_top_k_top_p(out, top_k, top_p)
            else:
                try:
                    import flashinfer # type: ignore
                    new_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(out, top_k, top_p)
                except:
                    new_tokens = torch_top_k_top_p(out, top_k, top_p)
            
            new_tokens = new_tokens.tolist()
            
            for task in task_pool:
                state_pos = task["state_pos"]
                task["new_token"] = new_tokens[state_pos]
    
    finally:
        del states
        del occurrence
        del alpha_presence_vector
        gc.collect()
        output_queue.put("EOF")  # 发送结束信号


def _continuous_batching_sync(
    model,
    tokenizer,
    inputs,
    stop_tokens,
    max_generate_tokens,
    batch_size,
    pad_zero=True,
    temperature=1,
    top_k=50,
    top_p=0.3,
    alpha_presence=0.5,
    alpha_frequency=0.5,
    alpha_decay=0.996,
):
    """
    同步版本：执行模型推理，直接返回所有生成的结果
    非流式输出，等待所有任务完成后一次性返回
    """
    STOP_TOKENS = stop_tokens
    MAX_GENERATE_TOKENS = max_generate_tokens
    BATCH_SIZE = batch_size
    PAD_ZERO = pad_zero
    
    device = model.z["head.weight"].device
    alpha_presence_val = torch.tensor(alpha_presence, dtype=torch.float32, device=device)
    
    if temperature == 0:
        temperature = 1.0
        top_k = 1
    
    # 准备输入
    encoded_inputs = []
    for prompt in inputs:
        input_token = tokenizer.encode(prompt)
        if PAD_ZERO:
            input_token = [0] + input_token
        encoded_inputs.append((prompt, input_token))
    input_queue = deque(encoded_inputs)
    
    # 初始化状态
    states = model.generate_zero_state(BATCH_SIZE)
    task_pool = []
    # 用于存储每个prompt的完整生成文本
    results = {}
    
    prompt_idx = 0
    for i in range(BATCH_SIZE):
        prompt, input_token = input_queue.popleft()
        task_pool.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "input_token": input_token,
            "state_pos": i,
            "generated_tokens": [],
            "new_token": None,
        })
        prompt_idx += 1
    
    occurrence = torch.zeros((BATCH_SIZE, args.vocab_size), dtype=torch.float32, device=device)
    no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
    alpha_presence_vector = torch.zeros((BATCH_SIZE, args.vocab_size), dtype=torch.float32, device=device)
    
    try:
        while True:
            accomplished_task_indices = []
            state_slots_to_remove = set()
            
            # 检查任务状态
            for task_idx, task in enumerate(task_pool):
                if len(task["input_token"]) == 0:
                    if task["new_token"] is None:
                        continue
                    
                    new_token = task["new_token"]
                    prompt_id = task["prompt_idx"]
                    
                    is_finished = (new_token in STOP_TOKENS or 
                                 len(task["generated_tokens"]) >= MAX_GENERATE_TOKENS)
                    
                    if not is_finished:
                        task["generated_tokens"].append(new_token)
                    
                    if is_finished:
                        # 将生成的tokens解码为文本，存储到results中
                        if task["generated_tokens"]:
                            text = tokenizer.decode(task["generated_tokens"], utf8_errors="ignore")
                            results[prompt_id] = text
                        else:
                            results[prompt_id] = ""
                        
                        if len(input_queue) > 0:
                            prompt, input_token = input_queue.popleft()
                            new_prompt_idx = prompt_idx
                            task_pool[task_idx] = {
                                "prompt_idx": new_prompt_idx,
                                "prompt": prompt,
                                "input_token": input_token,
                                "state_pos": task["state_pos"],
                                "generated_tokens": [],
                                "new_token": None,
                            }
                            prompt_idx += 1
                            
                            state_pos = task["state_pos"]
                            states[0][:, :, state_pos, :] = 0
                            states[1][:, state_pos, :, :] = 0
                            occurrence[state_pos, :] = 0
                            alpha_presence_vector[state_pos, :] = 0
                        else:
                            accomplished_task_indices.append(task_idx)
                            state_slots_to_remove.add(task["state_pos"])
                    else:
                        task["input_token"].append(new_token)
                        www = 0.0 if new_token in no_penalty_token_ids else 1.0
                        occurrence[task["state_pos"], new_token] += www
                        alpha_presence_vector[task["state_pos"], new_token] = alpha_presence_val
            
            # 压缩状态张量
            if accomplished_task_indices:
                sorted_slots = sorted(list(state_slots_to_remove), reverse=True)
                
                for slot in sorted_slots:
                    states[0] = torch.cat([states[0][:, :, :slot, :], states[0][:, :, slot+1:, :]], dim=2)
                    states[1] = torch.cat([states[1][:, :slot, :, :], states[1][:, slot+1:, :, :]], dim=1)
                    occurrence = torch.cat([occurrence[:slot, :], occurrence[slot+1:, :]], dim=0)
                    alpha_presence_vector = torch.cat([alpha_presence_vector[:slot, :], 
                                                       alpha_presence_vector[slot+1:, :]], dim=0)
                
                for task_idx in sorted(accomplished_task_indices, reverse=True):
                    del task_pool[task_idx]
                
                remaining_slots = sorted([t["state_pos"] for t in task_pool])
                pos_map = {old_pos: new_pos for new_pos, old_pos in enumerate(remaining_slots)}
                for task in task_pool:
                    task["state_pos"] = pos_map[task["state_pos"]]
            
            if len(task_pool) == 0:
                break
            
            # 准备下一批tokens
            current_batch_size = len(task_pool)
            next_tokens = [None] * current_batch_size
            for task in task_pool:
                next_tokens[task["state_pos"]] = [task["input_token"].pop(0)]
            
            # 模型前向传播
            out = model.forward_batch(next_tokens, states)

            if alpha_presence != 0 or alpha_frequency != 0:
                mask = (occurrence > 0).float()
                out -= (mask * alpha_presence + occurrence * alpha_frequency)
            
            # 应用惩罚和采样
            occurrence *= alpha_decay
            out -= alpha_presence_vector + occurrence * alpha_frequency
            
            if temperature != 1.0:
                out /= temperature
            
            if ROCm_Flag:
                new_tokens = torch_top_k_top_p(out, top_k, top_p)
            else:
                try:
                    import flashinfer # type: ignore
                    new_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(out, top_k, top_p)
                except:
                    new_tokens = torch_top_k_top_p(out, top_k, top_p)
            
            new_tokens = new_tokens.tolist()
            
            for task in task_pool:
                state_pos = task["state_pos"]
                task["new_token"] = new_tokens[state_pos]
    
    finally:
        del states
        del occurrence
        del alpha_presence_vector
        gc.collect()
    
    # 返回结果列表，按照输入顺序
    return [results.get(i, "") for i in range(len(inputs))]


async def continuous_batching_stream(
    model,
    tokenizer,
    inputs,
    stop_tokens,
    max_generate_tokens,
    batch_size,
    pad_zero=True,
    temperature=1,
    top_k=50,
    top_p=0.3,
    alpha_presence=0.5,
    alpha_frequency=0.5,
    alpha_decay=0.996,
    chunk_size=32,
):
    from queue import Queue
    
    # 创建队列用于接收后台线程的输出
    output_queue = Queue()
    
    loop = asyncio.get_event_loop()
    
    # 在线程池中执行推理，传递队列用于实时输出
    with model_lock:  
        future = loop.run_in_executor(
            executor,
            _continuous_batching_stream_sync,
            model,
            tokenizer,
            inputs,
            stop_tokens,
            max_generate_tokens,
            batch_size,
            output_queue,  # 传递队列
            pad_zero,
            temperature,
            top_k,
            top_p,
            alpha_presence,
            alpha_frequency,
            alpha_decay,
            chunk_size,
        )
    
    # 持续从队列读取并yield输出
    while True:
        # 使用asyncio.to_thread或者定期检查队列
        try:
            await asyncio.sleep(0.01)  # 短暂等待避免忙等待
            
            # 非阻塞地从队列中取出数据
            while not output_queue.empty():
                data = output_queue.get_nowait()
                if data == "EOF":
                    yield "data: [DONE]\n\n"
                    # 等待后台任务完成
                    await future
                    return
                yield data
            
            # 检查future是否已完成
            if future.done():
                # 处理剩余的队列数据
                while not output_queue.empty():
                    data = output_queue.get_nowait()
                    if data == "EOF":
                        yield "data: [DONE]\n\n"
                        return
                    yield data
                break
        except Exception as e:
            print(f"Error in stream: {e}")
            break
    
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request):
    body = json.loads(request.body)
    req = ChatRequest(**body)

    if args_cli.password and req.password != args_cli.password:
        return Response(
            status_code=401,
            description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
            headers={"Content-Type": "application/json"}
        )

    prompts = req.contents

    # print(f"[REQUEST] /v1/: {req.model_dump()}")
    # print(f"[REQUEST] /v1/: {prompts}")

    if req.stream:
        return StreamingResponse(
            batch_infer_stream(
                prompts = prompts,
                max_length = req.max_tokens,
                temperature = req.temperature,
                top_k = req.top_k,
                top_p = req.top_p,
                alpha_presence = req.alpha_presence,
                alpha_frequency = req.alpha_frequency,
                alpha_decay = req.alpha_decay,
                stop_tokens=req.stop_tokens,
                chunk_size = req.chunk_size,
            ),
            media_type="text/event-stream"
        )

    results = batch_generate(                
                prompts = prompts,
                max_length = req.max_tokens,
                temperature = req.temperature,
                top_k = req.top_k,
                top_p = req.top_p,
                alpha_presence = req.alpha_presence,
                alpha_frequency = req.alpha_frequency,
                alpha_decay = req.alpha_decay,
                stop_tokens=req.stop_tokens,
            )
    choices = []
    for i, text in enumerate(results):
        choices.append({
            "index": i,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        })

    response = {
        "id": "rwkv7-batch",
        "object": "chat.completion",
        "model": req.model,
        "choices": choices,
    }
    return Response(
        status_code=200,
        description=json.dumps(response, ensure_ascii=False),
        headers={"Content-Type": "application/json"}
    )

@app.post("/v2/chat/completions")
async def continuous_batching(request):
    try:
        body = json.loads(request.body)
        req = ChatRequest(**body)

        if args_cli.password and req.password != args_cli.password:
            return Response(
                status_code=401,
                description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
                headers={"Content-Type": "application/json"}
            )

        prompts = req.contents
        

        if not prompts:
            return Response(
                status_code=400,
                description=json.dumps({"error": "Empty prompts list"}),
                headers={"Content-Type": "application/json"}
            )

        if req.stream:
            return StreamingResponse(
                continuous_batching_stream(model=model,
                                           tokenizer=tokenizer,
                                           inputs=prompts,
                                           stop_tokens=req.stop_tokens,
                                           max_generate_tokens=req.max_tokens,
                                           batch_size=len(prompts),
                                           pad_zero=req.pad_zero,
                                           temperature=req.temperature,
                                           top_k=req.top_k,
                                           top_p=req.top_p,
                                           alpha_presence=req.alpha_presence,
                                           alpha_frequency=req.alpha_frequency,
                                           alpha_decay=req.alpha_decay,
                                           chunk_size=req.chunk_size),
                media_type="text/event-stream"
            )

        results = _continuous_batching_sync(model=model,
                                            tokenizer=tokenizer,
                                            inputs=prompts,
                                            stop_tokens=req.stop_tokens,
                                            max_generate_tokens=req.max_tokens,
                                            batch_size=len(prompts),
                                            pad_zero=req.pad_zero,
                                            temperature=req.temperature,
                                            top_k=req.top_k,
                                            top_p=req.top_p,
                                            alpha_presence=req.alpha_presence,
                                            alpha_frequency=req.alpha_frequency,
                                            alpha_decay=req.alpha_decay)
        choices = []
        for i, text in enumerate(results):
            choices.append({
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            })

        response = {
            "id": "rwkv7-batch",
            "object": "chat.completion",
            "model": req.model,
            "choices": choices,
        }
        return Response(
            status_code=200,
            description=json.dumps(response, ensure_ascii=False),
            headers={"Content-Type": "application/json"}
        )
    except json.JSONDecodeError as e:
        return Response(
            status_code=400,
            description=json.dumps({"error": f"Invalid JSON: {str(e)}"}),
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        import traceback
        print(f"[ERROR] /v2/chat/completions: {traceback.format_exc()}")
        return Response(
            status_code=500,
            description=json.dumps({"error": str(e)}),
            headers={"Content-Type": "application/json"}
        )

@app.post("/v3/chat/completions")
async def v3_chat_completions(request):
    try:
        body = json.loads(request.body)
        # 若传入 messages，则提取其中的 user 内容作为 contents
        if "contents" not in body and "messages" in body:
            msgs = body.get("messages") or []
            # 提取所有 user 内容；若无，则取所有 content
            user_texts = [m.get("content", "") for m in msgs if m.get("role") == "user"]
            if not user_texts and msgs:
                user_texts = [m.get("content", "") for m in msgs]
            body = {**body, "contents": user_texts}

        req = ChatRequest(**body)

        if args_cli.password and req.password != args_cli.password:
            return Response(
                status_code=401,
                description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
                headers={"Content-Type": "application/json"}
            )

        prompts = req.contents
        # 将输入问题转换为指定提示模板："User: 问题\n\nAssistant:"
        if req.enable_think:
            prompts_formatted = [f"User: {q}\n\nAssistant: <think" for q in prompts]
        else:
            prompts_formatted = [f"User: {q}\n\nAssistant: <think>\n</think>" for q in prompts]

        if not prompts:
            return Response(
                status_code=400,
                description=json.dumps({"error": "Empty prompts list"}),
                headers={"Content-Type": "application/json"}
            )

        if req.stream:
            return StreamingResponse(
                graph_infer_stream(inputs=prompts_formatted,
                                           stop_tokens=req.stop_tokens,
                                           max_generate_tokens=req.max_tokens,
                                           temperature=req.temperature,
                                           top_k=req.top_k,
                                           top_p=req.top_p,
                                           alpha_presence=req.alpha_presence,
                                           alpha_frequency=req.alpha_frequency,
                                           alpha_decay=req.alpha_decay,
                                           chunk_size=req.chunk_size),
                media_type="text/event-stream"
            )

        results = await graph_generate(inputs=prompts_formatted,
                                            stop_tokens=req.stop_tokens,
                                            max_generate_tokens=req.max_tokens,
                                            temperature=req.temperature,
                                            top_k=req.top_k,
                                            top_p=req.top_p,
                                            alpha_presence=req.alpha_presence,
                                            alpha_frequency=req.alpha_frequency,
                                            alpha_decay=req.alpha_decay)
        choices = []
        for i, text in enumerate(results):
            choices.append({
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            })

        response = {
            "id": "rwkv7-batch-v3",
            "object": "chat.completion",
            "model": req.model,
            "choices": choices,
        }
        return Response(
            status_code=200,
            description=json.dumps(response, ensure_ascii=False),
            headers={"Content-Type": "application/json"}
        )
    except json.JSONDecodeError as e:
        return Response(
            status_code=400,
            description=json.dumps({"error": f"Invalid JSON: {str(e)}"}),
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        import traceback
        print(f"[ERROR] /v3/chat/completions: {traceback.format_exc()}")
        return Response(
            status_code=500,
            description=json.dumps({"error": str(e)}),
            headers={"Content-Type": "application/json"}
        )

#=== RWKV-7 Batch Translate Server ===#

class TranslateRequest(BaseModel):
    source_lang: str = "auto"
    target_lang: str
    text_list: list[str]
    placeholders: list[str] = None

class TranslateResponse(BaseModel):
    translations: list[dict]

def create_translation_prompt(source_lang, target_lang, text):
    lang_names = {
        "zh-CN": "Chinese",
        "zh-TW": "Chinese",  
        "en": "English",
        "ja": "Japanese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
    }
    
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    
    prompt = f"{source_name}: {text}\n\n{target_name}:"
    return prompt

@app.post("/translate/v1/batch-translate")
async def batch_translate(request):
    body = json.loads(request.body)
    req = TranslateRequest(**body)

    # print(f"[REQUEST] /translate/v1/batch-translate: {req.model_dump()}")

    try:
        processed_texts = req.text_list
        
        prompts = []
        for text in processed_texts:
            prompt = create_translation_prompt(req.source_lang, req.target_lang, text)
            prompts.append(prompt)
        
        max_tokens = 2048
        temperature = 1.0

        translated_texts = batch_generate(                
                prompts = prompts,
                max_length = max_tokens,
                temperature = temperature,
                top_k = 1,
                top_p = 0,
                alpha_presence = 0,
                alpha_frequency = 0,
                alpha_decay = 0.996,
                stop_tokens=[0],
        )
                
        translations_result = []
        for i, translation in enumerate(translated_texts):
            translations_result.append({
                "detected_source_lang": req.source_lang if req.source_lang != "auto" else "en",
                "text": translation.strip()
            })
        
        response = TranslateResponse(
            translations=translations_result,
        )
        
        # print(f"[RESPONSE] /translate/v1/batch-translate: {response.model_dump()}")

        return Response(
            status_code=200,
            description=response.model_dump_json(),
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        error_response = TranslateResponse(
            translations=[],
            detected_source_lang=req.source_lang,
        )
        return Response(
            status_code=500,
            description=error_response.model_dump_json(),
            headers={"Content-Type": "application/json"}
        )

@app.post("/FIM/v1/batch-FIM")
async def FIM_completions(request):
    body = json.loads(request.body)
    req = ChatRequest(**body)

    if args_cli.password and req.password != args_cli.password:
        return Response(
            status_code=401,
            description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
            headers={"Content-Type": "application/json"}
        )
    
    prompts = []
    prefix_list = req.prefix
    suffix_list = req.suffix
    for prefix, suffix in zip(prefix_list, suffix_list):
        prompt = f"✿prefix✿✿suffix✿{suffix}✿middle✿{prefix}"
        prompts.append(prompt)
    # print(f"[REQUEST] /FIM/v1/batch-FIM: {req.model_dump()}")
    # print(f"[REQUEST] /FIM/v1/batch-FIM: {prompts}")
    if len(prompts) == 1:
        # BachSize == 1 Super Fast Infer with CUDA graph
        if req.stream:
            return StreamingResponse(
                graph_infer_stream(inputs=prompts,
                                   stop_tokens=req.stop_tokens,
                                   max_generate_tokens=req.max_tokens,
                                   temperature=req.temperature,
                                   top_k=req.top_k,
                                   top_p=req.top_p,
                                   alpha_presence=req.alpha_presence,
                                   alpha_frequency=req.alpha_frequency,
                                   alpha_decay=req.alpha_decay,
                                   chunk_size=req.chunk_size),
                                   media_type="text/event-stream"
            )
        else:
            results = await graph_generate(inputs=prompts,
                                           stop_tokens=req.stop_tokens,
                                           max_generate_tokens=req.max_tokens,
                                           temperature=req.temperature,
                                           top_k=req.top_k,
                                           top_p=req.top_p,
                                           alpha_presence=req.alpha_presence,
                                           alpha_frequency=req.alpha_frequency,
                                           alpha_decay=req.alpha_decay)
            choices = []
            for i, text in enumerate(results):
                choices.append({
                    "index": i,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                })

            response = {
                "id": "rwkv7-batch-v3",
                "object": "chat.completion",
                "model": req.model,
                "choices": choices,
            }
        return Response(
            status_code=200,
            description=json.dumps(response, ensure_ascii=False),
            headers={"Content-Type": "application/json"}
        )
    else: 
        # BachSize > 1 Fast Batch Infer
        if req.stream:
            return StreamingResponse(
                batch_infer_stream(
                    prompts = prompts,
                    max_length = req.max_tokens,
                    temperature = req.temperature,
                    top_k = req.top_k,
                    top_p = req.top_p,
                    alpha_presence = req.alpha_presence,
                    alpha_frequency = req.alpha_frequency,
                    alpha_decay = req.alpha_decay,
                    stop_tokens=req.stop_tokens,
                    chunk_size = req.chunk_size,),
                    media_type="text/event-stream"
            )
        results = batch_generate(
            prompts = prompts,
            max_length = req.max_tokens,
            temperature = req.temperature,
            top_k = req.top_k,
            top_p = req.top_p,
            alpha_presence = req.alpha_presence,
            alpha_frequency = req.alpha_frequency,
            alpha_decay = req.alpha_decay,
            stop_tokens=req.stop_tokens,
        )
        choices = []
        for i, text in enumerate(results):
            choices.append({
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            })

        response = {
            "id": "rwkv7-batch",
            "object": "FIM.completion",
            "model": req.model,
            "choices": choices,
        }
        return Response(
            status_code=200,
            description=json.dumps(response, ensure_ascii=False),
            headers={"Content-Type": "application/json"}
        )
if __name__ == "__main__":
    app.start(host="0.0.0.0", port=args_cli.port)
