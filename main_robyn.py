import argparse
import torch
import types
import json
import gc
import asyncio
from robyn import Robyn, Response, StreamingResponse
from pydantic import BaseModel
from rwkv_batch.rwkv7 import RWKV_x070
from rwkv_batch.utils import TRIE_TOKENIZER, sampler_simple_batch
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True, help="RWKV model path")
parser.add_argument("--port", type=int, default=8000)
args_cli = parser.parse_args()
ROCm_Flag = torch.version.hip is not None

print(f"\n[INFO] Loading RWKV-7 model from {args_cli.model_path}\n")

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
args.MODEL_NAME = args_cli.model_path

model = RWKV_x070(args)
tokenizer = TRIE_TOKENIZER("rwkv_batch/rwkv_vocab_v20230424.txt")

print(f"[INFO] Model loaded successfully.\n")


app = Robyn(__file__)

class ChatRequest(BaseModel):
    model: str = "rwkv7"
    contents: list[str]               # 输入句子列表
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


def batch_generate(prompts, max_length=512, noise=1.5 ,temperature=1.0, stop_tokens=[0, 261, 24281]):
    B = len(prompts)
    state = model.generate_zero_state(B)
    encoded_prompts = [tokenizer.encode(p) for p in prompts]
    out = model.forward_batch(encoded_prompts, state)

    finished = [False] * B
    generated_tokens = [[] for _ in range(B)]

    for step in range(max_length):
        new_tokens = sampler_simple_batch(out, noise=noise, temp=temperature).tolist()
        out = model.forward_batch(new_tokens, state)

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

async def batch_infer_stream(prompts, max_length=512, noise=1.5, temperature=1.0, stop_tokens=[0, 261, 24281]):
    B = len(prompts)
    state = model.generate_zero_state(B)
    encoded_prompts = [tokenizer.encode(p) for p in prompts]
    out = model.forward_batch(encoded_prompts, state)

    finished = [False] * B
    generated_tokens = [[] for _ in range(B)]
    token_buffers = [[] for _ in range(B)] 

    try:
        while not all(finished) and max_length > 0:
            new_tokens = sampler_simple_batch(out, noise=noise, temp=temperature).tolist()
            out = model.forward_batch(new_tokens, state)
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
                
                if len(token_buffers[i]) >= 32:
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

async def continuous_batching_stream(
    model,
    tokenizer,
    inputs,  # prompts 列表
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

    STOP_TOKENS = stop_tokens
    MAX_GENERATE_TOKENS = max_generate_tokens
    BATCH_SIZE = batch_size
    PAD_ZERO = pad_zero
    CHUNK_SIZE = chunk_size
    
    device = model.z["head.weight"].device

    alpha_presence = torch.tensor(alpha_presence, dtype=torch.float32, device=device)

    if temperature == 0:  # 贪婪采样
        temperature = 1.0
        top_k = 1

    total_inputs = len(inputs)
    
    # 准备输入队列
    encoded_inputs = []
    for prompt in inputs:
        input_token = tokenizer.encode(prompt)
        if PAD_ZERO:
            input_token = [0] + input_token
        encoded_inputs.append((prompt, input_token))
    inputs = deque(encoded_inputs)

    # 初始化模型状态和任务池
    states = model.generate_zero_state(BATCH_SIZE)
    task_pool = []
    token_buffers = {} # 用于流式分块 {prompt_idx: [tokens]}
    
    prompt_idx = 0
    for i in range(BATCH_SIZE):
        prompt, input_token = inputs.popleft()
        task_pool.append(
            {
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "input_token": input_token, # 待处理的 token 列表
                "state_pos": i,             # 在 states 张量中的索引
                "generated_tokens": [],     # 已生成的 token 列表
                "new_token": None,          # 下一个要处理的 token
            }
        )
        token_buffers[prompt_idx] = [] # 为新任务初始化 token 缓冲区
        prompt_idx += 1

    # 初始化惩罚张量
    occurrence = torch.zeros((BATCH_SIZE, args.vocab_size), dtype=torch.float32, device=device)
    # ' \t0123456789'
    no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
    alpha_presence_vector = torch.zeros((BATCH_SIZE, args.vocab_size), dtype=torch.float32, device=device)

    try:
        # --- 2. 主循环 ---
        while True:
            accomplished_task_indices = [] # 索引在 task_pool 中的位置
            state_slots_to_remove = set()    # 索引在 states 张量中的位置
            contents_to_send = {}            # {prompt_idx: "text_chunk"}
            
            # 2.a. 检查任务状态 (完成, 流式, 或继续)
            for task_idx, task in enumerate(task_pool):
                
                # 只有在 prompt 处理完毕后 (input_token 为空) 才开始检查解码
                if len(task["input_token"]) == 0:
                    
                    if task["new_token"] is None: # 刚处理完 prompt，等待第一个解码 token
                        continue
                        
                    new_token = task["new_token"]
                    prompt_id = task["prompt_idx"]

                    token_in_stop = new_token in STOP_TOKENS
                    length_exceed = len(task["generated_tokens"]) >= MAX_GENERATE_TOKENS
                    
                    is_finished = token_in_stop or length_exceed

                    # 如果任务未完成，添加 token 并检查分块
                    if not is_finished:
                        task["generated_tokens"].append(new_token)
                        token_buffers[prompt_id].append(new_token)
                        
                        # 检查是否达到分块大小
                        if len(token_buffers[prompt_id]) >= CHUNK_SIZE:
                            text_chunk = tokenizer.decode(token_buffers[prompt_id], utf8_errors="ignore")
                            contents_to_send[prompt_id] = text_chunk
                            token_buffers[prompt_id].clear()
                    
                    # 如果任务已完成
                    if is_finished:
                        # 发送缓冲区中剩余的 token
                        if token_buffers[prompt_id]:
                            text_chunk = tokenizer.decode(token_buffers[prompt_id], utf8_errors="ignore")
                            # 如果已有块，则附加；否则新建
                            contents_to_send[prompt_id] = contents_to_send.get(prompt_id, "") + text_chunk
                            token_buffers[prompt_id].clear()
                        
                        del token_buffers[prompt_id] # 清理已完成任务的缓冲区
                        
                        # 替换任务
                        if len(inputs) > 0:  # 动态添加新任务
                            prompt, input_token = inputs.popleft()
                            new_prompt_idx = prompt_idx
                            task_pool[task_idx] = {
                                "prompt_idx": new_prompt_idx,
                                "prompt": prompt,
                                "input_token": input_token,
                                "state_pos": task["state_pos"],
                                "generated_tokens": [],
                                "new_token": None,
                            }
                            token_buffers[new_prompt_idx] = [] # 为新任务初始化缓冲区
                            prompt_idx += 1
                            
                            # 重置此状态槽的状态和惩罚
                            state_pos = task["state_pos"]
                            states[0][:, :, state_pos, :] = 0
                            states[1][:, state_pos, :, :] = 0
                            occurrence[state_pos, :] = 0
                            alpha_presence_vector[state_pos, :] = 0
                        
                        else:  # 没有新任务，标记此槽以便移除
                            accomplished_task_indices.append(task_idx)
                            state_slots_to_remove.add(task["state_pos"])
                    
                    else:  # 任务未完成，将新 token 添加回 input_token 以便下一轮
                        task["input_token"].append(new_token)
                        # 更新惩罚
                        www = 0.0 if new_token in no_penalty_token_ids else 1.0
                        occurrence[task["state_pos"], new_token] += www
                        alpha_presence_vector[task["state_pos"], new_token] = alpha_presence
            
            # 2.b. Yield 流式数据块
            if contents_to_send:
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": pid, "delta": {"content": content}}
                        for pid, content in contents_to_send.items() if content
                    ]
                }
                if chunk["choices"]:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0) # 释放控制权

            # 2.c. 移除已完成的状态槽 (当没有新任务替换时)
            if accomplished_task_indices:
                sorted_slots_to_remove = sorted(list(state_slots_to_remove), reverse=True)

                # 通过拼接 (cat) 来物理上压缩状态张量
                for slot in sorted_slots_to_remove:
                    part1_s0 = states[0][:, :, :slot, :]
                    part2_s0 = states[0][:, :, slot + 1 :, :]
                    states[0] = torch.cat([part1_s0, part2_s0], dim=2)

                    part1_s1 = states[1][:, :slot, :, :]
                    part2_s1 = states[1][:, slot + 1 :, :, :]
                    states[1] = torch.cat([part1_s1, part2_s1], dim=1)

                    occ_part1 = occurrence[:slot, :]
                    occ_part2 = occurrence[slot + 1 :, :]
                    occurrence = torch.cat([occ_part1, occ_part2], dim=0)

                    alpha_presence_part1 = alpha_presence_vector[:slot, :]
                    alpha_presence_part2 = alpha_presence_vector[slot + 1 :, :]
                    alpha_presence_vector = torch.cat([alpha_presence_part1, alpha_presence_part2], dim=0)

                # 从 task_pool 中移除任务
                for task_idx in sorted(accomplished_task_indices, reverse=True):
                    del task_pool[task_idx]

                # 重新映射剩余任务的 state_pos，使其连续
                remaining_slots = sorted([t["state_pos"] for t in task_pool])
                pos_map = {old_pos: new_pos for new_pos, old_pos in enumerate(remaining_slots)}
                for task in task_pool:
                    task["state_pos"] = pos_map[task["state_pos"]]

            # 2.d. 检查是否所有任务都已完成
            if len(task_pool) == 0:
                break
            
            # 2.e. 准备下一批 token
            # 此时 task_pool 的大小可能小于 BATCH_SIZE
            current_batch_size = len(task_pool)
            next_tokens = [None] * current_batch_size
            for task in task_pool:
                # 消耗一个 token (无论是来自 prompt 还是上一步生成的)
                next_tokens[task["state_pos"]] = [task["input_token"].pop(0)]

            # 2.f. 执行模型前向传播
            out = model.forward_batch(next_tokens, states)

            # 2.g. 应用惩罚和采样
            occurrence *= alpha_decay
            out -= alpha_presence_vector + occurrence * alpha_frequency

            if temperature != 1.0:
                out /= temperature
            
            if 'ROCm_Flag' in globals() and ROCm_Flag: # 检查 ROCm_Flag 是否已定义
                new_tokens = torch_top_k_top_p(out, top_k, top_p)
            else:
                import flashinfer # type: ignore
                new_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(out, top_k, top_p)
            
            new_tokens = new_tokens.tolist()

            # 2.h. 将新 token 分配回任务
            for task in task_pool:
                state_pos = task["state_pos"]
                tok = new_tokens[state_pos]
                task["new_token"] = tok

    finally:
        # --- 3. 清理 ---
        del states
        del occurrence
        del alpha_presence_vector
        gc.collect()

    # 发送结束信号
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request):
    body = json.loads(request.body)
    req = ChatRequest(**body)
    prompts = req.contents

    if req.stream:
        return StreamingResponse(
            batch_infer_stream(prompts, req.max_tokens, req.noise, req.temperature, req.stop_tokens),
            media_type="text/event-stream"
        )

    results = batch_generate(prompts, req.max_tokens, req.noise, req.temperature, req.stop_tokens)
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
    body = json.loads(request.body)
    req = ChatRequest(**body)
    prompts = req.contents

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

    results = batch_generate(prompts, req.max_tokens, req.noise, req.temperature, req.stop_tokens)
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

    print(f"[REQUEST] /translate/v1/batch-translate: {req.model_dump()}")

    try:
        processed_texts = req.text_list
        
        prompts = []
        for text in processed_texts:
            prompt = create_translation_prompt(req.source_lang, req.target_lang, text)
            prompts.append(prompt)
        
        max_tokens = 2048
        temperature = 1.0

        translated_texts = batch_generate(prompts, max_length=max_tokens, noise=0, temperature=temperature)
                
        translations_result = []
        for i, translation in enumerate(translated_texts):
            translations_result.append({
                "detected_source_lang": req.source_lang if req.source_lang != "auto" else "en",
                "text": translation.strip()
            })
        
        response = TranslateResponse(
            translations=translations_result,
        )
        
        print(f"[RESPONSE] /translate/v1/batch-translate: {response.model_dump()}")

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

if __name__ == "__main__":
    app.start(host="0.0.0.0", port=args_cli.port)
