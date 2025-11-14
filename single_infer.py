import argparse
import torch
import types
import json
import gc, re
import asyncio
from robyn import Robyn, Response, StreamingResponse, ALLOW_CORS
from pydantic import BaseModel
from rwkv_batch.rwkv7 import RWKV_x070
from rwkv_batch.utils import TRIE_TOKENIZER, sampler_simple
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True, help="RWKV model path")
parser.add_argument("--port", type=int, default=8000)
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

static_graph = None
static_input_global = None
static_state_global = [None, None, None]
static_output_global = None
graph_initialized = False
graph_lock = Lock()

def initialize_cuda_graph():
    global static_graph, static_input_global, static_state_global, static_output_global, graph_initialized
    
    with graph_lock:
        if graph_initialized:
            return  
            
        try:
            state = model.generate_zero_state(0)
            
            example_prompt = "User: hello\n\nAssistant:"
            encoded_prompt = tokenizer.encode(example_prompt)
            
            out = model.forward(encoded_prompt, state)
            
            token = sampler_simple(out, noise=0).item()
            x = model.z['emb.weight'][token]
            
            static_input_global = torch.empty_like(x, device="cuda")
            static_state_global[0] = torch.empty_like(state[0], device="cuda")
            static_state_global[1] = torch.empty_like(state[1], device="cuda")
            static_state_global[2] = torch.empty_like(state[2], device="cuda")
            static_output_global = torch.empty_like(out, device="cuda")
            
            static_output_global = model.forward(static_input_global, static_state_global)
            
            static_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(static_graph):
                static_output_global = model.forward(static_input_global, static_state_global)
            
            static_input_global.copy_(x)
            static_state_global[0].copy_(state[0])
            static_state_global[1].copy_(state[1])
            static_state_global[2].copy_(state[2])
            
            graph_initialized = True
            print("[INFO] CUDA graph initialized successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to initialize CUDA graph: {e}")
            print("[INFO] Will use eager mode instead.")

app = Robyn(__file__)
@app.after_request()
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response
@app.options("/")
@app.options("/v4/chat/completions")
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
    contents: str = None
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

def generate_single(prompt, max_tokens=50, stop_tokens=[0, 261, 24281], temperature=1.0, noise=1.5):
    global static_graph, static_input_global, static_state_global, static_output_global, graph_initialized
    
    state = model.generate_zero_state(0)
    encoded_prompt = tokenizer.encode(prompt)
    out = model.forward(encoded_prompt, state)
    
    if isinstance(out, torch.Tensor):
        logits = out.clone()
        if temperature != 1.0:
            logits /= temperature
            
        token = sampler_simple(logits, noise=noise).item()
    else:
        token = out

    generated_tokens = [token]  # 第一个token加入结果
    finished = False
    
    x = model.z['emb.weight'][token]
    static_input_global.copy_(x)
    static_state_global[0].copy_(state[0])
    static_state_global[1].copy_(state[1])
    static_state_global[2].copy_(state[2])
    
    # 循环次数减1，保证总共生成max_tokens个token
    for step in range(max_tokens - 1):
        static_graph.replay()
        
        logits = static_output_global.clone()
        if temperature != 1.0:
            logits /= temperature
        token = sampler_simple(logits, noise=noise).item()
        
        if token in stop_tokens:
            finished = True
            break
            
        generated_tokens.append(token)
        
        x = model.z['emb.weight'][token]
        static_input_global.copy_(x)
    
    del state
    gc.collect()
    
    text = tokenizer.decode(generated_tokens, utf8_errors="ignore")
    return text

async def single_infer_stream(prompt, max_tokens=50, stop_tokens=[0, 261, 24281], 
                             temperature=1.0, noise=1.5, chunk_size=32):
    global static_graph, static_input_global, static_state_global, static_output_global, graph_initialized
    
    state = model.generate_zero_state(0)
    encoded_prompt = tokenizer.encode(prompt)
    out = model.forward(encoded_prompt, state)
    
    if isinstance(out, torch.Tensor):
        logits = out.clone()
        if temperature != 1.0:
            logits /= temperature
        token = sampler_simple(logits, noise=noise).item()
    else:
        token = out

    generated_tokens = [token]  # 第一个token加入结果
    token_buffer = [token]      # 第一个token也加入缓冲区，以便流式输出
    finished = False
    
    x = model.z['emb.weight'][token]
    static_input_global.copy_(x)
    static_state_global[0].copy_(state[0])
    static_state_global[1].copy_(state[1])
    static_state_global[2].copy_(state[2])
    
    try:
        # 循环次数减1，保证总共生成max_tokens个token
        for step in range(max_tokens - 1):
            static_graph.replay()
            
            logits = static_output_global.clone()
            if temperature != 1.0:
                logits /= temperature
            token = sampler_simple(logits, noise=noise).item()
            
            if token in stop_tokens:
                finished = True
                # 发送剩余的缓冲区内容
                if token_buffer:
                    text_chunk = tokenizer.decode(token_buffer, utf8_errors="ignore")
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": text_chunk}}]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    token_buffer.clear()
                break
                
            generated_tokens.append(token)
            token_buffer.append(token)
            
            # 缓冲区达到指定大小时发送数据
            if len(token_buffer) >= chunk_size:
                text_chunk = tokenizer.decode(token_buffer, utf8_errors="ignore")
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": text_chunk}}]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                token_buffer.clear()
            
            x = model.z['emb.weight'][token]
            static_input_global.copy_(x)
            
            await asyncio.sleep(0)
        
        # 发送最后剩余的内容
        if token_buffer and not finished:
            text_chunk = tokenizer.decode(token_buffer, utf8_errors="ignore")
            chunk = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": text_chunk}}]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
    finally:
        del state
        gc.collect()
    
    yield "data: [DONE]\n\n"

@app.post("/v4/chat/completions")
async def v4_chat_completions(request):
    try:
        body = json.loads(request.body)
        if "contents" not in body and "messages" in body:
            msgs = body.get("messages") or []
            user_texts = [m.get("content", "") for m in msgs if m.get("role") == "user"]
            if not user_texts and msgs:
                user_texts = [m.get("content", "") for m in msgs]
            body = {**body, "contents": "\n".join(user_texts) if user_texts else ""}

        req = ChatRequest(**body)
        prompt = req.contents
        if req.enable_think:
            prompt_formatted = f"User: {prompt}\n\nAssistant: <think"
        else:
            prompt_formatted = f"User: {prompt}\n\nAssistant:"

        if not prompt:
            return Response(
                status_code=400,
                description=json.dumps({"error": "Empty prompt"}),
                headers={"Content-Type": "application/json"}
            )

        if req.stream:
            return StreamingResponse(
                single_infer_stream(prompt=prompt_formatted,
                                   max_tokens=req.max_tokens,
                                   stop_tokens=req.stop_tokens,
                                   temperature=req.temperature,
                                   noise=req.noise,
                                   chunk_size=req.chunk_size),
                media_type="text/event-stream"
            )

        with model_lock:
            result = await asyncio.get_event_loop().run_in_executor(
                executor,
                generate_single,
                prompt_formatted,
                req.max_tokens,
                req.stop_tokens,
                req.temperature,
                req.noise,
            )
        
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": result},
            "finish_reason": "stop",
        }

        response = {
            "id": "rwkv7-single-v4",
            "object": "chat.completion",
            "model": req.model,
            "choices": [choice],
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
        print(f"[ERROR] /v4/chat/completions: {traceback.format_exc()}")
        return Response(
            status_code=500,
            description=json.dumps({"error": str(e)}),
            headers={"Content-Type": "application/json"}
        )
    
if __name__ == "__main__":
    # 尝试初始化CUDA graph，如果失败则回退到原始模式
    try:
        print("[INFO] Initializing CUDA graph...")
        initialize_cuda_graph()
    except Exception as e:
        print(f"[WARNING] Failed to initialize CUDA graph: {e}")
        print("[INFO] Will use eager mode instead.")
    
    app.start(host="0.0.0.0", port=args_cli.port)