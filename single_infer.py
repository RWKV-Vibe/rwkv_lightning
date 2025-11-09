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

def generate_single(prompt, max_tokens=50, stop_tokens=[0, 261, 24281], temperature=1.0, 
                   top_k=1, top_p=0.3, alpha_presence=0.5, alpha_frequency=0.5, 
                   alpha_decay=0.996):
    state = model.generate_zero_state(0)
    encoded_prompt = tokenizer.encode(prompt)
    out = model.forward(encoded_prompt, state)
    
    # Initial sampling to get the first token
    if isinstance(out, torch.Tensor):
        # Process initial logits with temperature
        logits = out.clone()
        if temperature != 1.0:
            logits /= temperature
            
        # Sample first token using top-k and top-p
        if ROCm_Flag:
            token = torch_top_k_top_p(logits, top_k, top_p).item()
        else:
            import flashinfer # type: ignore
            token = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p).item()
    else:
        token = out

    generated_tokens = []
    finished = False
    
    occurrence = torch.zeros(args.vocab_size, dtype=torch.float32, device="cuda")
    no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
    
    for step in range(max_tokens):
        # Run model inference directly (no CUDA graph)
        x = model.z['emb.weight'][token]
        out = model.forward(x, state)
        
        # Process logits with penalties
        logits = out.clone()
        logits -= (alpha_presence + occurrence * alpha_frequency)
        
        # Apply temperature
        if temperature != 1.0:
            logits /= temperature
        
        # Sample token using top-k and top-p
        if ROCm_Flag:
            token = torch_top_k_top_p(logits, top_k, top_p).item()
        else:
            import flashinfer # type: ignore
            token = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p).item()
        
        if token in stop_tokens:
            finished = True
            break
            
        generated_tokens.append(token)
        
        www = 0.0 if token in no_penalty_token_ids else 1.0
        occurrence[token] += www
        
        occurrence *= alpha_decay
    
    del state, occurrence
    gc.collect()
    
    text = tokenizer.decode(generated_tokens, utf8_errors="ignore")
    return text

async def single_infer_stream(prompt, max_tokens=50, stop_tokens=[0, 261, 24281], 
                             temperature=1.0, top_k=1, top_p=0.3,
                             alpha_presence=0.5, alpha_frequency=0.5, alpha_decay=0.996,
                             chunk_size=32):
    state = model.generate_zero_state(0)
    
    encoded_prompt = tokenizer.encode(prompt)
    
    out = model.forward(encoded_prompt, state)
    
    # Initial sampling to get the first token
    if isinstance(out, torch.Tensor):
        # Process initial logits with temperature
        logits = out.clone()
        if temperature != 1.0:
            logits /= temperature
            
        # Sample first token using top-k and top-p
        if ROCm_Flag:
            token = torch_top_k_top_p(logits, top_k, top_p).item()
        else:
            import flashinfer # type: ignore
            token = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p).item()
    else:
        token = out

    generated_tokens = []
    token_buffer = []
    finished = False
    
    occurrence = torch.zeros(args.vocab_size, dtype=torch.float32, device="cuda")
    no_penalty_token_ids = set([33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58])
    
    try:
        for step in range(max_tokens):
            # Run model inference directly (no CUDA graph)
            x = model.z['emb.weight'][token]
            out = model.forward(x, state)
            
            # Process logits with penalties
            logits = out.clone()
            logits -= (alpha_presence + occurrence * alpha_frequency)
            
            # Apply temperature
            if temperature != 1.0:
                logits /= temperature

            # Sample token using top-k and top-p
            if ROCm_Flag:
                token = torch_top_k_top_p(logits, top_k, top_p).item()
            else:
                import flashinfer # type: ignore
                token = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p).item()
            
            if token in stop_tokens:
                finished = True
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
            
            if len(token_buffer) >= chunk_size:
                text_chunk = tokenizer.decode(token_buffer, utf8_errors="ignore")
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": text_chunk}}]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                token_buffer.clear()
            
            www = 0.0 if token in no_penalty_token_ids else 1.0
            occurrence[token] += www
            
            occurrence *= alpha_decay
            
            await asyncio.sleep(0)
        
        if token_buffer and not finished:
            text_chunk = tokenizer.decode(token_buffer, utf8_errors="ignore")
            chunk = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": text_chunk}}]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            
    finally:
        del state, occurrence
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
                                   top_k=req.top_k,
                                   top_p=req.top_p,
                                   alpha_presence=req.alpha_presence,
                                   alpha_frequency=req.alpha_frequency,
                                   alpha_decay=req.alpha_decay,
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
                req.top_k,
                req.top_p,
                req.alpha_presence,
                req.alpha_frequency,
                req.alpha_decay
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
    app.start(host="0.0.0.0", port=args_cli.port)