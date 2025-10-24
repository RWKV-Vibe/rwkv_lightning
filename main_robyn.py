import argparse
import types
import json
import gc
import asyncio
from robyn import Robyn, Response, StreamingResponse
from pydantic import BaseModel
from rwkv_batch.rwkv7 import RWKV_x070
from rwkv_batch.utils import TRIE_TOKENIZER, sampler_simple_batch

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True, help="RWKV model path")
parser.add_argument("--port", type=int, default=8000)
args_cli = parser.parse_args()

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
    noise: float = 1.5
    stream: bool = False
    pad_zero:bool = True
    alpha_presence: float = 0.5
    alpha_frequency: float = 0.5
    alpha_decay: float = 0.996
    enable_think: bool = False


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
