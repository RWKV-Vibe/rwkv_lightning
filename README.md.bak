# rwkv_lightning 🕊️ ⚡
RWKV Batch infer backend Base on [Albatross](https://github.com/BlinkDL/Albatross) 🕊️ and [Robyn](https://github.com/sparckles/Robyn) 🦀 
- Thanks to [Rapid-Sampling](https://github.com/Triang-jyed-driung/Rapid-Sampling) Kernel From [Triang-jyed-driung](https://github.com/Triang-jyed-driung), it also have native HIP kerel compatible with ROCm😎
## Install requirements
**For Nvidia CUDA**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install robyn pydantic ninja numpy 
[optional] pip install flashinfer-python
```
**For AMD ROCm**

(The Flashinfer-python is not transfer to the AMD ROCm officially yet, please wait for the official compatibility. I actually tried to transfer it, but the Flash Infer library is a bit abstract and huge. I use the Pytorch base top_k top_p decode to implement the Flash infer CUDA GPU decode kernel)

**No problem! This could work too. It's not that it can't be used 🫣**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install robyn pydantic ninja numpy 
```

## Usage
```bash
python app.py --model-path <your model path> --port <your port number> --password rwkv7_7.2b
```
- if no password, you can do not add ```--password``` flag


## Test API quickly
```bash
bash ./test/test_curl.sh
```

## Tips
If you want to the max performance optimization, you can use the ```torch.compile(mode='max-autotune-no-cudagraphs')```  

you can modify the code in the ```rwkv_batch/rwkv7.py``` line 30, 31
```python
MyFunction = torch.compile(mode='max-autotune-no-cudagraphs')
MyStatic = torch.compile(mode='max-autotune-no-cudagraphs')
```
**But it will be slow in first inference request, Because it needs to compile the Triton kernel firstly.**

## API Docs 


<div style="overflow-x: auto; margin: 20px 0;">
<table style="width: 100%; min-width: 850px; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 24px rgba(45, 106, 79, 0.06); background-color: #ffffff; text-align: left; line-height: 1.6; font-size: 14px;">
<thead style="background: linear-gradient(90deg, #f0fdf4 0%, #f4fbfa 100%); color: #2d6a4f; border-bottom: 2px solid #d1fae5;">
<tr>
<th style="padding: 18px 24px; font-weight: 600; width: 25%;">🚀 API Endpoint</th>
<th style="padding: 18px 24px; font-weight: 600; width: 18%;">💡 Core Function</th>
<th style="padding: 18px 24px; font-weight: 600; width: 23%;">✨ Features & Description</th>
<th style="padding: 18px 24px; font-weight: 600; width: 40%;">📡 Request Method</th>
</tr>
</thead>
<tbody style="color: #4b5563;">
<tr style="border-bottom: 1px solid #f3f4f6; transition: background-color 0.2s;">
<td style="padding: 16px 24px; vertical-align: top;"><code style="background-color: #f8fafc; color: #0f172a; padding: 4px 8px; border-radius: 6px; font-size: 13px; border: 1px solid #e2e8f0;">/translate/v1/batch-translate</code></td>
<td style="padding: 16px 24px; font-weight: 500; vertical-align: top;">Batch Sync Translation</td>
<td style="padding: 16px 24px; font-size: 13.5px; vertical-align: top;">Perfectly compatible with Immersive Translate custom API, extremely stable.</td>
<td style="padding: 16px 24px; vertical-align: top;">
<details style="cursor: pointer; outline: none;">
<summary style="display: inline-block; outline: none;">
<span style="background-color: #eff6ff; color: #1d4ed8; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid #bfdbfe;">Non-stream ▾</span>
</summary>
<div style="margin-top: 10px; padding: 12px; background-color: #1e293b; color: #e2e8f0; border-radius: 8px; font-size: 12px; font-family: monospace; white-space: pre; overflow-x: auto;">curl -X POST http://localhost:8000/translate/v1/batch-translate
-H "Content-Type: application/json"
-d '{
"source_lang": "en",
"target_lang": "zh-CN",
"text_list": ["Hello world!", "Good morning"]
}'</div>
</details>
</td>
</tr>
<tr style="border-bottom: 1px solid #f3f4f6; background-color: #fafdfc;">
<td style="padding: 16px 24px; vertical-align: top;"><code style="background-color: #f8fafc; color: #0f172a; padding: 4px 8px; border-radius: 6px; font-size: 13px; border: 1px solid #e2e8f0;">/v1/chat/completions</code></td>
<td style="padding: 16px 24px; font-weight: 500; vertical-align: top;">Standard Chat Completion</td>
<td style="padding: 16px 24px; font-size: 13.5px; vertical-align: top;">Batch synchronous processing, supports <strong>all decode parameters</strong>, extremely stable.</td>
<td style="padding: 16px 24px; vertical-align: top;">
<details style="cursor: pointer; outline: none; margin-bottom: 8px;">
<summary style="display: inline-block; outline: none;">
<span style="background-color: #ecfdf5; color: #047857; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid #a7f3d0;">Stream ▾</span>
</summary>
<div style="margin-top: 10px; padding: 12px; background-color: #1e293b; color: #e2e8f0; border-radius: 8px; font-size: 12px; font-family: monospace; white-space: pre; overflow-x: auto;">curl -X POST http://localhost:8000/v1/chat/completions
-H "Content-Type: application/json"
-d '{
"contents": ["Hello", "World"],
"max_tokens": 1024,
"stream": true,
"password": "rwkv7_7.2b"
}'</div>
</details>
</td>
</tr>
<tr style="border-bottom: 1px solid #f3f4f6;">
<td style="padding: 16px 24px; vertical-align: top;"><code style="background-color: #f8fafc; color: #0f172a; padding: 4px 8px; border-radius: 6px; font-size: 13px; border: 1px solid #e2e8f0;">/v2/chat/completions</code></td>
<td style="padding: 16px 24px; font-weight: 500; vertical-align: top;">Continuous Batching Chat</td>
<td style="padding: 16px 24px; font-size: 13.5px; vertical-align: top;">Synchronous <strong>Continuous Batching</strong> processing, supports all decode parameters, extremely stable.</td>
<td style="padding: 16px 24px; vertical-align: top;">
<details style="cursor: pointer; outline: none;">
<summary style="display: inline-block; outline: none;">
<span style="background-color: #ecfdf5; color: #047857; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid #a7f3d0;">Stream ▾</span>
</summary>
<div style="margin-top: 10px; padding: 12px; background-color: #1e293b; color: #e2e8f0; border-radius: 8px; font-size: 12px; font-family: monospace; white-space: pre; overflow-x: auto;">curl -X POST http://localhost:8000/v2/chat/completions
-H "Content-Type: application/json"
-N
-d '{
"contents": ["Hi there!", "Tell me a joke."],
"max_tokens": 1024,
"chunk_size": 128,
"stream": true,
"password": "rwkv7_7.2b"
}'</div>
</details>
</td>
</tr>
<tr style="border-bottom: 1px solid #f3f4f6; background-color: #fafdfc;">
<td style="padding: 16px 24px; vertical-align: top;"><code style="background-color: #f8fafc; color: #0f172a; padding: 4px 8px; border-radius: 6px; font-size: 13px; border: 1px solid #e2e8f0;">/state/chat/completions</code></td>
<td style="padding: 16px 24px; font-weight: 500; vertical-align: top;">Stateful Chat Completion 💾</td>
<td style="padding: 16px 24px; font-size: 13.5px; vertical-align: top;">Supports state cache manager. Features <strong>L1(VRAM), L2(RAM), L3(Sqlite3)</strong> cache design. Only supports single session (Bsz=1), requires <code>session_id</code>.</td>
<td style="padding: 16px 24px; vertical-align: top;">
<details style="cursor: pointer; outline: none;">
<summary style="display: inline-block; outline: none;">
<span style="background-color: #ecfdf5; color: #047857; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid #a7f3d0;">Stream ▾</span>
</summary>
<div style="margin-top: 10px; padding: 12px; background-color: #1e293b; color: #e2e8f0; border-radius: 8px; font-size: 12px; font-family: monospace; white-space: pre; overflow-x: auto;">curl -X POST http://localhost:8000/state/chat/completions
-H "Content-Type: application/json"
-N
-d '{
"contents": ["User: What should we eat for dinner?\n\nAssistant:"],
"stream": true,
"password": "rwkv7_7.2b",
"session_id": "session_one"
}'</div>
</details>
</td>
</tr>
<tr style="border-bottom: 1px solid #f3f4f6;">
<td style="padding: 16px 24px; vertical-align: top;">
<code style="background-color: #f8fafc; color: #0f172a; padding: 4px 8px; border-radius: 6px; font-size: 13px; border: 1px solid #e2e8f0; display: inline-block; margin-bottom: 6px;">/state/status</code>


<code style="background-color: #f8fafc; color: #0f172a; padding: 4px 8px; border-radius: 6px; font-size: 13px; border: 1px solid #e2e8f0; display: inline-block;">/state/delete</code>
</td>
<td style="padding: 16px 24px; font-weight: 500; vertical-align: top;">Cache State Management</td>
<td style="padding: 16px 24px; font-size: 13.5px; vertical-align: top;">Used to check the status of the state pool for a specific session (<code>session_id</code>), or delete its cached state.</td>
<td style="padding: 16px 24px; vertical-align: top;">
<details style="cursor: pointer; outline: none; margin-bottom: 8px;">
<summary style="display: inline-block; outline: none;">
<span style="background-color: #eff6ff; color: #1d4ed8; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid #bfdbfe;">Check Status ▾</span>
</summary>
<div style="margin-top: 10px; padding: 12px; background-color: #1e293b; color: #e2e8f0; border-radius: 8px; font-size: 12px; font-family: monospace; white-space: pre; overflow-x: auto;">curl -X POST http://localhost:8000/state/status
-H "Content-Type: application/json"
-d '{"password": "rwkv7_7.2b"}'</div>
</details>
<details style="cursor: pointer; outline: none;">
<summary style="display: inline-block; outline: none;">
<span style="background-color: #fef2f2; color: #b91c1c; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid #fecaca;">Delete State ▾</span>
</summary>
<div style="margin-top: 10px; padding: 12px; background-color: #1e293b; color: #e2e8f0; border-radius: 8px; font-size: 12px; font-family: monospace; white-space: pre; overflow-x: auto;">curl -X POST http://localhost:8000/state/delete
-H "Content-Type: application/json"
-d '{
"session_id": "your_session_id_to_delete",
"password": "rwkv7_7.2b"
}'</div>
</details>
</td>
</tr>
<tr style="border-bottom: 1px solid #f3f4f6; background-color: #fafdfc;">
<td style="padding: 16px 24px; vertical-align: top;"><code style="background-color: #f8fafc; color: #0f172a; padding: 4px 8px; border-radius: 6px; font-size: 13px; border: 1px solid #e2e8f0;">/openai/v1/chat/completions</code></td>
<td style="padding: 16px 24px; font-weight: 500; vertical-align: top;">OpenAI Format Support</td>
<td style="padding: 16px 24px; font-size: 13.5px; vertical-align: top;">Standard OpenAI format interface adaptation. Supports appending <code>session_id</code> for stateful incremental processing.</td>
<td style="padding: 16px 24px; vertical-align: top;">
<details style="cursor: pointer; outline: none;">
<summary style="display: inline-block; outline: none;">
<span style="background-color: #ecfdf5; color: #047857; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid #a7f3d0;">Stream ▾</span>
</summary>
<div style="margin-top: 10px; padding: 12px; background-color: #1e293b; color: #e2e8f0; border-radius: 8px; font-size: 12px; font-family: monospace; white-space: pre; overflow-x: auto;">curl -X POST http://localhost:8000/openai/v1/chat/completions
-H "Content-Type: application/json"
-d '{
"model": "rwkv7",
"messages": [
{"role": "user", "content": "Tell me about AI."}
],
"stream": true
}'</div>
</details>
</td>
</tr>
<tr style="border-bottom: 1px solid #f3f4f6;">
<td style="padding: 16px 24px; vertical-align: top;"><code style="background-color: #f8fafc; color: #0f172a; padding: 4px 8px; border-radius: 6px; font-size: 13px; border: 1px solid #e2e8f0;">/big_batch/completions</code></td>
<td style="padding: 16px 24px; font-weight: 500; vertical-align: top;">Extreme Batching ⚡</td>
<td style="padding: 16px 24px; font-size: 13.5px; vertical-align: top;"><strong>The fastest batch processing API.</strong> As a performance trade-off, only supports <code>noise</code> and <code>temperature</code> decode parameters.</td>
<td style="padding: 16px 24px; vertical-align: top;">
<details style="cursor: pointer; outline: none;">
<summary style="display: inline-block; outline: none;">
<span style="background-color: #ecfdf5; color: #047857; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid #a7f3d0;">Stream ▾</span>
</summary>
<div style="margin-top: 10px; padding: 12px; background-color: #1e293b; color: #e2e8f0; border-radius: 8px; font-size: 12px; font-family: monospace; white-space: pre; overflow-x: auto;">curl -X POST http://localhost:8000/big_batch/completions
-H "Content-Type: application/json"
-d '{
"contents": ["Text 1...", "Text 2..."],
"temperature": 1.0,
"stream": true,
"password": "rwkv7_7.2b"
}'</div>
</details>
</tbody>
</table>
</div>