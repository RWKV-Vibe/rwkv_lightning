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


### 1. Batch synchronous Translate 

<details>
<summary>curl examples</summary>

**Compatible with immersive translation custom API**
**--- Very stable 🚀 ---** 
```bash
curl -X POST http://localhost:8000/translate/v1/batch-translate \
         -H "Content-Type: application/json" \
         -d '{
           "source_lang": "en",
           "target_lang": "zh-CN",
           "text_list": ["Hello world!", "Good morning"]
         }'
```
```bash
curl -X POST http://localhost:8000/translate/v1/batch-translate \
         -H "Content-Type: application/json" \
         -d '{
           "source_lang": "zh-CN",
           "target_lang": "en",
           "text_list": ["你好世界", "早上好"]
         }'
```
</details>


### 2. ```v1/chat/completions```  [Support all decode parameters]

<details>
<summary>curl examples</summary>

**--- Very stable 🚀 ---** 
- Streaming synchronous batch processing 
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```
- Non-streaming synchronous batch processing
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

</details>


### 3. ```v2/chat/completions``` [Support all decode parameters]

<details>
<summary>curl examples</summary>

**--- Very stable 🚀 ---** 
- Streaming synchronous continuous batching processing 
```bash
curl -X POST http://localhost:8000/v2/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "contents": [
      "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 0.3,
    "pad_zero": true,
    "alpha_presence": 0.8,
    "alpha_frequency": 0.8,
    "alpha_decay": 0.996,
    "chunk_size": 128,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```
- Non-streaming synchronous continuous batching processing
```bash
curl -X POST http://localhost:8000/v2/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 0.3,
    "pad_zero": true,
    "alpha_presence": 0.8,
    "alpha_frequency": 0.8,
    "alpha_decay": 0.996,
    "chunk_size": 32,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

</details>


### 4. ```state/chat/completions``` [Support state cache manager] 😜

#### Have 3 Levels Cache design 🤓
- **L1 cache(VRAM) 16**
- **L2 cache(RAM) 32**
- **L3 cache(Sqlite3 database)**
#### The all cached state will be stored in the database when shout down the server 😋
- could modify the cache size in ```./state_pool.py``` in line 14-16

***Need to add a unique "session_id": "XXX" in the request body as a unique identifier for each session***👆

**ONLY support for bsz = 1 one session** 🤫

<details>
<summary>curl examples</summary>

- Streaming asynchronous batch processing With CUDA Graph For Bsz=1
```bash
curl -X POST http://localhost:8000/state/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "contents": [
      "User: What should we eat for dinner? Any brief suggestions?\n\nAssistant: <think>\n</think>\n"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": true,
    "chunk_size": 128,
    "password": "rwkv7_7.2b",
    "session_id": "session_one"
  }'
```
- Non-streaming asynchronous batch processing With CUDA Graph For Bsz=1
```bash
curl -X POST http://localhost:8000/state/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
    "contents": [
      "User: What should we eat for dinner? Any brief suggestions?\n\nAssistant: <think>\n</think>\n"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": false,
    "password": "rwkv7_7.2b",
    "session_id": "session_one"
  }'
```

</details>


### 5. **State Management API** [Support state cache manager] 😜 

#### Use ```state/status```  Interface to check the state pool status of a session

<details>
<summary>curl examples</summary>

```bash
curl -X POST http://localhost:8000/state/status \
  -H "Content-Type: application/json" \
  -d '{
    "password": "rwkv7_7.2b"
  }'
```

</details>

#### Use ```state/delete```  Interface to delete the state of a session

<details>
<summary>curl examples</summary>


```bash
curl -X POST http://localhost:8000/state/delete \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id_to_delete",
    "password": "rwkv7_7.2b"
  }'
```

</details>

### 6. ```/openai/v1/chat/completions``` [Open AI format support]

<details>
<summary>curl examples</summary>

- Streaming asynchronous Open AI API
```bash
curl -X POST 'http://localhost:8000/openai/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer your-password-if-set' \
  --data '{
    "model": "rwkv7",
    "messages": [
      {"role": "user", "content": "please tell me about the history of artificial intelligence"}
    ],
    "top_p": 0.6,
    "max_tokens": 2048,
    "temperature": 0.8,
    "stream": true
  }'
```
- Non-streaming asynchronous Open AI API
```bash
curl -X POST 'http://localhost:8000/openai/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer your-password-if-set' \
  --data '{
    "model": "rwkv7",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "please tell me about the history of artificial intelligence"}
    ],
    "top_p": 0.6,
    "max_tokens": 2048,
    "temperature": 1,
    "stream": false
  }'
```

- Stateful incremental Open AI API with `session_id`
```bash
curl -X POST 'http://localhost:8000/openai/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer your-password-if-set' \
  --data '{
    "model": "rwkv7",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Please continue from our last turn and give me 3 short ideas."}
    ],
    "top_p": 0.6,
    "max_tokens": 2048,
    "temperature": 1,
    "stream": false
  }'
```

- Related focused test scripts
```bash
python test/test_openai_adapter.py
python test/test_openai_routes.py
```
</details>

### 7. ```/big_batch/completions```  [Only Support noise & temperature decode parameters]

<details>
<summary>curl examples</summary>

**The Fastest Batch Processing API 🚀** 
- Streaming synchronous batch processing 
```bash
curl -X POST 'http://localhost:8000/big_batch/completions' \
  --header 'Content-Type: application/json' \
  --data '{
    "contents": [
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 1.0,
    "chunk_size": 8,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```
</details>

### FIM ( For RWKV7_G1c series model )

<details>
<summary>curl examples</summary>

**Batch stream inference using [FIM/v1/batch-FIM interface]**

```bash
curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "The rain had stopped, but the street still glistened like a river of broken glass.",
      "She wasn’t sure why she’d come back.",
      "A cat darted from the alley,"
    ],
    "suffix": [
      "though everyone knew Mr. Ellis hadn’t opened that door in three years.",
      "sounding almost like her name.",
      "And then, from inside, a single lamp clicked on."
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```

**Batch inference using [FIM/v1/batch-FIM interface]**

```bash
curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "The rain had stopped, but the street still glistened like a river of broken glass.",
      "She wasn’t sure why she’d come back.",
      "A cat darted from the alley,"
    ],
    "suffix": [
      "though everyone knew Mr. Ellis hadn’t opened that door in three years.",
      "sounding almost like her name.",
      "And then, from inside, a single lamp clicked on."
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```
