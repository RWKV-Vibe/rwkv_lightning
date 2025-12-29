# rwkv_lightning 🕊️ ⚡
RWKV Batch infer backend Base on [Albatross](https://github.com/BlinkDL/Albatross) 🕊️ and [Robyn](https://github.com/sparckles/Robyn) 🦀 

## Install requirements
**For Nvidia CUDA**
```bash
pip install torch robyn pydantic ninja numpy flashinfer-python
```
**For AMD ROCm**

(The Flashinfer-python is not transfer to the AMD ROCm officially yet, please wait for the official compatibility. I actually tried to transfer it, but the Flash Infer library is a bit abstract and huge. I use the Pytorch base top_k top_p decode to implement the Flash infer CUDA GPU decode kernel)

**No problem! This could work too. It's not that it can't be used 🫣**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install torch robyn pydantic ninja numpy 
```

## Usage
```bash
python main_robyn.py --model-path <your model path> --port <your port number> --password rwkv7_7.2b
```
- if no password, you can do not add ```--password``` flag


## Test API quickly
```bash
bash ./test_curl.sh 
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


### 2. ```v1/chat/completions``` [Fastest Speed But Only support noise temperature decode] 
**--- Very stable 🚀 ---** 
- Streaming synchronous batch processing 
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "Chinese: RWKV-8 ROSA 机制：超越注意力机制的神经符号无限范围无损信息传播器，使大语言模型(LLM)能够发明自己的内心独白语言。迈向可扩展后神经方法的第一步，开启人工智能的新时代\n\nEnglish:",
      "Chinese: 他的脸上写满了痛苦和绝望，你会发现自己对这些故事产生了共鸣\n\nEnglish:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 1.0,
    "noise": 0,
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
      "Chinese: RWKV-8 ROSA 机制：超越注意力机制的神经符号无限范围无损信息传播器，使大语言模型(LLM)能够发明自己的内心独白语言。迈向可扩展后神经方法的第一步，开启人工智能的新时代\n\nEnglish:",
      "Chinese: 他的脸上写满了痛苦和绝望，你会发现自己对这些故事产生了共鸣\n\nEnglish:"
    ],
    "max_tokens": 1024,
    "stop_tokens": [0, 261, 24281],
    "temperature": 1.0,
    "noise": 0,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```


### 3. ```v2/chat/completions``` [Little slower than V1 But Only support all decode parameters]
**--- Very stable 🚀 ---** 
- Streaming synchronous continuous batching processing 
```bash
curl -X POST http://localhost:8000/v2/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "contents": [
      "Chinese: RWKV-8 ROSA 机制：超越注意力机制的神经符号无限范围无损信息传播器，使大语言模型(LLM)能够发明自己的内心独白语言。迈向可扩展后神经方法的第一步，开启人工智能的新时代\n\nEnglish:",
      "Chinese: 他的脸上写满了痛苦和绝望，你会发现自己对这些故事产生了共鸣\n\nEnglish:"
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
      "Chinese: RWKV-8 ROSA 机制：超越注意力机制的神经符号无限范围无损信息传播器，使大语言模型(LLM)能够发明自己的内心独白语言。迈向可扩展后神经方法的第一步，开启人工智能的新时代\n\nEnglish:",
      "Chinese: 他的脸上写满了痛苦和绝望，你会发现自己对这些故事产生了共鸣\n\nEnglish:"
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


### 4. ```v3/chat/completions``` [Support all decode parameters]

- Streaming asynchronous batch processing With CUDA Graph For Bsz=1
```bash
curl -X POST http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
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
    "enable_think": true,
    "password": "rwkv7_7.2b"
  }'
```
- Non-streaming asynchronous batch processing With CUDA Graph For Bsz=1
```bash
curl -X POST http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
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
    "stream": false,
    "enable_think": true,
    "password": "rwkv7_7.2b"
  }'
```