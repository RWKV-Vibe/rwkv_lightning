# rwkv_lightning 🕊️ ⚡
RWKV Batch infer backend Base on [Albatross](https://github.com/BlinkDL/Albatross) 🕊️ and [Robyn](https://github.com/sparckles/Robyn) 🦀 
## Usage
```bash
python main_robyn.py --model-path <your model path> --port <your port number>
```
## Test API quickly
```bash
bash ./test_curl.sh 
```
## API Docs
1. Batch synchronous Translate 
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
2. ```v1/chat/completions``` [Fastest Speed But Only support noise temperature decode] 
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
    "stream": true
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
    "stream": true
  }'
```
2. ```v2/chat/completions``` [Little slower than V1 But Only support all decode parameters]
- Streaming synchronous batch processing
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
    "alpha_presence": 0.5,
    "alpha_frequency": 0.5,
    "alpha_decay": 0.996,
    "chunk_size": 128,
    "stream": true
  }'
```
- Non-streaming synchronous batch processing
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
    "alpha_presence": 0.5,
    "alpha_frequency": 0.5,
    "alpha_decay": 0.996,
    "chunk_size": 32,
    "stream": false
  }'
```
3. ```v3/chat/completions``` [Little slower than V1 But Only support all decode parameters]

**--- Under construction 🚧 🥲 ---** 
- Streaming asynchronous batch processing
```bash
curl -X POST http://localhost:8000/v3/chat/completions \
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
    "alpha_presence": 0.5,
    "alpha_frequency": 0.5,
    "alpha_decay": 0.996,
    "chunk_size": 128,
    "stream": true
  }'
```
- Non-streaming asynchronous batch processing
```bash
curl -X POST http://localhost:8000/v3/chat/completions \
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
    "alpha_presence": 0.5,
    "alpha_frequency": 0.5,
    "alpha_decay": 0.996,
    "chunk_size": 128,
    "stream": false
  }'
```