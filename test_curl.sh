#!/bin/bash

# 非流式请求
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
    "noise": 1.5,
    "pad_zero": true,
    "alpha_presence": 0.5,
    "alpha_frequency": 0.5,
    "alpha_decay": 0.996,
    "chunk_size": 1,
    "stream": false
  }'

echo -e "\n\n========================================\n\n"

# 流式请求
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
    "noise": 1.5,
    "pad_zero": true,
    "alpha_presence": 0.5,
    "alpha_frequency": 0.5,
    "alpha_decay": 0.996,
    "chunk_size": 8,
    "stream": true
  }'
