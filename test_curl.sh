#!/bin/bash

# ============================================================================
# RWKV Lightning API 测试脚本
# ============================================================================
# 
# 此脚本测试以下API端点：
# 1. /translate/v1/batch-translate - 批量翻译（非流式）
# 2. /v1/chat/completions          - V1：基础批处理（支持流式和非流式）
# 3. /v2/chat/completions          - V2：连续批处理（支持流式和非流式）
# 4. /v3/chat/completions          - V3：异步高并发（VLLM风格，支持流式和非流式）
# 
# 使用方法：
#   bash test_curl.sh        # 执行所有测试
# 
# ============================================================================

# ============================================================================
# 测试 1: 批量翻译 (中译英)
# ============================================================================
echo -e "\n[测试 1/7] 批量翻译 (英 -> 中)\n"
curl -X POST http://localhost:8000/translate/v1/batch-translate \
         -H "Content-Type: application/json" \
         -d '{
           "source_lang": "en",
           "target_lang": "zh-CN",
           "text_list": ["Hello world!", "Good morning"]
         }'
echo -e "\n\n========================================\n\n"

echo -e "\n[测试 2/7] 批量翻译 (中 -> 英)\n"
curl -X POST http://localhost:8000/translate/v1/batch-translate \
         -H "Content-Type: application/json" \
         -d '{
           "source_lang": "zh-CN",
           "target_lang": "en",
           "text_list": ["你好世界", "早上好"]
         }'
echo -e "\n\n========================================\n\n"

# ============================================================================
# 测试 3: V1 - 非流式和流式批处理 (最快)
# ============================================================================
echo -e "\n[测试 3/7] V1 非流式批处理\n"
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
echo -e "\n[测试 3/7] V1 流式批处理\n"
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


echo -e "\n\n========================================\n\n"

# ============================================================================
# 测试 4: V2 - 非流式连续批处理
# ============================================================================
echo -e "\n[测试 4/7] V2 非流式连续批处理\n"
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

echo -e "\n\n========================================\n\n"

# ============================================================================
# 测试 5: V2 - 流式连续批处理
# ============================================================================
echo -e "\n[测试 5/7] V2 流式连续批处理\n"
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

echo -e "\n\n========================================\n\n"

# ============================================================================
# 测试 6: V3 - 非流式异步高并发
# ============================================================================
echo -e "\n[测试 6/7] V3 非流式异步高并发\n"
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

echo -e "\n\n========================================\n\n"

# ============================================================================
# 测试 7: V3 - 流式异步高并发 (VLLM风格)
# ============================================================================
echo -e "\n[测试 7/7] V3 流式异步高并发\n"
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

echo -e "\n\n========================================\n\n"
echo "[完成] 所有测试已完成！"
echo "========================================\n"