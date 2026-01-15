#!/bin/bash

while true; do
    echo -e "\n[执行] V1 流式批处理请求 $(date '+%Y-%m-%d %H:%M:%S')\n"
    
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
        "chunk_size": 8,
        "password": "rwkv7_7.2b"
      }'
    
    echo -e "\n\n[完成] 本次请求已结束，等待下次执行...\n"
    echo "----------------------------------------"
done