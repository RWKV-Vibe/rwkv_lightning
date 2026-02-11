#!/usr/bin/bash
set -euo pipefail

URL="http://localhost:8000/v2/chat/completions"
DATASET="/mnt/pc411_data/rwkv_lightning/ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPTS=10

python ./test/benchmark_api.py \
  --url "$URL" \
  --dataset "$DATASET" \
  --num-prompts "$NUM_PROMPTS" \
  "$@"
