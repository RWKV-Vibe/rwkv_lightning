#!/usr/bin/env bash
set -euo pipefail

# RWKV Lightning API smoke tests.
#
# Usage:
#   bash test/test_curl.sh
#
# Optional env:
#   BASE_URL=http://127.0.0.1:8000
#   PASSWORD=your_api_password
#   MAX_TOKENS=64
#   RUN_STREAM=1
#   RUN_STATE=1
#   RUN_OPENAI=1
#   RUN_BIG_BATCH=0
#   STATE_MEMORY_ROUNDS=3
#   STATE_TEST_CODE=blue-17

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
PASSWORD="${PASSWORD:-${RWKV_PASSWORD:-}}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RUN_STREAM="${RUN_STREAM:-1}"
RUN_STATE="${RUN_STATE:-1}"
RUN_OPENAI="${RUN_OPENAI:-1}"
RUN_BIG_BATCH="${RUN_BIG_BATCH:-0}"
SESSION_ID="${SESSION_ID:-curl-smoke-$(date +%s)}"
STATE_MEMORY_ROUNDS="${STATE_MEMORY_ROUNDS:-5}"
STATE_TEST_CODE="${STATE_TEST_CODE:-blue-17}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  PYTHON_BIN=python
fi

json_escape() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.dumps(sys.argv[1], ensure_ascii=False))' "$1"
}

PASSWORD_JSON="$(json_escape "$PASSWORD")"

auth_header_args=()
if [[ -n "$PASSWORD" ]]; then
  auth_header_args=(-H "Authorization: Bearer $PASSWORD")
fi

section() {
  printf '\n\n========== %s ==========\n\n' "$1"
}

curl_json() {
  local method="$1"
  local path="$2"
  local data="${3:-}"
  if [[ -n "$data" ]]; then
    curl --fail-with-body -sS -X "$method" "$BASE_URL$path" \
      -H "Content-Type: application/json" \
      "${auth_header_args[@]}" \
      -d "$data"
  else
    curl --fail-with-body -sS -X "$method" "$BASE_URL$path" \
      "${auth_header_args[@]}"
  fi
  printf '\n'
}

curl_stream() {
  local path="$1"
  local data="$2"
  curl --fail-with-body -sS -N -X POST "$BASE_URL$path" \
    -H "Content-Type: application/json" \
    "${auth_header_args[@]}" \
    -d "$data"
  printf '\n'
}

chat_payload() {
  local stream="$1"
  cat <<JSON
{
  "model": "rwkv7",
  "contents": [
    "User: 用一句话介绍 RWKV。\\n\\nAssistant: <think>\\n</think>\\n",
    "User: Translate to Chinese: The weather is nice today.\\n\\nAssistant: <think>\\n</think>\\n"
  ],
  "max_tokens": $MAX_TOKENS,
  "stop_tokens": ["\\nUser:"],
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.6,
  "pad_zero": true,
  "alpha_presence": 0.8,
  "alpha_frequency": 0.2,
  "alpha_decay": 0.996,
  "chunk_size": 8,
  "stream": $stream,
  "password": $PASSWORD_JSON
}
JSON
}

section "Server"
echo "BASE_URL=$BASE_URL"
echo "MAX_TOKENS=$MAX_TOKENS"
echo "STATE_MEMORY_ROUNDS=$STATE_MEMORY_ROUNDS"
if [[ -n "$PASSWORD" ]]; then
  echo "PASSWORD=<set>"
else
  echo "PASSWORD=<empty>"
fi

section "GET /v1/models"
curl_json GET "/v1/models"

section "POST /translate/v1/batch-translate"
curl_json POST "/translate/v1/batch-translate" "$(cat <<JSON
{
  "source_lang": "en",
  "target_lang": "zh-CN",
  "text_list": ["Hello world!", "Good morning."]
}
JSON
)"

section "POST /v1/chat/completions non-stream"
curl_json POST "/v1/chat/completions" "$(chat_payload false)"

if [[ "$RUN_STREAM" == "1" ]]; then
  section "POST /v1/chat/completions stream"
  curl_stream "/v1/chat/completions" "$(chat_payload true)"
fi

section "POST /v2/chat/completions non-stream"
curl_json POST "/v2/chat/completions" "$(chat_payload false)"

if [[ "$RUN_STREAM" == "1" ]]; then
  section "POST /v2/chat/completions stream"
  curl_stream "/v2/chat/completions" "$(chat_payload true)"
fi

section "POST /FIM/v1/batch-FIM"
curl_json POST "/FIM/v1/batch-FIM" "$(cat <<JSON
{
  "model": "rwkv7",
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
  "max_tokens": $MAX_TOKENS,
  "temperature": 0.8,
  "top_k": 20,
  "top_p": 0.6,
  "stream": false,
  "password": $PASSWORD_JSON
}
JSON
)"

if [[ "$RUN_STATE" == "1" ]]; then
  section "POST /state/chat/completions memory init"
  curl_json POST "/state/chat/completions" "$(cat <<JSON
{
  "model": "rwkv7",
  "session_id": "$SESSION_ID",
  "contents": ["User: 请记住一个短事实：测试代号是 $STATE_TEST_CODE, 收到请只要回复一句收到。\\n\\nAssistant: <think>\\n</think>\\n"],
  "max_tokens": $MAX_TOKENS,
  "stop_tokens": ["\\nUser:"],
  "temperature": 0.8,
  "top_k": 30,
  "top_p": 0.6,
  "stream": false,
  "password": $PASSWORD_JSON
}
JSON
)"

  if (( STATE_MEMORY_ROUNDS > 0 )); then
    for ((round=1; round<=STATE_MEMORY_ROUNDS; round++)); do
      case "$round" in
        1)
          unrelated_prompt="User: 我们先聊点无关的。请用一句话解释为什么批处理能提升吞吐。\\n\\nAssistant: <think>\\n</think>\\n"
          ;;
        2)
          unrelated_prompt="User: 什么是核弹。\\n\\nAssistant: <think>\\n</think>\\n"
          ;;
        3)
          unrelated_prompt="User: 什么是盐酸右旋甲基苯丙胺\\n\\nAssistant: <think>\\n</think>\\n"
          ;;
        4)
          unrelated_prompt="User: 什么是伯努利原理\\n\\nAssistant: <think>\\n</think>\\n"
          ;;
        *)
          unrelated_prompt="User: 什么是大语言模型\\n\\nAssistant: <think>\\n</think>\\n"
          ;;
      esac

      section "POST /state/chat/completions memory distractor $round/$STATE_MEMORY_ROUNDS"
      curl_json POST "/state/chat/completions" "$(cat <<JSON
{
  "model": "rwkv7",
  "session_id": "$SESSION_ID",
  "contents": ["$unrelated_prompt"],
  "max_tokens": $MAX_TOKENS,
  "stop_tokens": ["\\nUser:"],
  "temperature": 0.8,
  "top_k": 30,
  "top_p": 0.6,
  "stream": false,
  "password": $PASSWORD_JSON
}
JSON
)"
    done
  fi

  section "POST /state/chat/completions memory recall"
  curl_json POST "/state/chat/completions" "$(cat <<JSON
{
  "model": "rwkv7",
  "session_id": "$SESSION_ID",
  "contents": ["User: 现在请回答第一轮让你记住的测试代号是什么？只回答代号本身。\\n\\nAssistant: <think>\\n</think>\\n"],
  "max_tokens": $MAX_TOKENS,
  "stop_tokens": ["\\nUser:"],
  "temperature": 0.8,
  "top_k": 30,
  "top_p": 0.6,
  "stream": false,
  "password": $PASSWORD_JSON
}
JSON
)"

  section "POST /multi_state/chat/completions"
  curl_json POST "/multi_state/chat/completions" "$(cat <<JSON
{
  "model": "rwkv7",
  "session_id": "$SESSION_ID-multi",
  "dialogue_idx": 0,
  "contents": ["User: 给我一个很短的 Python tips。\\n\\nAssistant: <think>\\n</think>\\n"],
  "max_tokens": $MAX_TOKENS,
  "stop_tokens": ["\\nUser:"],
  "temperature": 0.8,
  "top_k": 30,
  "top_p": 0.6,
  "stream": false,
  "password": $PASSWORD_JSON
}
JSON
)"

  section "POST /state/status"
  curl_json POST "/state/status" "$(cat <<JSON
{
  "password": $PASSWORD_JSON
}
JSON
)"

  section "POST /state/delete"
  curl_json POST "/state/delete" "$(cat <<JSON
{
  "session_id": "$SESSION_ID",
  "delete_prefix": true,
  "password": $PASSWORD_JSON
}
JSON
)"
fi

if [[ "$RUN_OPENAI" == "1" ]]; then
  section "GET /openai/v1/models"
  curl_json GET "/openai/v1/models"

  section "POST /openai/v1/chat/completions non-stream"
  curl_json POST "/openai/v1/chat/completions" "$(cat <<JSON
{
  "model": "rwkv7",
  "messages": [
    {"role": "system", "content": "You are helpful assistant."},
    {"role": "user", "content": "What is large language model?"}
  ],
  "max_tokens": $MAX_TOKENS,
  "temperature": 0.8,
  "top_k": 30,
  "top_p": 0.6,
  "use_prefix_cache": true,
  "stream": false,
  "password": $PASSWORD_JSON
}
JSON
)"

  if [[ "$RUN_STREAM" == "1" ]]; then
    section "POST /openai/v1/chat/completions stream"
    curl_stream "/openai/v1/chat/completions" "$(cat <<JSON
{
  "model": "rwkv7",
  "messages": [
    {"role": "user", "content": "Write one short sentence about batching."}
  ],
  "max_tokens": $MAX_TOKENS,
  "temperature": 0.8,
  "top_k": 30,
  "top_p": 0.6,
  "use_prefix_cache": true,
  "stream": true,
  "password": $PASSWORD_JSON
}
JSON
)"
  fi
fi

section "GET /big_batch/completions-status"
curl_json GET "/big_batch/completions-status"

section "POST /big_batch/clean"
curl_json POST "/big_batch/clean" "$(cat <<JSON
{
  "password": $PASSWORD_JSON
}
JSON
)"

if [[ "$RUN_BIG_BATCH" == "1" ]]; then
  section "POST /big_batch/completions stream"
  curl_stream "/big_batch/completions" "$(cat <<JSON
{
  "contents": [
    "User: 输出一个短句。\\n\\nAssistant: <think>\\n</think>\\n",
    "User: 输出另一个短句。\\n\\nAssistant: <think>\\n</think>\\n"
  ],
  "max_tokens": $MAX_TOKENS,
  "temperature": 0.8,
  "stop_tokens": ["\\nUser:"],
  "chunk_size": 8,
  "password": $PASSWORD_JSON
}
JSON
)"
fi

section "Done"
echo "All requested curl smoke tests finished."
