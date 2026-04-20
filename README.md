# rwkv_lightning

RWKV-7 inference server built on Robyn with CUDA/HIP kernels, batch decoding, continuous batching, stateful sessions, and prefix/state caching.

## What is the main API?

Use `POST /v1/chat/completions`.

The server now exposes one canonical chat API with scheduler modes:

- `scheduler="standard"` for normal batch chat
- `scheduler="continuous"` for continuous batching
- `scheduler="throughput"` for the highest-throughput decode path
- `session_id="..."` for stateful chat

Legacy endpoints like `/v2/chat/completions`, `/state/chat/completions`, `/big_batch/completions`, and `/openai/v1/chat/completions` still exist as compatibility surfaces.

## Install

GPU is required.

### NVIDIA CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install robyn pydantic ninja numpy
```

Optional:

```bash
pip install flashinfer-python
```

### AMD ROCm

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install robyn pydantic ninja numpy
```

## Run

### FP16

```bash
python app.py --model-path <path-to-model.pth> --runtime fp16 --port 8000 --password <password>
```

### INT8

```bash
python app.py --model-path <path-to-model.pth> --runtime int8 --port 8000 --password <password>
```

Compatibility launcher:

```bash
python app_big_batch.py --model-path <path-to-model.pth> --runtime fp16 --port 8000
```

`app_big_batch.py` now starts the same unified server. Use `scheduler="throughput"` or `/big_batch/completions` for the throughput path.

## Main request shape

```json
{
  "model": "rwkv7",
  "runtime": "fp16",
  "scheduler": "standard",
  "contents": ["User: hello\n\nAssistant: <think>\n</think>\n"],
  "max_tokens": 512,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 0.6,
  "stream": true,
  "session_id": null,
  "use_prefix_cache": true,
  "password": "<password>"
}
```

You can also send OpenAI-style `messages` and `system` fields to `/v1/chat/completions` for single-prompt chat.

## Examples

### Standard chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7",
    "scheduler": "standard",
    "contents": ["English: Hello world\n\nChinese:"],
    "max_tokens": 256,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

### Continuous batching

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7",
    "scheduler": "continuous",
    "contents": ["prompt 1", "prompt 2", "prompt 3"],
    "max_tokens": 256,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```

### Throughput mode

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7",
    "scheduler": "throughput",
    "contents": ["prompt 1", "prompt 2"],
    "max_tokens": 256,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```

### Stateful chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7",
    "session_id": "demo-session",
    "contents": ["User: Give me three dinner ideas.\n\nAssistant: <think>\n</think>\n"],
    "max_tokens": 256,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```

### OpenAI-style body on canonical route

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize RWKV in two sentences."}
    ],
    "max_tokens": 256,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

## Advanced routes

These are still available, but they are no longer the primary public surface:

- `POST /v1/advanced/translate`
- `POST /v1/advanced/fim`
- `POST /v1/advanced/multi_state`
- `POST /v1/admin/state/status`
- `POST /v1/admin/state/delete`

Legacy aliases are still available:

- `/translate/v1/batch-translate`
- `/FIM/v1/batch-FIM`
- `/multi_state/chat/completions`
- `/state/status`
- `/state/delete`

## OpenAI compatibility

Compatibility route:

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <password>" \
  -d '{
    "model": "rwkv7",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

## Health and models

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/v1/models
```

## Benchmarks and tests

Quick smoke test against a running server:

```bash
bash ./test/test_curl.sh
```

Streaming benchmark client:

```bash
python test/benchmark_api.py --url http://localhost:8000/v1/chat/completions
```

## Notes

- Default runtime path uses `torch.compile(mode='max-autotune-no-cudagraphs')`
- Set `RWKV_USE_JIT=1` to force the older JIT path
- Slow first request is expected with `torch.compile`
- Stateful chat supports single-prompt requests only
- Prefix cache is available on canonical chat requests with `use_prefix_cache=true`
