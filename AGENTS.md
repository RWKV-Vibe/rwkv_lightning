# AGENTS.md

## Main Surface

- Use `python app.py --model-path <path> --runtime <fp16|int8> --port 8000`.
- `app_big_batch.py` is now only a compatibility launcher for the same unified server.
- The primary public API is `POST /v1/chat/completions`.

## Request Routing

- `scheduler="standard"` = normal batch chat.
- `scheduler="continuous"` = continuous batching.
- `scheduler="throughput"` = highest-throughput batch decode path.
- `session_id` on `/v1/chat/completions` = stateful single-session chat.
- `messages` / `system` are accepted on `/v1/chat/completions`; you do not need `/openai/v1/chat/completions` unless testing compatibility.

## Model Selection

- The server loads exactly one runtime per process: `fp16` or `int8`.
- The request `model` field is validated against the loaded aliases instead of being echoed blindly.
- `/v1/models` and `/healthz` expose the real loaded model id and runtime.

## Advanced / Legacy Endpoints

- Advanced endpoints are kept but are not the main public surface:
  - `/v1/advanced/translate`
  - `/v1/advanced/fim`
  - `/v1/advanced/multi_state`
  - `/v1/admin/state/status`
  - `/v1/admin/state/delete`
- Legacy compatibility routes still exist:
  - `/v2/chat/completions`
  - `/state/chat/completions`
  - `/big_batch/completions`
  - `/openai/v1/chat/completions`

## Performance Facts

- Default path uses `torch.compile(mode='max-autotune-no-cudagraphs')`; set `RWKV_USE_JIT=1` to force JIT.
- `rwkv7_int8.py` is now a real runtime path through `model_load/model_loader.py`; do not assume fp16 is the only served model.
- Continuous batching no longer uses `torch.cat` slot compaction in `infer/inference.py`.
- Prefix cache now applies to canonical stateful and continuous paths too.
- DB-backed state persistence in `state_manager/state_pool.py` uses compressed blobs when smaller.

## Existing Low-Level Optimizations

- RWKV recurrence kernel is already fused in `infer/rwkv_batch/cuda/rwkv7_state_fwd_fp16.cu`.
- Sampling kernel is already fused in `infer/rwkv_batch/cuda/sampling.cu`.
- INT8 weight dequant + matmul already exists in `infer/rwkv_batch/cuda/operators.cu`.
- Sparse matvec already exists in CUDA and Triton; do not list it as missing work.

## Verification

- Syntax check: `python3 -c "import ast; [ast.parse(open(f).read()) for f in [...]]"`
- Smoke test a live server: `bash ./test/test_curl.sh`
- Streaming benchmark: `python test/benchmark_api.py --url http://localhost:8000/v1/chat/completions`

## Repo Notes

- There is no `pyproject.toml` or `requirements.txt`; install deps manually.
- `Dockerfile` now runs the inference server, not `opencode`.
- Backup files `README.md.bak` and `infer/rwkv_batch/rwkv7_int8.py.bak` were removed; `bak/` still exists as archive code and is not part of the live product.
