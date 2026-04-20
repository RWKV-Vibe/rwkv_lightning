import json
import os
from threading import Lock
from typing import Optional

import torch
from pydantic import BaseModel, Field, ValidationError
from robyn import Robyn, Response, StreamingResponse

from API_servers.openai_routes import format_openai_prompt, register_openai_routes
from state_manager.state_pool import get_state_manager, remove_session_from_any_level


class ChatRequest(BaseModel):
    model: str = "rwkv7"
    runtime: Optional[str] = None
    scheduler: str = "auto"
    contents: list[str] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)
    system: Optional[str] = None
    prefix: list[str] = Field(default_factory=list)
    suffix: list[str] = Field(default_factory=list)
    max_tokens: int = 8192
    stop_tokens: list[str] = Field(default_factory=lambda: ["\nUser:"])
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.6
    noise: float = 1.5
    stream: bool = False
    pad_zero: bool = True
    alpha_presence: float = 2
    alpha_frequency: float = 0.2
    alpha_decay: float = 0.996
    enable_think: bool = False
    chunk_size: int = 4
    password: Optional[str] = None
    session_id: Optional[str] = None
    dialogue_idx: Optional[int] = 0
    use_prefix_cache: bool = True


class TranslateRequest(BaseModel):
    source_lang: str = "auto"
    target_lang: str
    text_list: list[str]
    placeholders: Optional[list[str]] = None


class TranslateResponse(BaseModel):
    translations: list[dict]


def create_translation_prompt(source_lang, target_lang, text):
    lang_names = {
        "zh-CN": "Chinese",
        "zh-TW": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
    }

    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)

    prompt = f"{source_name}: {text}\n\n{target_name}:"
    return prompt


def create_app(engine, password=None):
    app = Robyn(__file__)
    dialogue_idx_lock = Lock()
    dialogue_idx_counters: dict[str, int] = {}

    def _json_response(status_code: int, payload: dict):
        return Response(
            status_code=status_code,
            description=json.dumps(payload, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
        )

    def _normalize_state_prompts(prompts: list[str], reuse_existing_state: bool) -> list[str]:
        if not reuse_existing_state:
            return prompts

        normalized_prompts = []
        for prompt in prompts:
            if prompt and not prompt.startswith("\n\n"):
                normalized_prompts.append(f"\n\n{prompt}")
            else:
                normalized_prompts.append(prompt)
        return normalized_prompts

    def _collect_session_indices(state_manager, session_index: str) -> list[int]:
        prefix = f"{session_index}:"
        all_states = state_manager.list_all_states()
        indices = []
        for key in (
            all_states["l1_cache"] + all_states["l2_cache"] + all_states["database"]
        ):
            if key.startswith(prefix):
                tail = key[len(prefix) :]
                if tail.isdigit():
                    indices.append(int(tail))
        return indices

    def _allocate_next_dialogue_idx(state_manager, session_index: str) -> int:
        with dialogue_idx_lock:
            if session_index in dialogue_idx_counters:
                next_idx = dialogue_idx_counters[session_index]
                dialogue_idx_counters[session_index] = next_idx + 1
                return next_idx

            indices = _collect_session_indices(state_manager, session_index)
            max_idx = max(indices) if indices else 0
            next_idx = max_idx + 1
            dialogue_idx_counters[session_index] = next_idx + 1
            return next_idx

    served_model_id = getattr(
        engine.args,
        "served_model_id",
        os.path.basename(f"{engine.args.MODEL_NAME}"),
    )
    served_runtime = getattr(engine.args, "runtime", "fp16")
    served_aliases = set(getattr(engine.args, "model_aliases", (served_model_id,)))

    def _require_auth(req: ChatRequest):
        if password and req.password != password:
            return _json_response(
                401, {"error": "Unauthorized: invalid or missing password"}
            )
        return None

    def _validate_model_target(req: ChatRequest):
        requested_model = (req.model or served_model_id).strip()
        if requested_model and requested_model not in served_aliases:
            return _json_response(
                404,
                {
                    "error": f"Model '{requested_model}' is not served by this process",
                    "available_models": [served_model_id],
                },
            )

        if req.runtime and req.runtime.strip().lower() != served_runtime:
            return _json_response(
                400,
                {
                    "error": f"Runtime '{req.runtime}' is not available on this server",
                    "served_runtime": served_runtime,
                },
            )
        return None

    def _resolve_chat_prompts(req: ChatRequest, body: dict) -> list[str]:
        if req.contents:
            return req.contents
        if req.messages or req.system:
            return [format_openai_prompt(body, req.enable_think)]
        return []

    def _resolve_scheduler(req: ChatRequest, scheduler_override: Optional[str] = None) -> str:
        scheduler = (scheduler_override or req.scheduler or "auto").strip().lower()
        scheduler_aliases = {
            "auto": "auto",
            "standard": "standard",
            "default": "standard",
            "continuous": "continuous",
            "throughput": "throughput",
            "big_batch": "throughput",
            "state": "stateful",
            "stateful": "stateful",
        }
        if scheduler not in scheduler_aliases:
            raise ValueError(
                "Unsupported scheduler. Expected one of: auto, standard, continuous, throughput, stateful"
            )
        resolved = scheduler_aliases[scheduler]
        if resolved == "auto":
            return "stateful" if req.session_id else "standard"
        return resolved

    def _build_choices(results: list[str]) -> list[dict]:
        return [
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
            for i, text in enumerate(results)
        ]

    def _chat_completion_payload(req: ChatRequest, results: list[str], **extra):
        payload = {
            "id": "rwkv7-chat",
            "object": "chat.completion",
            "model": served_model_id,
            "runtime": served_runtime,
            "scheduler": req.scheduler,
            "choices": _build_choices(results),
        }
        payload.update(extra)
        return payload

    def _stream_response(iterator):
        return StreamingResponse(iterator, media_type="text/event-stream")

    async def _execute_standard_chat(req: ChatRequest, prompts: list[str]):
        if req.stream:
            return _stream_response(
                engine.batch_infer_stream(
                    prompts=prompts,
                    max_length=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    alpha_presence=req.alpha_presence,
                    alpha_frequency=req.alpha_frequency,
                    alpha_decay=req.alpha_decay,
                    stop_tokens=req.stop_tokens,
                    chunk_size=req.chunk_size,
                )
            )

        results = engine.batch_generate(
            prompts=prompts,
            max_length=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            stop_tokens=req.stop_tokens,
        )
        return _json_response(200, _chat_completion_payload(req, results, scheduler="standard"))

    async def _execute_continuous_chat(req: ChatRequest, prompts: list[str]):
        prefix_cache_manager = get_state_manager() if req.use_prefix_cache else None
        if req.stream:
            return _stream_response(
                engine.continuous_batching_stream(
                    inputs=prompts,
                    stop_tokens=req.stop_tokens,
                    max_generate_tokens=req.max_tokens,
                    batch_size=len(prompts),
                    pad_zero=req.pad_zero,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    alpha_presence=req.alpha_presence,
                    alpha_frequency=req.alpha_frequency,
                    alpha_decay=req.alpha_decay,
                    chunk_size=req.chunk_size,
                    prefix_cache_manager=prefix_cache_manager,
                )
            )

        results = engine.continuous_batching(
            inputs=prompts,
            stop_tokens=req.stop_tokens,
            max_generate_tokens=req.max_tokens,
            batch_size=len(prompts),
            pad_zero=req.pad_zero,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            prefix_cache_manager=prefix_cache_manager,
        )
        return _json_response(
            200, _chat_completion_payload(req, results, scheduler="continuous")
        )

    async def _execute_throughput_chat(req: ChatRequest, prompts: list[str]):
        if req.stream:
            return _stream_response(
                engine.big_batch_stream(
                    prompts=prompts,
                    max_length=req.max_tokens,
                    temperature=req.temperature,
                    stop_tokens=req.stop_tokens,
                    chunk_size=req.chunk_size,
                )
            )

        results = engine.big_batch_generate(
            prompts=prompts,
            max_length=req.max_tokens,
            temperature=req.temperature,
            stop_tokens=req.stop_tokens,
        )
        return _json_response(
            200, _chat_completion_payload(req, results, scheduler="throughput")
        )

    async def _execute_stateful_chat(req: ChatRequest, prompts: list[str]):
        if len(prompts) != 1:
            return _json_response(400, {"error": "Stateful chat only supports a single prompt"})
        if not req.session_id:
            return _json_response(400, {"error": "Stateful chat requires session_id"})

        session_id = req.session_id
        state_manager = get_state_manager()
        state = state_manager.get_state(session_id)
        had_existing_state = state is not None

        if state is None:
            state = engine.model.generate_zero_state(0)
            state_manager.put_state(session_id, state)
            print(f"[INIT] Created new state for session: {session_id}")
        else:
            print(f"[REUSE] Reusing existing state for session: {session_id}")

        prompts = _normalize_state_prompts(
            prompts, reuse_existing_state=had_existing_state
        )
        prefix_cache_manager = state_manager if req.use_prefix_cache else None
        use_cuda_graph = torch.cuda.is_available()

        if req.stream:
            infer_fn = (
                engine.graph_infer_stream_state
                if use_cuda_graph
                else engine.batch_infer_stream_state
            )
            return _stream_response(
                infer_fn(
                    prompts=prompts,
                    state=state,
                    max_length=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    alpha_presence=req.alpha_presence,
                    alpha_frequency=req.alpha_frequency,
                    alpha_decay=req.alpha_decay,
                    stop_tokens=req.stop_tokens,
                    chunk_size=req.chunk_size,
                    session_id=session_id,
                    state_manager=state_manager,
                    prefix_cache_manager=prefix_cache_manager,
                )
            )

        infer_fn = (
            engine.graph_generate_state if use_cuda_graph else engine.batch_generate_state
        )
        results = infer_fn(
            prompts=prompts,
            state=state,
            max_length=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            stop_tokens=req.stop_tokens,
            session_id=session_id,
            prefix_cache_manager=prefix_cache_manager,
        )
        state_manager.put_state(session_id, state)
        print("[RESPONSE] /state/chat/completions state[2]: ", state[2], "\n")
        del state
        return _json_response(
            200,
            _chat_completion_payload(
                req,
                results,
                scheduler="stateful",
                session_id=session_id,
            ),
        )

    async def _dispatch_chat_request(body: dict, scheduler_override: Optional[str] = None):
        req = ChatRequest(**body)

        auth_error = _require_auth(req)
        if auth_error is not None:
            return auth_error

        target_error = _validate_model_target(req)
        if target_error is not None:
            return target_error

        prompts = _resolve_chat_prompts(req, body)
        if not prompts:
            return _json_response(400, {"error": "Empty prompts list"})

        scheduler = _resolve_scheduler(req, scheduler_override=scheduler_override)
        req.scheduler = scheduler

        if scheduler == "standard":
            return await _execute_standard_chat(req, prompts)
        if scheduler == "continuous":
            return await _execute_continuous_chat(req, prompts)
        if scheduler == "throughput":
            return await _execute_throughput_chat(req, prompts)
        return await _execute_stateful_chat(req, prompts)

    @app.after_request()
    def after_request(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.options("/")
    @app.options("/healthz")
    @app.options("/v1/models")
    @app.options("/v1/chat/completions")
    @app.options("/v1/advanced/translate")
    @app.options("/v1/advanced/fim")
    @app.options("/v1/advanced/multi_state")
    @app.options("/v1/admin/state/status")
    @app.options("/v1/admin/state/delete")
    @app.options("/v2/chat/completions")
    @app.options("/translate/v1/batch-translate")
    @app.options("/FIM/v1/batch-FIM")
    @app.options("/state/chat/completions")
    @app.options("/multi_state/chat/completions")
    @app.options("/big_batch/completions")
    @app.options("/openai/v1/chat/completions")
    async def handle_options():
        return Response(
            status_code=204,
            description="",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            },
        )

    @app.get("/healthz")
    async def healthz():
        return _json_response(
            200,
            {
                "status": "ok",
                "model": served_model_id,
                "runtime": served_runtime,
            },
        )

    @app.get("/v1/models")
    async def list_models():
        response = {
            "object": "list",
            "data": [
                {
                    "id": served_model_id,
                    "object": "model",
                    "owned_by": "rwkv_lightning",
                    "root": getattr(engine.args, "model_name", served_model_id),
                    "runtime": served_runtime,
                    "aliases": sorted(served_aliases),
                }
            ],
        }
        return _json_response(200, response)

    @app.post("/v1/chat/completions")
    async def chat_completions(request):
        try:
            body = json.loads(request.body)
            return await _dispatch_chat_request(body)
        except json.JSONDecodeError as exc:
            return _json_response(400, {"error": f"Invalid JSON: {str(exc)}"})
        except ValidationError as exc:
            return _json_response(400, {"error": exc.errors()})
        except ValueError as exc:
            return _json_response(400, {"error": str(exc)})
        except Exception as exc:
            import traceback

            print(f"[ERROR] /v1/chat/completions: {traceback.format_exc()}")
            return _json_response(500, {"error": str(exc)})

    @app.post("/v2/chat/completions")
    async def continuous_batching(request):
        try:
            body = json.loads(request.body)
            return await _dispatch_chat_request(body, scheduler_override="continuous")
        except json.JSONDecodeError as exc:
            return _json_response(400, {"error": f"Invalid JSON: {str(exc)}"})
        except ValidationError as exc:
            return _json_response(400, {"error": exc.errors()})
        except ValueError as exc:
            return _json_response(400, {"error": str(exc)})
        except Exception as exc:
            import traceback

            print(f"[ERROR] /v2/chat/completions: {traceback.format_exc()}")
            return _json_response(500, {"error": str(exc)})

    @app.post("/translate/v1/batch-translate")
    @app.post("/v1/advanced/translate")
    async def batch_translate(request):
        body = json.loads(request.body)
        req = TranslateRequest(**body)

        try:
            processed_texts = req.text_list

            prompts = []
            for text in processed_texts:
                prompt = create_translation_prompt(
                    req.source_lang, req.target_lang, text
                )
                prompts.append(prompt)

            max_tokens = 2048
            temperature = 1.0

            translated_texts = engine.batch_generate(
                prompts=prompts,
                max_length=max_tokens,
                temperature=temperature,
                top_k=1,
                top_p=0,
                alpha_presence=0,
                alpha_frequency=0,
                alpha_decay=0.996,
                stop_tokens=[],
            )

            translations_result = []
            for translation in translated_texts:
                translations_result.append(
                    {
                        "detected_source_lang": req.source_lang
                        if req.source_lang != "auto"
                        else "en",
                        "text": translation.strip(),
                    }
                )

            response = TranslateResponse(
                translations=translations_result,
            )

            return Response(
                status_code=200,
                description=response.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
        except Exception:
            error_response = TranslateResponse(
                translations=[],
            )
            return Response(
                status_code=500,
                description=error_response.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )

    @app.post("/FIM/v1/batch-FIM")
    @app.post("/v1/advanced/fim")
    async def fim_completions(request):
        body = json.loads(request.body)
        req = ChatRequest(**body)

        if password and req.password != password:
            return Response(
                status_code=401,
                description=json.dumps(
                    {"error": "Unauthorized: invalid or missing password"}
                ),
                headers={"Content-Type": "application/json"},
            )

        prompts = []
        prefix_list = req.prefix
        suffix_list = req.suffix
        for prefix, suffix in zip(prefix_list, suffix_list):
            prompt = f"✿prefix✿✿suffix✿{suffix}✿middle✿{prefix}"
            prompts.append(prompt)

        if req.stream:
            return StreamingResponse(
                engine.batch_infer_stream(
                    prompts=prompts,
                    max_length=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    alpha_presence=req.alpha_presence,
                    alpha_frequency=req.alpha_frequency,
                    alpha_decay=req.alpha_decay,
                    stop_tokens=[],
                    chunk_size=req.chunk_size,
                ),
                media_type="text/event-stream",
            )
        results = engine.batch_generate(
            prompts=prompts,
            max_length=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            stop_tokens=[],
        )
        choices = []
        for i, text in enumerate(results):
            choices.append(
                {
                    "index": i,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            )

        response = {
            "id": "rwkv7-batch",
            "object": "FIM.completion",
            "model": req.model,
            "choices": choices,
        }
        return Response(
            status_code=200,
            description=json.dumps(response, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
        )

    @app.post("/state/chat/completions")
    async def state_chat_completions(request):
        try:
            body = json.loads(request.body)
            return await _dispatch_chat_request(body, scheduler_override="stateful")
        except json.JSONDecodeError as exc:
            return _json_response(400, {"error": f"Invalid JSON: {str(exc)}"})
        except ValidationError as exc:
            return _json_response(400, {"error": exc.errors()})
        except ValueError as exc:
            return _json_response(400, {"error": str(exc)})
        except Exception as exc:
            import traceback

            print(f"[ERROR] /state/chat/completions: {traceback.format_exc()}")
            return _json_response(500, {"error": str(exc)})

    @app.post("/multi_state/chat/completions")
    @app.post("/v1/advanced/multi_state")
    async def multi_state_chat_completions(request):
        body = json.loads(request.body)
        req = ChatRequest(**body)

        if password and req.password != password:
            return Response(
                status_code=401,
                description=json.dumps(
                    {"error": "Unauthorized: invalid or missing password"}
                ),
                headers={"Content-Type": "application/json"},
            )

        if "dialogue_idx" not in body:
            return Response(
                status_code=400,
                description=json.dumps({"error": "Missing dialogue_idx parameter"}),
                headers={"Content-Type": "application/json"},
            )

        session_index = req.session_id
        if not session_index:
            return Response(
                status_code=400,
                description=json.dumps({"error": "Missing session_id parameter"}),
                headers={"Content-Type": "application/json"},
            )

        prompts = req.contents
        batch_size = len(prompts)
        if batch_size != 1:
            return Response(
                status_code=500,
                description=json.dumps(
                    {"error": "Server Error: Request must be single prompt!"}
                ),
                headers={"Content-Type": "application/json"},
            )

        dialogue_idx = int(req.dialogue_idx or 0)
        state_key = f"{session_index}:{dialogue_idx}"

        state_manager = get_state_manager()
        state = state_manager.get_state(state_key)
        had_existing_state = state is not None

        if state is None:
            if dialogue_idx != 0:
                return Response(
                    status_code=404,
                    description=json.dumps(
                        {"error": f"State not found for dialogue_idx={dialogue_idx}"}
                    ),
                    headers={"Content-Type": "application/json"},
                )
            state = engine.model.generate_zero_state(0)
            print(f"[INIT] Created new root state for session: {session_index}")
        else:
            print(f"[REUSE] Reusing state for session: {state_key}")

        prompts = _normalize_state_prompts(
            prompts, reuse_existing_state=had_existing_state
        )

        if req.stream:

            async def stream_with_dialogue_idx():
                stored = False
                try:
                    async for chunk in engine.batch_infer_stream_state(
                        prompts=prompts,
                        state=state,
                        max_length=req.max_tokens,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        alpha_presence=req.alpha_presence,
                        alpha_frequency=req.alpha_frequency,
                        alpha_decay=req.alpha_decay,
                        stop_tokens=req.stop_tokens,
                        chunk_size=req.chunk_size,
                        session_id=None,
                        state_manager=None,
                    ):
                        if chunk == "data: [DONE]\n\n" and not stored:
                            new_dialogue_idx = _allocate_next_dialogue_idx(
                                state_manager, session_index
                            )
                            new_session_id = f"{session_index}:{new_dialogue_idx}"
                            state_manager.put_state(new_session_id, state)
                            stored = True
                            meta = {
                                "object": "multi_state.dialogue_idx",
                                "session_id": new_session_id,
                                "dialogue_idx": new_dialogue_idx,
                            }
                            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
                        yield chunk
                finally:
                    if not stored:
                        new_dialogue_idx = _allocate_next_dialogue_idx(
                            state_manager, session_index
                        )
                        new_session_id = f"{session_index}:{new_dialogue_idx}"
                        state_manager.put_state(new_session_id, state)
                        print(
                            "[RESPONSE] /multi_state/chat/completions state[2]: ",
                            state[2],
                            "\n",
                        )

            return StreamingResponse(
                stream_with_dialogue_idx(),
                media_type="text/event-stream",
            )

        results = engine.batch_generate_state(
            prompts=prompts,
            state=state,
            max_length=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            stop_tokens=req.stop_tokens,
        )
        new_dialogue_idx = _allocate_next_dialogue_idx(state_manager, session_index)
        new_session_id = f"{session_index}:{new_dialogue_idx}"
        state_manager.put_state(new_session_id, state)

        choices = []
        for i, text in enumerate(results):
            choices.append(
                {
                    "index": i,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            )

        response = {
            "id": "rwkv7-multi-state",
            "object": "chat.completion",
            "model": req.model,
            "choices": choices,
            "dialogue_idx": new_dialogue_idx,
        }
        print("[RESPONSE] /multi_state/chat/completions state[2]: ", state[2], "\n")
        del state
        return Response(
            status_code=200,
            description=json.dumps(response, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
        )

    @app.post("/state/status")
    @app.post("/v1/admin/state/status")
    async def state_status(request):
        try:
            body = json.loads(request.body) if request.body else {}
            if password and body.get("password") != password:
                return Response(
                    status_code=401,
                    description=json.dumps(
                        {"error": "Unauthorized: invalid or missing password"}
                    ),
                    headers={"Content-Type": "application/json"},
                )

            manager = get_state_manager()
            all_states = manager.list_all_states()

            from datetime import datetime

            detailed_states = []

            for session_id in all_states["l1_cache"]:
                detailed_states.append(
                    {
                        "session_id": session_id,
                        "cache_level": "L1 (VRAM)",
                        "last_updated": "In Memory",
                        "timestamp": datetime.now().timestamp(),
                    }
                )

            for session_id in all_states["l2_cache"]:
                detailed_states.append(
                    {
                        "session_id": session_id,
                        "cache_level": "L2 (RAM)",
                        "last_updated": "In Memory",
                        "timestamp": datetime.now().timestamp(),
                    }
                )

            for session_id in all_states["database"]:
                manager.db_cursor.execute(
                    "SELECT last_updated FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                row = manager.db_cursor.fetchone()
                if row:
                    timestamp = row[0]
                    readable_time = datetime.fromtimestamp(timestamp).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    detailed_states.append(
                        {
                            "session_id": session_id,
                            "cache_level": "Database (Disk)",
                            "last_updated": readable_time,
                            "timestamp": timestamp,
                        }
                    )

            response_data = {
                "status": "success",
                "total_sessions": all_states["total_count"],
                "l1_cache_count": len(all_states["l1_cache"]),
                "l2_cache_count": len(all_states["l2_cache"]),
                "database_count": len(all_states["database"]),
                "sessions": detailed_states,
            }

            print(
                f"[StatePool] Status requested. Total sessions: {all_states['total_count']}, "
                f"L1: {len(all_states['l1_cache'])}, L2: {len(all_states['l2_cache'])}, "
                f"DB: {len(all_states['database'])}"
            )

            return Response(
                status_code=200,
                description=json.dumps(response_data, ensure_ascii=False),
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            print(f"[ERROR] /state/status: {exc}")
            return Response(
                status_code=500,
                description=json.dumps({"error": str(exc)}),
                headers={"Content-Type": "application/json"},
            )

    @app.post("/state/delete")
    @app.post("/v1/admin/state/delete")
    async def state_delete(request):
        try:
            body = json.loads(request.body)
            session_id = body.get("session_id")
            delete_prefix = body.get("delete_prefix", False)

            if not session_id:
                return Response(
                    status_code=400,
                    description=json.dumps({"error": "Missing session_id parameter"}),
                    headers={"Content-Type": "application/json"},
                )

            if password and body.get("password") != password:
                return Response(
                    status_code=401,
                    description=json.dumps(
                        {"error": "Unauthorized: invalid or missing password"}
                    ),
                    headers={"Content-Type": "application/json"},
                )

            success = remove_session_from_any_level(session_id)
            engine.cleanup_cuda_graph_session(session_id)
            if delete_prefix:
                manager = get_state_manager()
                prefix = f"{session_id}:"
                all_states = manager.list_all_states()
                ids = set()
                ids.update(all_states["l1_cache"])
                ids.update(all_states["l2_cache"])
                ids.update(all_states["database"])
                for sid in ids:
                    if isinstance(sid, str) and sid.startswith(prefix):
                        remove_session_from_any_level(sid)

            if success or delete_prefix:
                response_data = {
                    "status": "success",
                    "message": f"Session {session_id} deleted successfully",
                }
                status_code = 200
            else:
                response_data = {
                    "status": "not_found",
                    "message": f"Session {session_id} not found in database",
                }
                status_code = 404

            print(
                f"[StatePool] Delete session {session_id}: {'Success' if success else 'Not Found'}"
            )

            return Response(
                status_code=status_code,
                description=json.dumps(response_data, ensure_ascii=False),
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            print(f"[ERROR] /state/delete: {exc}")
            return Response(
                status_code=500,
                description=json.dumps({"error": str(exc)}),
                headers={"Content-Type": "application/json"},
            )

    @app.post("/big_batch/completions")
    async def big_batch_completions(request):
        try:
            body = json.loads(request.body)
            return await _dispatch_chat_request(body, scheduler_override="throughput")
        except json.JSONDecodeError as exc:
            return _json_response(400, {"error": f"Invalid JSON: {str(exc)}"})
        except ValidationError as exc:
            return _json_response(400, {"error": exc.errors()})
        except ValueError as exc:
            return _json_response(400, {"error": str(exc)})
        except Exception as exc:
            import traceback

            print(f"[ERROR] /big_batch/completions: {traceback.format_exc()}")
            return _json_response(500, {"error": str(exc)})

    register_openai_routes(
        app=app, engine=engine, password=password, chat_request_model=ChatRequest
    )
    return app
