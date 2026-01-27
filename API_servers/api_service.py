import json
from typing import Optional

from pydantic import BaseModel
from robyn import Robyn, Response, StreamingResponse

from state_manager.state_pool import get_state_manager, remove_session_from_any_level


class ChatRequest(BaseModel):
    model: str = "rwkv7"
    contents: list[str] = []
    prefix: list[str] = []
    suffix: list[str] = []
    max_tokens: int = 50
    stop_tokens: list[int] = [0, 261, 24281]
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 0.3
    noise: float = 1.5
    stream: bool = False
    pad_zero: bool = True
    alpha_presence: float = 0.5
    alpha_frequency: float = 0.5
    alpha_decay: float = 0.996
    enable_think: bool = False
    chunk_size: int = 4
    password: Optional[str] = None
    session_id: Optional[str] = None


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

    @app.after_request()
    def after_request(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.options("/")
    @app.options("/v1/chat/completions")
    @app.options("/v2/chat/completions")
    @app.options("/v3/chat/completions")
    @app.options("/translate/v1/batch-translate")
    @app.options("/FIM/v1/batch-FIM")
    @app.options("/state/chat/completions")
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

    @app.post("/v1/chat/completions")
    async def chat_completions(request):
        body = json.loads(request.body)
        req = ChatRequest(**body)

        if password and req.password != password:
            return Response(
                status_code=401,
                description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
                headers={"Content-Type": "application/json"},
            )

        prompts = req.contents

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
                    stop_tokens=req.stop_tokens,
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
            stop_tokens=req.stop_tokens,
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
            "object": "chat.completion",
            "model": req.model,
            "choices": choices,
        }
        return Response(
            status_code=200,
            description=json.dumps(response, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
        )

    @app.post("/v2/chat/completions")
    async def continuous_batching(request):
        try:
            body = json.loads(request.body)
            req = ChatRequest(**body)

            if password and req.password != password:
                return Response(
                    status_code=401,
                    description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
                    headers={"Content-Type": "application/json"},
                )

            prompts = req.contents

            if not prompts:
                return Response(
                    status_code=400,
                    description=json.dumps({"error": "Empty prompts list"}),
                    headers={"Content-Type": "application/json"},
                )

            if req.stream:
                return StreamingResponse(
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
                    ),
                    media_type="text/event-stream",
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
                "object": "chat.completion",
                "model": req.model,
                "choices": choices,
            }
            return Response(
                status_code=200,
                description=json.dumps(response, ensure_ascii=False),
                headers={"Content-Type": "application/json"},
            )
        except json.JSONDecodeError as exc:
            return Response(
                status_code=400,
                description=json.dumps({"error": f"Invalid JSON: {str(exc)}"}),
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            import traceback

            print(f"[ERROR] /v2/chat/completions: {traceback.format_exc()}")
            return Response(
                status_code=500,
                description=json.dumps({"error": str(exc)}),
                headers={"Content-Type": "application/json"},
            )

    @app.post("/v3/chat/completions")
    async def v3_chat_completions(request):
        try:
            body = json.loads(request.body)
            if "contents" not in body and "messages" in body:
                msgs = body.get("messages") or []
                user_texts = [m.get("content", "") for m in msgs if m.get("role") == "user"]
                if not user_texts and msgs:
                    user_texts = [m.get("content", "") for m in msgs]
                body = {**body, "contents": user_texts}

            req = ChatRequest(**body)

            if password and req.password != password:
                return Response(
                    status_code=401,
                    description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
                    headers={"Content-Type": "application/json"},
                )

            prompts = req.contents
            if req.enable_think:
                prompts_formatted = [f"User: {q}\n\nAssistant: <think" for q in prompts]
            else:
                prompts_formatted = [f"User: {q}\n\nAssistant: <think>\n</think>" for q in prompts]

            if not prompts:
                return Response(
                    status_code=400,
                    description=json.dumps({"error": "Empty prompts list"}),
                    headers={"Content-Type": "application/json"},
                )

            if req.stream:
                return StreamingResponse(
                    engine.graph_infer_stream(
                        inputs=prompts_formatted,
                        stop_tokens=req.stop_tokens,
                        max_generate_tokens=req.max_tokens,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        alpha_presence=req.alpha_presence,
                        alpha_frequency=req.alpha_frequency,
                        alpha_decay=req.alpha_decay,
                        chunk_size=req.chunk_size,
                    ),
                    media_type="text/event-stream",
                )

            results = await engine.graph_generate(
                inputs=prompts_formatted,
                stop_tokens=req.stop_tokens,
                max_generate_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                alpha_presence=req.alpha_presence,
                alpha_frequency=req.alpha_frequency,
                alpha_decay=req.alpha_decay,
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
                "id": "rwkv7-batch-v3",
                "object": "chat.completion",
                "model": req.model,
                "choices": choices,
            }
            return Response(
                status_code=200,
                description=json.dumps(response, ensure_ascii=False),
                headers={"Content-Type": "application/json"},
            )
        except json.JSONDecodeError as exc:
            return Response(
                status_code=400,
                description=json.dumps({"error": f"Invalid JSON: {str(exc)}"}),
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            import traceback

            print(f"[ERROR] /v3/chat/completions: {traceback.format_exc()}")
            return Response(
                status_code=500,
                description=json.dumps({"error": str(exc)}),
                headers={"Content-Type": "application/json"},
            )

    @app.post("/translate/v1/batch-translate")
    async def batch_translate(request):
        body = json.loads(request.body)
        req = TranslateRequest(**body)

        try:
            processed_texts = req.text_list

            prompts = []
            for text in processed_texts:
                prompt = create_translation_prompt(req.source_lang, req.target_lang, text)
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
                stop_tokens=[0],
            )

            translations_result = []
            for translation in translated_texts:
                translations_result.append(
                    {
                        "detected_source_lang": req.source_lang if req.source_lang != "auto" else "en",
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
    async def fim_completions(request):
        body = json.loads(request.body)
        req = ChatRequest(**body)

        if password and req.password != password:
            return Response(
                status_code=401,
                description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
                headers={"Content-Type": "application/json"},
            )

        prompts = []
        prefix_list = req.prefix
        suffix_list = req.suffix
        for prefix, suffix in zip(prefix_list, suffix_list):
            prompt = f"✿prefix✿✿suffix✿{suffix}✿middle✿{prefix}"
            prompts.append(prompt)

        if len(prompts) == 1:
            if req.stream:
                return StreamingResponse(
                    engine.graph_infer_stream(
                        inputs=prompts,
                        stop_tokens=req.stop_tokens,
                        max_generate_tokens=req.max_tokens,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        alpha_presence=req.alpha_presence,
                        alpha_frequency=req.alpha_frequency,
                        alpha_decay=req.alpha_decay,
                        chunk_size=req.chunk_size,
                    ),
                    media_type="text/event-stream",
                )
            results = await engine.graph_generate(
                inputs=prompts,
                stop_tokens=req.stop_tokens,
                max_generate_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                alpha_presence=req.alpha_presence,
                alpha_frequency=req.alpha_frequency,
                alpha_decay=req.alpha_decay,
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
                "id": "rwkv7-batch-v3",
                "object": "chat.completion",
                "model": req.model,
                "choices": choices,
            }
            return Response(
                status_code=200,
                description=json.dumps(response, ensure_ascii=False),
                headers={"Content-Type": "application/json"},
            )

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
                    stop_tokens=req.stop_tokens,
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
            stop_tokens=req.stop_tokens,
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
        body = json.loads(request.body)
        req = ChatRequest(**body)
        session_id = req.session_id
        batch_size = len(req.contents)
        if batch_size > 1:
            return Response(
                status_code=500,
                description=json.dumps({"error": "Server Error: Requst must be single prompt !"}),
                headers={"Content-Type": "application/json"},
            )
        if password and req.password != password:
            return Response(
                status_code=401,
                description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
                headers={"Content-Type": "application/json"},
            )

        prompts = req.contents

        state_manager = get_state_manager()
        state = state_manager.get_state(session_id)

        if state is None:
            batch_size = len(prompts)
            if batch_size > 1:
                return Response(
                    status_code=500,
                    description=json.dumps(
                        {"error": "Server Error: Request must be single prompt when initializing new session!"}
                    ),
                    headers={"Content-Type": "application/json"},
                )
            state = engine.model.generate_zero_state(0)
            state_manager.put_state(session_id, state)
            print(f"[INIT] Created new state for session: {session_id}")
        else:
            print(f"[REUSE] Reusing existing state for session: {session_id}")

        if req.stream:
            return StreamingResponse(
                engine.batch_infer_stream_state(
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
                ),
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
        state_manager.put_state(session_id, state)
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
            "object": "chat.completion",
            "model": req.model,
            "choices": choices,
        }
        print("[RESPONSE] /state/chat/completions state[2]: ", state[2], "\n")
        del state
        return Response(
            status_code=200,
            description=json.dumps(response, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
        )

    @app.post("/state/status")
    async def state_status(request):
        try:
            body = json.loads(request.body) if request.body else {}
            if password and body.get("password") != password:
                return Response(
                    status_code=401,
                    description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
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
                manager.db_cursor.execute("SELECT last_updated FROM sessions WHERE session_id = ?", (session_id,))
                row = manager.db_cursor.fetchone()
                if row:
                    timestamp = row[0]
                    readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
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
    async def state_delete(request):
        try:
            body = json.loads(request.body)
            session_id = body.get("session_id")

            if not session_id:
                return Response(
                    status_code=400,
                    description=json.dumps({"error": "Missing session_id parameter"}),
                    headers={"Content-Type": "application/json"},
                )

            if password and body.get("password") != password:
                return Response(
                    status_code=401,
                    description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
                    headers={"Content-Type": "application/json"},
                )

            success = remove_session_from_any_level(session_id)

            if success:
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

            print(f"[StatePool] Delete session {session_id}: {'Success' if success else 'Not Found'}")

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

    return app
