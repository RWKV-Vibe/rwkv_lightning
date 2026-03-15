import json
import os
import time
import uuid
from threading import Lock
from typing import Optional

from pydantic import BaseModel
from robyn import Robyn, Response, StreamingResponse

from state_manager.state_pool import get_state_manager, remove_session_from_any_level


class ChatRequest(BaseModel):
    model: str = "rwkv7"
    contents: list[str] = []
    messages: list[dict] = []
    system: Optional[str] = None
    prefix: list[str] = []
    suffix: list[str] = []
    max_tokens: int = 8192
    stop_tokens: list[int] = [0, 261, 24281]
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

    def _extract_bearer_token(request) -> Optional[str]:
        headers = getattr(request, "headers", {}) or {}
        auth_header = headers.get("authorization") or headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        return auth_header.split(" ", 1)[1].strip()

    def _check_openai_auth(request, body: dict):
        if not password:
            return None
        bearer_token = _extract_bearer_token(request)
        body_password = body.get("password")
        if bearer_token == password or body_password == password:
            return None
        return _json_response(401, {"error": "Unauthorized: invalid or missing password"})

    def _normalize_message_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts)
        if content is None:
            return ""
        return str(content)

    def _extract_openai_prompt(body: dict) -> str:
        contents = body.get("contents") or []
        if contents:
            return str(contents[0])

        messages = body.get("messages") or []
        if messages:
            return _normalize_message_content(messages[-1].get("content", ""))
        return ""

    def _format_openai_prompt(body: dict, enable_think: bool) -> str:
        contents = body.get("contents") or []
        messages = body.get("messages") or []
        system_field = body.get("system")
        current_prompt = str(contents[0]).strip() if contents else ""

        system_parts = []
        if system_field:
            system_parts.append(str(system_field))
        history_messages = []
        role_map = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
            "tool": "Tool",
            "developer": "Developer",
        }

        for message in messages:
            role = str(message.get("role", "user")).lower()
            content = _normalize_message_content(message.get("content", ""))
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
                continue
            dialogue_role = role_map.get(role, role.capitalize() or "User")
            history_messages.append((dialogue_role, content))

        if not current_prompt and history_messages:
            current_prompt = history_messages[-1][1].strip()

        if current_prompt and history_messages and history_messages[-1][1].strip() == current_prompt:
            history_messages = history_messages[:-1]

        prompt_parts = []
        if system_parts:
            prompt_parts.append(f"System: {'\n\n'.join(system_parts)}")
        if history_messages:
            prompt_parts.append("\n\n".join(f"{role}: {content}" for role, content in history_messages))
        if current_prompt:
            prompt_parts.append(f"User: {current_prompt}")

        prompt_text = "\n\n".join(part for part in prompt_parts if part).strip()
        if not prompt_text:
            prompt_text = _extract_openai_prompt(body)

        if enable_think:
            return f"{prompt_text}\n\nAssistant: <think"
        return f"{prompt_text}\n\nAssistant: <think>\n</think>"

    def _build_openai_usage(prompt_text: str, completion_text: str) -> dict:
        prompt_tokens = len(engine.tokenizer.encode(prompt_text))
        completion_tokens = len(engine.tokenizer.encode(completion_text)) if completion_text else 0
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _collect_session_indices(state_manager, session_index: str) -> list[int]:
        prefix = f"{session_index}:"
        all_states = state_manager.list_all_states()
        indices = []
        for key in all_states["l1_cache"] + all_states["l2_cache"] + all_states["database"]:
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

    @app.after_request()
    def after_request(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.options("/")
    @app.options("/v1/models")
    @app.options("/v1/chat/completions")
    @app.options("/v2/chat/completions")
    @app.options("/v3/chat/completions")
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

    @app.get("/v1/models")
    async def list_models():
        model_name = os.path.basename(f"{engine.args.MODEL_NAME}")
        response = {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "rwkv_lightning",
                }
            ],
        }
        return Response(
            status_code=200,
            description=json.dumps(response, ensure_ascii=False),
            headers={"Content-Type": "application/json"},
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

    @app.post("/multi_state/chat/completions")
    async def multi_state_chat_completions(request):
        body = json.loads(request.body)
        req = ChatRequest(**body)

        if password and req.password != password:
            return Response(
                status_code=401,
                description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
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
                description=json.dumps({"error": "Server Error: Request must be single prompt!"}),
                headers={"Content-Type": "application/json"},
            )

        dialogue_idx = int(req.dialogue_idx or 0)
        state_key = f"{session_index}:{dialogue_idx}"

        state_manager = get_state_manager()
        state = state_manager.get_state(state_key)

        if state is None:
            if dialogue_idx != 0:
                return Response(
                    status_code=404,
                    description=json.dumps({"error": f"State not found for dialogue_idx={dialogue_idx}"}),
                    headers={"Content-Type": "application/json"},
                )
            state = engine.model.generate_zero_state(0)
            print(f"[INIT] Created new root state for session: {session_index}")
        else:
            print(f"[REUSE] Reusing state for session: {state_key}")

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
                            new_dialogue_idx = _allocate_next_dialogue_idx(state_manager, session_index)
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
                        new_dialogue_idx = _allocate_next_dialogue_idx(state_manager, session_index)
                        new_session_id = f"{session_index}:{new_dialogue_idx}"
                        state_manager.put_state(new_session_id, state)
                        print("[RESPONSE] /multi_state/chat/completions state[2]: ", state[2], "\n")

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
                    description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
                    headers={"Content-Type": "application/json"},
                )

            success = remove_session_from_any_level(session_id)
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

    @app.post("/big_batch/completions")
    async def big_batch_completions(request):
        body = json.loads(request.body)
        req = ChatRequest(**body)

        if password and req.password != password:
            return Response(
                status_code=401,
                description=json.dumps({"error": "Unauthorized: invalid or missing password"}),
                headers={"Content-Type": "application/json"},
            )

        prompts = req.contents

        return StreamingResponse(
            engine.big_batch_stream(
                prompts=prompts,
                max_length=req.max_tokens,
                temperature=req.temperature,
                stop_tokens=req.stop_tokens,
                chunk_size=req.chunk_size,
            ),
            media_type="text/event-stream",
        )

    @app.post("/openai/v1/chat/completions")
    async def openai_chat_completions(request):
        try:
            body = json.loads(request.body)
            auth_error = _check_openai_auth(request, body)
            if auth_error is not None:
                return auth_error

            prompt = _extract_openai_prompt(body)
            if not prompt and not (body.get("messages") or []):
                return _json_response(400, {"error": "Empty prompt"})

            req_body = {
                "model": body.get("model", "rwkv7"),
                "contents": [prompt],
                "messages": body.get("messages", []),
                "system": body.get("system"),
                "max_tokens": body.get("max_tokens", 4096),
                "stop_tokens": body.get("stop_tokens", [0, 261, 24281]),
                "temperature": body.get("temperature", 1.0),
                "top_k": body.get("top_k", 20),
                "top_p": body.get("top_p", 0.6),
                "stream": body.get("stream", False),
                "pad_zero": body.get("pad_zero", False),
                "alpha_presence": body.get("alpha_presence", 1),
                "alpha_frequency": body.get("alpha_frequency", 0.1),
                "alpha_decay": body.get("alpha_decay", 0.996),
                "enable_think": body.get("enable_think", False),
                "chunk_size": body.get("chunk_size", 2),
                "password": body.get("password"),
            }
            req = ChatRequest(**req_body)

            print(f"[OpenAI] Request: {req}")

            prompt_formatted = _format_openai_prompt(body, req.enable_think)

            print(f"[OpenAI] Prompt: {prompt_formatted}")

            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            model_name = req.model or os.path.basename(f"{engine.args.MODEL_NAME}")

            if req.stream:
                async def stream_openai_chunks():
                    start_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(start_chunk, ensure_ascii=False)}\n\n"

                    async for item in engine.dynamic_batch_infer_stream(
                        prompt=prompt_formatted,
                        max_generate_tokens=req.max_tokens,
                        stop_tokens=req.stop_tokens,
                        pad_zero=req.pad_zero,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        alpha_presence=req.alpha_presence,
                        alpha_frequency=req.alpha_frequency,
                        alpha_decay=req.alpha_decay,
                        chunk_size=req.chunk_size,
                    ):
                        if item["type"] == "delta" and item["text"]:
                            chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": item["text"]},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        elif item["type"] == "done":
                            chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": item["finish_reason"]}],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            break

                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_openai_chunks(), media_type="text/event-stream")

            result_text, finish_reason = await engine.dynamic_batch_generate(
                prompt=prompt_formatted,
                max_generate_tokens=req.max_tokens,
                stop_tokens=req.stop_tokens,
                pad_zero=req.pad_zero,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                alpha_presence=req.alpha_presence,
                alpha_frequency=req.alpha_frequency,
                alpha_decay=req.alpha_decay,
                chunk_size=req.chunk_size,
            )

            response = {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result_text},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": _build_openai_usage(prompt_formatted, result_text),
            }
            return _json_response(200, response)
        except json.JSONDecodeError as exc:
            return _json_response(400, {"error": f"Invalid JSON: {str(exc)}"})
        except Exception as exc:
            import traceback

            print(f"[ERROR] /openai/v1/chat/completions: {traceback.format_exc()}")
            return _json_response(500, {"error": str(exc)})
    return app
