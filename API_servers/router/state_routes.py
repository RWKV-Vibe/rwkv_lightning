import json
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded
from state_manager.state_pool import get_state_manager, remove_session_from_any_level

from API_servers.router.common import (
    allocate_next_dialogue_idx,
    check_password,
    client_closed_response,
    json_response,
    normalize_state_prompts,
    prefill_bsz_limit_response,
    reserve_prefill_capacity,
    run_sync_with_disconnect_watch,
    stream_with_prefill_queue,
)
from API_servers.router.schemas import ChatRequest


router = APIRouter()


@router.post("/state/chat/completions")
async def state_chat_completions(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password
    body = await request.json()
    req = ChatRequest(**body)
    session_id = req.session_id

    if len(req.contents) > 1:
        return json_response(500, {"error": "Server Error: Requst must be single prompt !"})

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

    prompts = req.contents
    state_manager = get_state_manager()
    state = state_manager.get_state(session_id)
    had_existing_state = state is not None

    if state is None:
        state = engine.model.generate_zero_state(0)
        state_manager.put_state(session_id, state)
        print(f"[INIT] Created new state for session: {session_id}")
    else:
        print(f"[REUSE] Reusing existing state for session: {session_id}")

    prompts = normalize_state_prompts(prompts, reuse_existing_state=had_existing_state)

    if req.stream:
        cancel_token = CancellationToken()
        stream = engine.batch_infer_stream_state(
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
            cancel_token=cancel_token,
        )
        return StreamingResponse(
            stream_with_prefill_queue(request, stream, cancel_token, 1),
            media_type="text/event-stream",
        )

    try:
        async with reserve_prefill_capacity(request, 1):
            results = await run_sync_with_disconnect_watch(
                request,
                engine.batch_generate_state,
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
    except InferenceCancelled:
        del state
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        del state
        return prefill_bsz_limit_response(exc)

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
    return response


@router.post("/multi_state/chat/completions")
async def multi_state_chat_completions(request: Request):
    app_state = request.app.state
    engine = app_state.engine
    password = app_state.password
    body = await request.json()
    req = ChatRequest(**body)

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

    if "dialogue_idx" not in body:
        return json_response(400, {"error": "Missing dialogue_idx parameter"})

    session_index = req.session_id
    if not session_index:
        return json_response(400, {"error": "Missing session_id parameter"})

    prompts = req.contents
    if len(prompts) != 1:
        return json_response(500, {"error": "Server Error: Request must be single prompt!"})

    dialogue_idx = int(req.dialogue_idx or 0)
    state_key = f"{session_index}:{dialogue_idx}"
    state_manager = get_state_manager()
    state = state_manager.get_state(state_key)
    had_existing_state = state is not None

    if state is None:
        if dialogue_idx != 0:
            return json_response(404, {"error": f"State not found for dialogue_idx={dialogue_idx}"})
        state = engine.model.generate_zero_state(0)
        print(f"[INIT] Created new root state for session: {session_index}")
    else:
        print(f"[REUSE] Reusing state for session: {state_key}")

    prompts = normalize_state_prompts(prompts, reuse_existing_state=had_existing_state)

    if req.stream:
        cancel_token = CancellationToken()

        async def stream_with_dialogue_idx():
            inner_stream = engine.batch_infer_stream_state(
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
                cancel_token=cancel_token,
            )
            stored = False
            try:
                async for chunk in inner_stream:
                    if chunk == "data: [DONE]\n\n" and not stored and not cancel_token.is_cancelled():
                        new_dialogue_idx = allocate_next_dialogue_idx(
                            app_state, state_manager, session_index
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
                await inner_stream.aclose()
                if not stored and not cancel_token.is_cancelled():
                    new_dialogue_idx = allocate_next_dialogue_idx(
                        app_state, state_manager, session_index
                    )
                    new_session_id = f"{session_index}:{new_dialogue_idx}"
                    state_manager.put_state(new_session_id, state)
                    print(
                        "[RESPONSE] /multi_state/chat/completions state[2]: ",
                        state[2],
                        "\n",
                    )

        return StreamingResponse(
            stream_with_prefill_queue(request, stream_with_dialogue_idx(), cancel_token, 1),
            media_type="text/event-stream",
        )

    try:
        async with reserve_prefill_capacity(request, 1):
            results = await run_sync_with_disconnect_watch(
                request,
                engine.batch_generate_state,
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
    except InferenceCancelled:
        del state
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        del state
        return prefill_bsz_limit_response(exc)

    new_dialogue_idx = allocate_next_dialogue_idx(app_state, state_manager, session_index)
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
    return response


@router.post("/state/status")
async def state_status(request: Request):
    password = request.app.state.password
    try:
        body = await request.json() if (await request.body()) else {}
        auth_error = check_password(body.get("password"), password)
        if auth_error is not None:
            return auth_error

        manager = get_state_manager()
        all_states = manager.list_all_states()

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

        return response_data
    except Exception as exc:
        print(f"[ERROR] /state/status: {exc}")
        return json_response(500, {"error": str(exc)})


@router.post("/state/delete")
async def state_delete(request: Request):
    password = request.app.state.password
    try:
        body = await request.json()
        session_id = body.get("session_id")
        delete_prefix = body.get("delete_prefix", False)

        if not session_id:
            return json_response(400, {"error": "Missing session_id parameter"})

        auth_error = check_password(body.get("password"), password)
        if auth_error is not None:
            return auth_error

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

        print(
            f"[StatePool] Delete session {session_id}: {'Success' if success else 'Not Found'}"
        )

        return json_response(status_code, response_data)
    except Exception as exc:
        print(f"[ERROR] /state/delete: {exc}")
        return json_response(500, {"error": str(exc)})
