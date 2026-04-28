import ast
import inspect
import json
import time
from datetime import datetime
from typing import Any, Optional

import gradio as gr
import requests


DEFAULT_API_URL = "http://127.0.0.1:8000/state/chat/completions"
DEFAULT_DELETE_URL = "http://127.0.0.1:8000/state/delete"
DEFAULT_BATCH_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_STOP_TOKENS_JSON = json.dumps(["\nUser:"], ensure_ascii=False)


def copy_button_kwargs(component_cls: Any) -> dict[str, Any]:
    try:
        params = inspect.signature(component_cls).parameters
    except (TypeError, ValueError):
        return {}
    if "buttons" in params:
        return {"buttons": ["copy"]}
    if "show_copy_button" in params:
        return {"show_copy_button": True}
    return {}


def build_prompt(user_text: str, think_mode: bool = False) -> str:
    think_suffix = "<think" if think_mode else "<think>\n</think>\n"
    return f"User: {user_text}\n\nAssistant: {think_suffix}"


def toggle_think_mode(enabled: bool) -> tuple[bool, dict[str, Any]]:
    next_enabled = not enabled
    return next_enabled, gr.update(
        value=f"思考模式：{'开' if next_enabled else '关'}",
        variant="primary" if next_enabled else "secondary",
    )


def parse_stop_tokens(stop_tokens_input: Any) -> list[str]:
    if isinstance(stop_tokens_input, list):
        stop_tokens = stop_tokens_input
    else:
        text = str(stop_tokens_input or "").strip()
        if not text:
            return []
        try:
            stop_tokens = json.loads(text)
        except json.JSONDecodeError:
            stop_tokens = ast.literal_eval(text)

    if not isinstance(stop_tokens, list):
        raise ValueError("stop_tokens must be a list")
    return [str(token) for token in stop_tokens]


def stream_chat(
    user_input: str,
    history: list[dict[str, str]],
    api_url: str,
    password: str,
    session_id: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    alpha_presence: float,
    alpha_frequency: float,
    alpha_decay: float,
    chunk_size: int,
    stop_tokens_text: str,
) -> Any:
    history = history or []
    user_input = (user_input or "").strip()
    if not user_input:
        yield history, "请输入内容后再发送。"
        return
    try:
        stop_tokens = parse_stop_tokens(stop_tokens_text)
    except Exception:
        yield history, 'stop_tokens 格式错误，请使用 JSON 数组，例如 ["\\nUser:"]'
        return

    payload = {
        "contents": [build_prompt(user_input)],
        "max_tokens": int(max_tokens),
        "stop_tokens": stop_tokens,
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "alpha_presence": float(alpha_presence),
        "alpha_frequency": float(alpha_frequency),
        "alpha_decay": float(alpha_decay),
        "stream": True,
        "chunk_size": int(chunk_size),
        "password": password,
        "session_id": session_id,
    }

    history = list(history)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": ""})
    yield history, "正在连接后端..."

    start = time.time()
    chunk_count = 0
    est_tokens = 0
    assistant_text = ""

    try:
        with requests.post(
            api_url,
            json=payload,
            headers={"Accept": "*/*", "Content-Type": "application/json"},
            stream=True,
            timeout=300,
        ) as resp:
            resp.encoding = "utf-8"
            if resp.status_code != 200:
                detail = resp.text[:400]
                history[-1]["content"] = f"[请求失败] HTTP {resp.status_code}\n{detail}"
                yield history, "请求失败"
                return

            for raw_line in resp.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {}).get("content", "")
                if not delta:
                    continue

                assistant_text += delta
                history[-1]["content"] = assistant_text

                chunk_count += 1
                est_tokens = chunk_count * int(chunk_size)
                elapsed = max(time.time() - start, 1e-6)
                tps = est_tokens / elapsed
                status = (
                    f"流式生成中 | chunk: {chunk_count} | 估算tokens: {est_tokens} "
                    f"| 速度: {tps:.2f} tok/s | 耗时: {elapsed:.2f}s"
                )
                yield history, status

        elapsed = max(time.time() - start, 1e-6)
        tps = est_tokens / elapsed
        yield history, f"完成 | chunk: {chunk_count} | 估算tokens: {est_tokens} | 平均速度: {tps:.2f} tok/s"
    except requests.RequestException as exc:
        history[-1]["content"] = f"[网络错误] {exc}"
        yield history, "网络错误"


def clear_session_state(delete_url: str, password: str, session_id: str) -> str:
    payload = {"session_id": session_id, "password": password}
    try:
        resp = requests.post(delete_url, json=payload, timeout=20)
        if resp.status_code == 200:
            return f"已清空后端会话状态: {session_id}"
        return f"清空失败 HTTP {resp.status_code}: {resp.text[:300]}"
    except requests.RequestException as exc:
        return f"清空失败: {exc}"


def clear_chat_only() -> tuple[list[dict[str, str]], str]:
    return [], "聊天窗口已清空（后端状态未删除）"


def read_questions_from_txt(file_path: str) -> list[str]:
    path = (file_path or "").strip() or "./问题.txt"
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk") as f:
            lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def render_batch_slot(global_idx: int, question: str, answer: str) -> str:
    answer_show = answer if answer else "..."
    return f"[{global_idx}] Q: {question}\n\nA: {answer_show}"


def save_generation_results(result_state: Optional[dict[str, Any]]) -> tuple[Optional[str], str]:
    if not result_state:
        return None, "暂无可保存结果"

    mode = result_state.get("mode")
    out_path = f"{mode or 'generation'}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        if mode == "batch":
            questions = result_state.get("questions", [])
            answers = result_state.get("answers", [])
            for i, (q, a) in enumerate(zip(questions, answers), start=1):
                f.write(f"【{i}】{q}\n")
                f.write(f"{str(a).strip()}\n")
                f.write("=" * 80 + "\n")
        elif mode == "prompt32":
            prompt = result_state.get("prompt", "")
            answers = result_state.get("answers", [])
            f.write(f"Prompt: {prompt}\n")
            f.write("=" * 80 + "\n")
            for i, answer in enumerate(answers, start=1):
                f.write(f"【{i}】\n")
                f.write(f"{str(answer).strip()}\n")
                f.write("=" * 80 + "\n")
        else:
            f.write(json.dumps(result_state, ensure_ascii=False, indent=2))
            f.write("\n")

    return out_path, f"已保存: {out_path}"


def run_batch_file_stream(
    question_file: str,
    api_url: str,
    password: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    alpha_presence: float,
    alpha_frequency: float,
    alpha_decay: float,
    chunk_size: int,
    stop_tokens_text: str,
    batch_size: int,
) -> Any:
    slots = ["等待填充..."] * 32
    yield tuple(["准备开始..."] + slots + [None, None])

    try:
        stop_tokens = parse_stop_tokens(stop_tokens_text)
    except Exception:
        slots = ["stop_tokens 格式错误"] * 32
        yield tuple(['stop_tokens 格式错误，请用 JSON 数组，例如 ["\\\\nUser:"]'] + slots + [None, None])
        return

    try:
        questions = read_questions_from_txt(question_file)
    except Exception as exc:
        slots = [f"读取文件失败: {exc}"] + [""] * 31
        yield tuple([f"读取文件失败: {exc}"] + slots + [None, None])
        return

    if not questions:
        slots = ["问题文件为空"] + [""] * 31
        yield tuple(["问题文件为空，未执行"] + slots + [None, None])
        return

    effective_bsz = max(1, min(32, int(batch_size)))
    total = len(questions)
    all_answers = [""] * total
    delta_events_total = 0
    start_time = time.time()
    status = f"已读取 {total} 条问题，开始按 batch_size={effective_bsz} 处理..."
    yield tuple([status] + slots + [None, None])

    for batch_start in range(0, total, effective_bsz):
        batch_questions = questions[batch_start : batch_start + effective_bsz]
        active = len(batch_questions)
        batch_answers = [""] * active
        slots = ["(空)"] * 32
        for i in range(active):
            slots[i] = render_batch_slot(batch_start + i + 1, batch_questions[i], "")

        payload = {
            "contents": [build_prompt(q) for q in batch_questions],
            "max_tokens": int(max_tokens),
            "stop_tokens": stop_tokens,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "pad_zero": True,
            "alpha_presence": float(alpha_presence),
            "alpha_frequency": float(alpha_frequency),
            "alpha_decay": float(alpha_decay),
            "chunk_size": int(chunk_size),
            "stream": True,
            "password": password,
        }

        try:
            with requests.post(
                api_url,
                json=payload,
                headers={"Accept": "*/*", "Content-Type": "application/json"},
                stream=True,
                timeout=600,
            ) as resp:
                resp.encoding = "utf-8"
                if resp.status_code != 200:
                    err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    slots[0] = err
                    yield tuple([f"第 {batch_start // effective_bsz + 1} 批失败: {err}"] + slots + [None, None])
                    continue

                for raw_line in resp.iter_lines(decode_unicode=False):
                    if not raw_line:
                        continue
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError:
                        line = raw_line.decode("utf-8", errors="replace")

                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    updated = False
                    for choice in choices:
                        idx = choice.get("index", 0)
                        delta = choice.get("delta", {}).get("content", "")
                        if isinstance(idx, int) and 0 <= idx < active and delta:
                            batch_answers[idx] += delta
                            all_answers[batch_start + idx] = batch_answers[idx]
                            slots[idx] = render_batch_slot(batch_start + idx + 1, batch_questions[idx], batch_answers[idx])
                            delta_events_total += 1
                            updated = True

                    if updated:
                        elapsed = max(time.time() - start_time, 1e-6)
                        est_tokens = delta_events_total * int(chunk_size)
                        tps = est_tokens / elapsed
                        done_count = min(batch_start + active, total)
                        status = (
                            f"处理中 | {done_count}/{total} 条问题所在批次流式中 | "
                            f"delta事件: {delta_events_total} | 估算tokens: {est_tokens} | 速度: {tps:.2f} tok/s"
                        )
                        yield tuple([status] + slots + [None, None])

        except requests.RequestException as exc:
            slots[0] = f"请求异常: {exc}"
            yield tuple([f"第 {batch_start // effective_bsz + 1} 批网络异常: {exc}"] + slots + [None, None])
            continue

    elapsed = max(time.time() - start_time, 1e-6)
    est_tokens = delta_events_total * int(chunk_size)
    tps = est_tokens / elapsed
    final_status = (
        f"完成 | 总问题数: {total} | delta事件: {delta_events_total} | "
        f"估算tokens: {est_tokens} | 平均速度: {tps:.2f} tok/s"
    )
    result_state = {"mode": "batch", "questions": questions, "answers": all_answers}
    yield tuple([final_status] + slots + [result_state, None])


def run_prompt_32_stream(
    prompt_text: str,
    think_mode: bool,
    api_url: str,
    password: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    alpha_presence: float,
    alpha_frequency: float,
    alpha_decay: float,
    chunk_size: int,
    stop_tokens_text: str,
) -> Any:
    active = 32
    prompt_text = (prompt_text or "").strip()
    slots = ["等待填充..."] * active
    yield tuple(["准备开始..."] + slots + [None, None])

    if not prompt_text:
        slots = ["请输入 prompt"] + [""] * (active - 1)
        yield tuple(["请输入 prompt 后再生成"] + slots + [None, None])
        return
    try:
        stop_tokens = parse_stop_tokens(stop_tokens_text)
    except Exception:
        slots = ["stop_tokens 格式错误"] * active
        yield tuple(['stop_tokens 格式错误，请使用 JSON 数组，例如 ["\\\\nUser:"]'] + slots + [None, None])
        return

    answers = [""] * active
    slots = ["..."] * active
    delta_events_total = 0
    start_time = time.time()

    payload = {
        "contents": [build_prompt(prompt_text, think_mode=think_mode)] * active,
        "max_tokens": int(max_tokens),
        "stop_tokens": stop_tokens,
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "pad_zero": True,
        "alpha_presence": float(alpha_presence),
        "alpha_frequency": float(alpha_frequency),
        "alpha_decay": float(alpha_decay),
        "chunk_size": int(chunk_size),
        "stream": True,
        "password": password,
    }

    try:
        with requests.post(
            api_url,
            json=payload,
            headers={"Accept": "*/*", "Content-Type": "application/json"},
            stream=True,
            timeout=600,
        ) as resp:
            resp.encoding = "utf-8"
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                slots[0] = err
                yield tuple([f"请求失败: {err}"] + slots + [None, None])
                return

            for raw_line in resp.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw_line.decode("utf-8", errors="replace")

                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                updated = False
                for choice in choices:
                    idx = choice.get("index", 0)
                    delta = choice.get("delta", {}).get("content", "")
                    if isinstance(idx, int) and 0 <= idx < active and delta:
                        answers[idx] += delta
                        slots[idx] = answers[idx]
                        delta_events_total += 1
                        updated = True

                if updated:
                    elapsed = max(time.time() - start_time, 1e-6)
                    est_tokens = delta_events_total * int(chunk_size)
                    tps = est_tokens / elapsed
                    status = (
                        f"32路生成中 | delta事件: {delta_events_total} | "
                        f"估算tokens: {est_tokens} | 速度: {tps:.2f} tok/s"
                    )
                    yield tuple([status] + slots + [None, None])

    except requests.RequestException as exc:
        slots[0] = f"请求异常: {exc}"
        yield tuple([f"网络异常: {exc}"] + slots + [None, None])
        return

    elapsed = max(time.time() - start_time, 1e-6)
    est_tokens = delta_events_total * int(chunk_size)
    tps = est_tokens / elapsed
    final_status = (
        f"完成 | 生成回复数: {active} | delta事件: {delta_events_total} | "
        f"估算tokens: {est_tokens} | 平均速度: {tps:.2f} tok/s"
    )
    result_state = {"mode": "prompt32", "prompt": prompt_text, "answers": answers}
    yield tuple([final_status] + slots + [result_state, None])


with gr.Blocks(title="RWKV State Chat UI") as demo:
    with gr.Tab("💬 单轮对话"):
        gr.Markdown("## RWKV `/state/chat/completions` WebUI")

        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                with gr.Row():
                    api_url = gr.Textbox(label="API URL", value=DEFAULT_API_URL)
                    delete_url = gr.Textbox(label="Delete URL", value=DEFAULT_DELETE_URL)
                with gr.Row():
                    password = gr.Textbox(label="Password", value="rwkv7_7.2b", type="password")
                    session_id = gr.Textbox(label="Session ID", value="session_one")

                with gr.Row():
                    max_tokens = gr.Slider(1, 4096, value=1024, step=1, label="max_tokens")
                    chunk_size = gr.Slider(1, 128, value=8, step=1, label="chunk_size")
                    temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.01, label="temperature")
                with gr.Row():
                    top_k = gr.Slider(1, 200, value=50, step=1, label="top_k")
                    top_p = gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="top_p")
                    alpha_presence = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="alpha_presence")
                with gr.Row():
                    alpha_frequency = gr.Slider(0.0, 2.0, value=0.1, step=0.01, label="alpha_frequency")
                    alpha_decay = gr.Slider(0.0, 1.0, value=0.99, step=0.001, label="alpha_decay")
                    stop_tokens_text = gr.Textbox(label="stop_tokens(JSON)", value=DEFAULT_STOP_TOKENS_JSON)

            with gr.Column(scale=2, min_width=520):
                chatbot = gr.Chatbot(label="Chat", height=560, **copy_button_kwargs(gr.Chatbot))
                status = gr.Markdown("待命")
                user_input = gr.Textbox(label="输入问题", placeholder="例如：今晚吃什么？")
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_chat_btn = gr.Button("清空聊天")
                    clear_session_btn = gr.Button("清空后端会话状态")

            send_btn.click(
                stream_chat,
                inputs=[
                    user_input,
                    chatbot,
                    api_url,
                    password,
                    session_id,
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    chunk_size,
                    stop_tokens_text,
                ],
                outputs=[chatbot, status],
            ).then(lambda: "", outputs=[user_input])

            user_input.submit(
                stream_chat,
                inputs=[
                    user_input,
                    chatbot,
                    api_url,
                    password,
                    session_id,
                    max_tokens,
                    temperature,
                    top_k,
                    top_p,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    chunk_size,
                    stop_tokens_text,
                ],
                outputs=[chatbot, status],
            ).then(lambda: "", outputs=[user_input])

            clear_chat_btn.click(clear_chat_only, outputs=[chatbot, status])
            clear_session_btn.click(
                clear_session_state,
                inputs=[delete_url, password, session_id],
                outputs=[status],
            )
    with gr.Tab("并发处理文本"):
        gr.Markdown("## 读取txt文本的问题（每行是一个prompt），然后并发流式处理（默认 batch_size=32）")
        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                batch_api_url = gr.Textbox(label="Batch API URL", value=DEFAULT_BATCH_API_URL)
                batch_file = gr.File(label="上传问题txt文件", file_types=[".txt"])
                batch_password = gr.Textbox(label="Password", placeholder="输入 password",value="", type="password")
                batch_size = gr.Slider(1, 32, value=32, step=1, label="batch_size")
                batch_max_tokens = gr.Slider(1, 4096, value=1024, step=1, label="max_tokens")
                batch_chunk_size = gr.Slider(1, 128, value=8, step=1, label="chunk_size")
                batch_temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="temperature")
                batch_top_k = gr.Slider(1, 200, value=1, step=1, label="top_k")
                batch_top_p = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="top_p")
                batch_alpha_presence = gr.Slider(0.0, 2.0, value=1, step=0.01, label="alpha_presence")
                batch_alpha_frequency = gr.Slider(0.0, 2.0, value=0.2, step=0.01, label="alpha_frequency")
                batch_alpha_decay = gr.Slider(0.0, 1.0, value=0.996, step=0.001, label="alpha_decay")
                batch_stop_tokens = gr.Textbox(label="stop_tokens(JSON)", value=DEFAULT_STOP_TOKENS_JSON)
                with gr.Row():
                    run_batch_btn = gr.Button("开始并发处理", variant="primary")
                    stop_batch_btn = gr.Button("停止", variant="stop")
                batch_status = gr.Markdown("待命")
                save_batch_btn = gr.Button("保存结果 TXT")
                batch_download = gr.File(label="结果文件", interactive=False)
                batch_result_state = gr.State(None)
            with gr.Column(scale=2, min_width=700):
                gr.Markdown("### 实时输出槽位（最多 32 路）")
                batch_boxes = []
                for r in range(8):
                    with gr.Row():
                        for c in range(4):
                            idx = r * 4 + c + 1
                            box = gr.Textbox(
                                label=f"#{idx}",
                                value="(空)",
                                lines=6,
                                interactive=False,
                                **copy_button_kwargs(gr.Textbox),
                            )
                            batch_boxes.append(box)
        batch_run_event = run_batch_btn.click(
            run_batch_file_stream,
            inputs=[
                batch_file,
                batch_api_url,
                batch_password,
                batch_max_tokens,
                batch_temperature,
                batch_top_k,
                batch_top_p,
                batch_alpha_presence,
                batch_alpha_frequency,
                batch_alpha_decay,
                batch_chunk_size,
                batch_stop_tokens,
                batch_size,
                ],
                outputs=[batch_status] + batch_boxes + [batch_result_state, batch_download],
            )
        stop_batch_btn.click(
            lambda: "已停止",
            outputs=[batch_status],
            cancels=[batch_run_event],
        )
        save_batch_btn.click(
            save_generation_results,
            inputs=[batch_result_state],
            outputs=[batch_download, batch_status],
        )

    with gr.Tab("同Prompt并发32路"):
        gr.Markdown("## 输入一行 prompt，并发生成 32 条回复")
        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                prompt32_api_url = gr.Textbox(label="Batch API URL", value=DEFAULT_BATCH_API_URL)
                prompt32_input = gr.Textbox(label="Prompt", placeholder="输入一行 prompt", lines=1)
                prompt32_password = gr.Textbox(label="Password", placeholder="输入password",value="", type="password")
                prompt32_max_tokens = gr.Slider(1, 8192, value=4096, step=1, label="max_tokens")
                prompt32_chunk_size = gr.Slider(1, 128, value=8, step=1, label="chunk_size")
                prompt32_temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="temperature")
                prompt32_top_k = gr.Slider(1, 200, value=50, step=1, label="top_k")
                prompt32_top_p = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="top_p")
                prompt32_alpha_presence = gr.Slider(0.0, 2.0, value=1, step=0.01, label="alpha_presence")
                prompt32_alpha_frequency = gr.Slider(0.0, 2.0, value=0.2, step=0.01, label="alpha_frequency")
                prompt32_alpha_decay = gr.Slider(0.0, 1.0, value=0.996, step=0.001, label="alpha_decay")
                prompt32_stop_tokens = gr.Textbox(label="stop_tokens(JSON)", value=DEFAULT_STOP_TOKENS_JSON)
                prompt32_think_mode = gr.State(False)
                with gr.Row():
                    run_prompt32_btn = gr.Button("开始生成32条", variant="primary")
                    stop_prompt32_btn = gr.Button("停止", variant="stop")
                    think_mode_btn = gr.Button("思考模式：关", variant="secondary")
                prompt32_status = gr.Markdown("待命")
                save_prompt32_btn = gr.Button("保存结果 TXT")
                prompt32_download = gr.File(label="结果文件", interactive=False)
                prompt32_result_state = gr.State(None)
            with gr.Column(scale=2, min_width=700):
                gr.Markdown("### 实时输出槽位（32 路）")
                prompt32_boxes = []
                for r in range(8):
                    with gr.Row():
                        for c in range(4):
                            idx = r * 4 + c + 1
                            box = gr.Textbox(
                                label=f"#{idx}",
                                value="(空)",
                                lines=6,
                                interactive=False,
                                **copy_button_kwargs(gr.Textbox),
                            )
                            prompt32_boxes.append(box)
        think_mode_btn.click(
            toggle_think_mode,
            inputs=[prompt32_think_mode],
            outputs=[prompt32_think_mode, think_mode_btn],
        )
        prompt32_run_event = run_prompt32_btn.click(
            run_prompt_32_stream,
            inputs=[
                prompt32_input,
                prompt32_think_mode,
                prompt32_api_url,
                prompt32_password,
                prompt32_max_tokens,
                prompt32_temperature,
                prompt32_top_k,
                prompt32_top_p,
                prompt32_alpha_presence,
                prompt32_alpha_frequency,
                prompt32_alpha_decay,
                prompt32_chunk_size,
                prompt32_stop_tokens,
            ],
            outputs=[prompt32_status] + prompt32_boxes + [prompt32_result_state, prompt32_download],
        )
        stop_prompt32_btn.click(
            lambda: "已停止",
            outputs=[prompt32_status],
            cancels=[prompt32_run_event],
        )
        save_prompt32_btn.click(
            save_generation_results,
            inputs=[prompt32_result_state],
            outputs=[prompt32_download, prompt32_status],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
