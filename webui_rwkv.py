import json
import time
from datetime import datetime
from typing import Any

import gradio as gr
import requests


DEFAULT_API_URL = "http://127.0.0.1:8000/state/chat/completions"
DEFAULT_DELETE_URL = "http://127.0.0.1:8000/state/delete"
DEFAULT_BATCH_API_URL = "http://127.0.0.1:8000/v2/chat/completions"


def build_prompt(user_text: str) -> str:
    return f"User: {user_text}\n\nAssistant: <think>\n</think>\n"


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
        stop_tokens = json.loads(stop_tokens_text)
        if not isinstance(stop_tokens, list):
            raise ValueError
    except Exception:
        yield history, "stop_tokens 格式错误，请使用 JSON 数组，例如 [0,261,24281]"
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
    yield tuple(["准备开始...", "速度: 0.00 tok/s"] + slots + [None])

    try:
        stop_tokens = json.loads(stop_tokens_text)
        if not isinstance(stop_tokens, list):
            raise ValueError
    except Exception:
        slots = ["stop_tokens 格式错误"] * 32
        yield tuple(["stop_tokens 格式错误，请用 JSON 数组", "速度: 0.00 tok/s"] + slots + [None])
        return

    try:
        questions = read_questions_from_txt(question_file)
    except Exception as exc:
        slots = [f"读取文件失败: {exc}"] + [""] * 31
        yield tuple([f"读取文件失败: {exc}", "速度: 0.00 tok/s"] + slots + [None])
        return

    if not questions:
        slots = ["问题文件为空"] + [""] * 31
        yield tuple(["问题文件为空，未执行", "速度: 0.00 tok/s"] + slots + [None])
        return

    effective_bsz = max(1, min(32, int(batch_size)))
    total = len(questions)
    all_answers = [""] * total
    delta_events_total = 0
    start_time = time.time()
    status = f"已读取 {total} 条问题，开始按 batch_size={effective_bsz} 处理..."
    yield tuple([status, "速度: 0.00 tok/s"] + slots + [None])

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
                    yield tuple([f"第 {batch_start // effective_bsz + 1} 批失败: {err}", "速度: 0.00 tok/s"] + slots + [None])
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
                        speed = f"实时速度: {tps:.2f} tok/s | 估算tokens: {est_tokens} | delta事件: {delta_events_total}"
                        yield tuple([status, speed] + slots + [None])

        except requests.RequestException as exc:
            slots[0] = f"请求异常: {exc}"
            yield tuple([f"第 {batch_start // effective_bsz + 1} 批网络异常: {exc}", "速度: 0.00 tok/s"] + slots + [None])
            continue

    out_path = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (q, a) in enumerate(zip(questions, all_answers), start=1):
            f.write(f"【{i}】{q}\n")
            f.write(f"{a.strip()}\n")
            f.write("=" * 80 + "\n")

    elapsed = max(time.time() - start_time, 1e-6)
    est_tokens = delta_events_total * int(chunk_size)
    tps = est_tokens / elapsed
    final_status = (
        f"完成 | 总问题数: {total} | delta事件: {delta_events_total} | "
        f"估算tokens: {est_tokens} | 平均速度: {tps:.2f} tok/s"
    )
    final_speed = f"平均速度: {tps:.2f} tok/s | 估算tokens: {est_tokens} | delta事件: {delta_events_total}"
    yield tuple([final_status, final_speed] + slots + [out_path])

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
                    stop_tokens_text = gr.Textbox(label="stop_tokens(JSON)", value="[0, 261, 24281]")

            with gr.Column(scale=2, min_width=520):
                chatbot = gr.Chatbot(label="Chat", height=560)
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
        gr.Markdown("## 读取txt文本问题并发流式处理（默认 batch_size=32）")
        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                batch_api_url = gr.Textbox(label="Batch API URL", value=DEFAULT_BATCH_API_URL)
                batch_file = gr.File(label="上传问题txt文件", file_types=[".txt"])
                batch_password = gr.Textbox(label="Password", value="rwkv7_7.2b", type="password")
                batch_size = gr.Slider(1, 32, value=32, step=1, label="batch_size")
                batch_max_tokens = gr.Slider(1, 4096, value=1024, step=1, label="max_tokens")
                batch_chunk_size = gr.Slider(1, 128, value=8, step=1, label="chunk_size")
                batch_temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="temperature")
                batch_top_k = gr.Slider(1, 200, value=1, step=1, label="top_k")
                batch_top_p = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="top_p")
                batch_alpha_presence = gr.Slider(0.0, 2.0, value=1, step=0.01, label="alpha_presence")
                batch_alpha_frequency = gr.Slider(0.0, 2.0, value=0.2, step=0.01, label="alpha_frequency")
                batch_alpha_decay = gr.Slider(0.0, 1.0, value=0.996, step=0.001, label="alpha_decay")
                batch_stop_tokens = gr.Textbox(label="stop_tokens(JSON)", value="[0, 261, 24281]")
                run_batch_btn = gr.Button("开始并发处理", variant="primary")
                batch_status = gr.Markdown("待命")
                batch_speed = gr.Markdown("速度: 0.00 tok/s")
                batch_download = gr.File(label="导出结果 TXT", interactive=False)
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
                            )
                            batch_boxes.append(box)
        run_batch_btn.click(
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
                outputs=[batch_status, batch_speed] + batch_boxes + [batch_download],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
