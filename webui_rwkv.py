import ast
import html
import inspect
import json
import re
import threading
import time
from datetime import datetime
from typing import Any, Optional

import gradio as gr
import requests


DEFAULT_API_URL = "http://127.0.0.1:8000/state/chat/completions"
DEFAULT_DELETE_URL = "http://127.0.0.1:8000/state/delete"
DEFAULT_BATCH_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_STOP_TOKENS_JSON = json.dumps(["\nUser:"], ensure_ascii=False)

HTML_GEN_LIMIT = 12000
MAX_HTML_PREVIEWS = 15
HTML_GRID_UPDATE_EVERY = 32
HTML_IFRAME_MIN_DELTA = 1800
HTML_IFRAME_MAX_STALE = 2.0
HTML_COMPONENT_MIN_INTERVAL = 0.5
HTML_PREVIEW_HEIGHT = 300
HTML_CAPTION_HEIGHT = 15
HTML_BODY_HEIGHT = HTML_PREVIEW_HEIGHT - HTML_CAPTION_HEIGHT
HTML_FRAME_HEIGHT = 228
HTML_RAW_HEIGHT = HTML_BODY_HEIGHT - HTML_FRAME_HEIGHT
HTML_PROMPT_CHOICES = [
    "3D animation of cars in forest with animals",
    "interactive weather map with animated clouds and rain",
    "retro arcade RPG start screen",
    "3D animation of a SpaceX rocket landing on Mars",
    "storybook scene with a dragon flying over a castle",
    "interactive dashboard for a city traffic system",
    "animated aquarium with colorful fish and coral",
    "sci-fi spaceship navigation interface",
    "cozy cafe menu with animated steam and pastries",
    "character sheet for a high fantasy RPG",
    "a fancy hotel homepage",
]
HTML_GRID_CSS = """
div.main { padding-left: 0 !important; padding-right: 0 !important; }
.html-grid-tab { padding-top: 0 !important; }
.html-grid-main { display: grid !important; grid-template-columns: minmax(220px, 17.5%) 1fr !important; grid-template-rows: auto auto !important; gap: 4px !important; margin-top: 0 !important; align-items: start !important; }
.html-grid-main > div { gap: 4px !important; }
.html-grid-controls { grid-column: 1 !important; grid-row: 1 !important; gap: 4px !important; min-width: 0 !important; }
.html-grid-pages { grid-column: 2 !important; grid-row: 1 / span 2 !important; gap: 4px !important; min-width: 0 !important; }
.html-grid-output { grid-column: 1 !important; grid-row: 2 !important; gap: 4px !important; min-width: 0 !important; }
.html-grid-preview-row { gap: 4px !important; margin: 0 !important; }
.html-grid-preview { margin: 0 !important; }
.html-grid-preview > div { margin: 0 !important; }
.html-container { padding: 0 !important; }
.html-prompt-choice { margin: 0 !important; padding: 0 !important; min-height: 0 !important; }
.html-prompt-choice .wrap,
.html-prompt-choice .wrap-inner,
.html-prompt-choice .secondary-wrap { margin: 0 !important; padding: 0 !important; min-height: 0 !important; }
.html-prompt-choice label { margin: 0 !important; padding: 0 !important; }
@media (max-width: 768px) {
  .html-grid-main { grid-template-columns: 1fr !important; grid-template-rows: auto auto auto !important; }
  .html-grid-controls { grid-column: 1 !important; grid-row: 1 !important; }
  .html-grid-pages { grid-column: 1 !important; grid-row: 2 !important; }
  .html-grid-output { grid-column: 1 !important; grid-row: 3 !important; }
}
"""

html_view_lock = threading.Lock()
html_view_scale = 35
html_view_scroll_seconds = 5


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


def html_prompt_from_choice(choice: str) -> str:
    return f"User: Write HTML: {choice}\n\nAssistant: <think></think"


DEFAULT_HTML_PROMPT = html_prompt_from_choice(HTML_PROMPT_CHOICES[0])


def clamp_html_scale(scale: int | float) -> int:
    return max(20, min(100, int(scale)))


def clamp_scroll_seconds(seconds: int | float) -> int:
    return max(0, min(10, int(seconds)))


def set_html_scale(scale: int | float) -> int:
    global html_view_scale
    scale = clamp_html_scale(scale)
    with html_view_lock:
        html_view_scale = scale
    return scale


def get_html_scale() -> int:
    with html_view_lock:
        return html_view_scale


def set_scroll_seconds(seconds: int | float) -> int:
    global html_view_scroll_seconds
    seconds = clamp_scroll_seconds(seconds)
    with html_view_lock:
        html_view_scroll_seconds = seconds
    return seconds


def get_scroll_seconds() -> int:
    with html_view_lock:
        return html_view_scroll_seconds


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


def split_batch_output(text: str, count: int) -> list[str]:
    parts = text.split("\n====\n") if text else []
    parts += [""] * max(0, count - len(parts))
    return parts[:count]


def join_batch_output(outputs: list[str]) -> str:
    return "\n====\n".join(x.strip() for x in outputs)


def extract_html(text: str, prompt: str = "") -> str:
    stream = prompt + text
    lower_text = stream.lower()
    marker = lower_text.find("</think>")
    if marker < 0:
        return ""
    visible = stream[marker + len("</think>") :]
    lower = visible.lower()
    start = lower.find("<!doctype html")
    if start < 0:
        return ""
    if lower.find("<body", start) < 0:
        return ""
    end = lower.find("</html>", start)
    if end >= 0:
        return visible[start : end + len("</html>")].strip()
    return visible[start:].strip()


def html_complete(page: str) -> bool:
    return "</html>" in page.lower()


def inject_iframe_scroll_script(page: str, index: int, scroll_seconds: int | float) -> str:
    scroll_seconds = clamp_scroll_seconds(scroll_seconds)
    if scroll_seconds <= 0:
        return page
    delay = int((index * 733) % 5000)
    leg_ms = scroll_seconds * 1000
    script = f"""<script>
(() => {{
  const delay = {delay};
  const legMs = {leg_ms};
  let down = true;
  let running = false;
  function unique(list) {{
    const out = [];
    const seen = new Set();
    for (const el of list) {{
      if (!el || seen.has(el)) continue;
      seen.add(el);
      out.push(el);
    }}
    return out;
  }}
  function candidates() {{
    const base = [document.scrollingElement, document.documentElement, document.body, document.body && document.body.parentElement];
    const all = Array.from(document.querySelectorAll("*"));
    return unique(base.concat(all))
      .filter(el => Math.max(0, el.scrollHeight - el.clientHeight) > 4)
      .sort((a, b) => (b.scrollHeight - b.clientHeight) - (a.scrollHeight - a.clientHeight))
      .slice(0, 8);
  }}
  function getTop(el) {{
    if (el === document.scrollingElement || el === document.documentElement || el === document.body) {{
      return window.scrollY || document.documentElement.scrollTop || document.body.scrollTop || 0;
    }}
    return el.scrollTop || 0;
  }}
  function setTop(el, y) {{
    if (el === document.scrollingElement || el === document.documentElement || el === document.body) {{
      window.scrollTo(0, y);
      document.documentElement.scrollTop = y;
      document.body.scrollTop = y;
    }} else {{
      el.scrollTop = y;
    }}
  }}
  function animate() {{
    const targets = candidates();
    if (!targets.length) {{
      setTimeout(animate, 1000);
      return;
    }}
    const starts = targets.map(getTop);
    const ends = targets.map(el => down ? Math.max(0, el.scrollHeight - el.clientHeight) : 0);
    down = !down;
    const t0 = performance.now();
    function step(t) {{
      const p = Math.min(1, (t - t0) / legMs);
      for (let i = 0; i < targets.length; i++) {{
        setTop(targets[i], starts[i] + (ends[i] - starts[i]) * p);
      }}
      requestAnimationFrame(p < 1 ? step : animate);
    }}
    requestAnimationFrame(step);
  }}
  function start() {{
    if (running) return;
    running = true;
    setTimeout(animate, delay);
    try {{
      new MutationObserver(() => candidates()).observe(document.documentElement, {{childList: true, subtree: true}});
    }} catch (e) {{}}
    setInterval(candidates, 1000);
  }}
  if (document.readyState === "complete") start();
  else window.addEventListener("load", start, {{once: true}});
  setTimeout(start, delay + 1500);
}})();
</script>"""
    lower = page.lower()
    m = re.search(r"<head\b[^>]*>", lower)
    if m:
        return page[: m.end()] + script + page[m.end() :]
    m = re.search(r"<html\b[^>]*>", lower)
    if m:
        return page[: m.end()] + "<head>" + script + "</head>" + page[m.end() :]
    m = re.search(r"<!doctype\s+html[^>]*>", lower)
    if m:
        return page[: m.end()] + script + page[m.end() :]
    return script + page


def render_preview(
    text: str = "",
    index: int = 0,
    scale: int | float = 35,
    active: bool = True,
    prompt: str = "",
    token_count: Optional[int] = None,
    scroll_seconds: int | float = 5,
) -> str:
    tokens = f"{token_count:,}" if token_count is not None else "-"
    caption = f"#{index + 1} | ~{tokens} tokens, {len(text.encode('utf-8')):,} bytes" if active else ""
    opacity = "1" if active else ".35"
    zoom = max(0.2, min(1.2, scale / 100.0))
    page = extract_html(text, prompt)
    if not text:
        body = f'<div style="height:{HTML_BODY_HEIGHT}px;background:#fafafa;"></div>'
    elif page:
        srcdoc = html.escape(inject_iframe_scroll_script(page, index, scroll_seconds), quote=True)
        raw = html.escape(text)
        body = f"""<div style="height:{HTML_BODY_HEIGHT}px;display:flex;flex-direction:column;background:white;">
<div style="height:{HTML_FRAME_HEIGHT}px;overflow:hidden;background:white;"><iframe sandbox="allow-scripts allow-forms allow-popups" srcdoc="{srcdoc}" style="border:0;width:{100 / zoom:.3f}%;height:{HTML_FRAME_HEIGHT / zoom:.1f}px;background:white;transform:scale({zoom:.3f});transform-origin:top left;"></iframe></div>
<div id="html-raw-{index}" style="height:{HTML_RAW_HEIGHT}px;overflow:auto;display:flex;flex-direction:column-reverse;background:#fafafa;border-top:1px solid #111;"><pre style="zoom:{zoom:.3f};margin:0;padding:0;white-space:pre-wrap;word-break:break-word;color:#111;font:16px/1.2 ui-monospace,Consolas,monospace;">{raw}</pre></div>
</div>"""
    else:
        body = f'<div id="html-raw-{index}" style="height:{HTML_BODY_HEIGHT}px;overflow:auto;display:flex;flex-direction:column-reverse;background:#fafafa;"><pre style="zoom:{zoom:.3f};margin:0;padding:0;white-space:pre-wrap;word-break:break-word;color:#111;font:16px/1.2 ui-monospace,Consolas,monospace;">{html.escape(text)}</pre></div>'
    return f"""<div class="html-container" style="outline:1px solid #111;background:#fff;opacity:{opacity};height:{HTML_PREVIEW_HEIGHT}px;display:flex;flex-direction:column;padding:0;">
<div style="box-sizing:border-box;height:{HTML_CAPTION_HEIGHT}px;padding:1px 6px;background:#111;color:#fff;font:11px/13px ui-monospace,monospace;">{caption}</div>
{body}
</div>"""


def empty_html_grid() -> list[str]:
    return [render_preview("", i, active=False) for i in range(MAX_HTML_PREVIEWS)]


def render_html_grid_from_raw(
    prompt: str,
    raw_output: str,
    page_count: int,
    scale: int | float,
    scroll_seconds: int | float,
    token_counts: list[int],
) -> list[str]:
    page_count = max(1, min(MAX_HTML_PREVIEWS, int(page_count)))
    scale = set_html_scale(scale)
    scroll_seconds = set_scroll_seconds(scroll_seconds)
    pages = split_batch_output(raw_output, page_count)
    token_counts = token_counts or []
    return [
        render_preview(
            pages[i] if i < page_count else "",
            i,
            scale,
            i < page_count,
            prompt,
            token_counts[i] if i < len(token_counts) else None,
            scroll_seconds,
        )
        for i in range(MAX_HTML_PREVIEWS)
    ]


def clear_html_grid(page_count: int) -> tuple[Any, ...]:
    page_count = max(1, min(MAX_HTML_PREVIEWS, int(page_count)))
    return tuple(
        [
            *empty_html_grid(),
            gr.update(value="", label="Output"),
            [0 for _ in range(MAX_HTML_PREVIEWS)],
            page_count,
            "已清空",
        ]
    )


def stream_html_grid_api(
    prompt: str,
    api_url: str,
    password: str,
    token_count: int,
    page_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
    alpha_presence: float,
    alpha_frequency: float,
    alpha_decay: float,
    chunk_size: int,
    stop_tokens_text: str,
) -> Any:
    page_count = max(1, min(MAX_HTML_PREVIEWS, int(page_count)))
    prompt = (prompt or "").strip()
    if not prompt:
        yield "", "请输入 prompt", {"done": True, "token_counts": [0 for _ in range(page_count)]}
        return

    try:
        stop_tokens = parse_stop_tokens(stop_tokens_text)
    except Exception:
        yield "", 'stop_tokens 格式错误，请使用 JSON 数组，例如 ["\\nUser:"]', {
            "done": True,
            "token_counts": [0 for _ in range(page_count)],
        }
        return

    answers = [""] * page_count
    token_counts = [0] * page_count
    delta_events_total = 0
    start_time = time.time()

    payload = {
        "contents": [prompt] * page_count,
        "max_tokens": int(token_count),
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

    resp = None
    try:
        resp = requests.post(
            api_url,
            json=payload,
            headers={"Accept": "*/*", "Content-Type": "application/json", "Connection": "close"},
            stream=True,
            timeout=900,
        )
        resp.encoding = "utf-8"
        if resp.status_code != 200:
            detail = resp.text[:400]
            yield "", f"请求失败 HTTP {resp.status_code}: {detail}", {
                "done": True,
                "token_counts": token_counts,
            }
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
                if isinstance(idx, int) and 0 <= idx < page_count and delta:
                    answers[idx] += delta
                    token_counts[idx] += int(chunk_size)
                    delta_events_total += 1
                    updated = True

            if updated and delta_events_total % max(1, HTML_GRID_UPDATE_EVERY) == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                est_tokens = sum(token_counts)
                speed = est_tokens / elapsed
                text = join_batch_output(answers)
                yield text, (
                    f"{speed:.2f} tok/s | batch={page_count} | "
                    f"估算tokens={est_tokens} | 耗时={elapsed:.2f}s"
                ), {"done": False, "token_counts": token_counts[:]}

    except requests.RequestException as exc:
        yield join_batch_output(answers), f"网络异常: {exc}", {
            "done": True,
            "token_counts": token_counts,
        }
        return
    finally:
        if resp is not None:
            resp.close()

    elapsed = max(time.time() - start_time, 1e-6)
    est_tokens = sum(token_counts)
    speed = est_tokens / elapsed
    yield join_batch_output(answers), (
        f"完成 | batch={page_count} | 估算tokens={est_tokens} | 平均速度={speed:.2f} tok/s"
    ), {"done": True, "token_counts": token_counts}


def evaluate_html_grid_api(
    prompt: str,
    api_url: str,
    password: str,
    token_count: int = HTML_GEN_LIMIT,
    page_count: int = MAX_HTML_PREVIEWS,
    scale: int | float = 35,
    scroll_seconds: int | float = 5,
    temperature: float = 1.0,
    top_k: int = 500,
    top_p: float = 0.5,
    alpha_presence: float = 1.0,
    alpha_frequency: float = 0.1,
    alpha_decay: float = 0.99,
    chunk_size: int = 8,
    stop_tokens_text: str = DEFAULT_STOP_TOKENS_JSON,
) -> Any:
    page_count = max(1, min(MAX_HTML_PREVIEWS, int(page_count)))
    scale = set_html_scale(scale)
    scroll_seconds = set_scroll_seconds(scroll_seconds)
    last_text = ""
    cached_previews = empty_html_grid()
    cached_html = ["" for _ in range(MAX_HTML_PREVIEWS)]
    cached_html_at = [0.0 for _ in range(MAX_HTML_PREVIEWS)]
    cached_complete = [False for _ in range(MAX_HTML_PREVIEWS)]
    cached_text_len = [0 for _ in range(MAX_HTML_PREVIEWS)]
    cached_preview_at = [0.0 for _ in range(MAX_HTML_PREVIEWS)]
    last_raw_at = 0.0
    final_text = ""
    final_status = "正在连接后端..."
    final_token_counts = [0 for _ in range(MAX_HTML_PREVIEWS)]
    yield tuple([*cached_previews, gr.update(value="", label="Output"), final_token_counts, page_count, final_status])

    for text, speed_info, meta in stream_html_grid_api(
        prompt,
        api_url,
        password,
        token_count,
        page_count,
        temperature,
        top_k,
        top_p,
        alpha_presence,
        alpha_frequency,
        alpha_decay,
        chunk_size,
        stop_tokens_text,
    ):
        final_text = text
        final_status = speed_info
        done_batch = bool(meta and meta.get("done"))
        token_counts = meta.get("token_counts", final_token_counts) if meta else final_token_counts
        if token_counts:
            final_token_counts = token_counts + [0 for _ in range(max(0, MAX_HTML_PREVIEWS - len(token_counts)))]
        if len(text) - len(last_text) < 300 and not done_batch:
            continue
        last_text = text
        now = time.monotonic()
        render_scale = get_html_scale()
        render_scroll_seconds = get_scroll_seconds()
        scale_changed = render_scale != scale
        scroll_changed = render_scroll_seconds != scroll_seconds
        scale = render_scale
        scroll_seconds = render_scroll_seconds
        pages = split_batch_output(text, page_count)
        skip = gr.skip()
        updates = [skip for _ in range(MAX_HTML_PREVIEWS)]
        for i in range(MAX_HTML_PREVIEWS):
            page_text = pages[i] if i < page_count else ""
            active = i < page_count
            page = extract_html(page_text, prompt) if active else ""
            page_tokens = final_token_counts[i] if i < len(final_token_counts) else None
            if now - cached_preview_at[i] < HTML_COMPONENT_MIN_INTERVAL and not done_batch and not scale_changed and not scroll_changed:
                continue
            if not page:
                if len(page_text) == cached_text_len[i] and active == bool(cached_text_len[i]) and not done_batch and not scale_changed and not scroll_changed:
                    continue
                cached_previews[i] = render_preview(page_text, i, scale, active, prompt, page_tokens, scroll_seconds)
                cached_html[i] = ""
                cached_html_at[i] = now
                cached_complete[i] = False
                cached_text_len[i] = len(page_text)
                cached_preview_at[i] = now
                updates[i] = cached_previews[i]
                continue
            done = html_complete(page)
            should_reload = (
                not cached_html[i]
                or (done and not cached_complete[i])
                or done_batch
                or (
                    not cached_complete[i]
                    and (
                        len(page) - len(cached_html[i]) >= HTML_IFRAME_MIN_DELTA
                        or now - cached_html_at[i] >= HTML_IFRAME_MAX_STALE
                    )
                )
            )
            if should_reload:
                cached_previews[i] = render_preview(page_text, i, scale, active, prompt, page_tokens, scroll_seconds)
                cached_html[i] = page
                cached_html_at[i] = now
                cached_complete[i] = done
                cached_text_len[i] = len(page_text)
                cached_preview_at[i] = now
                updates[i] = cached_previews[i]
        raw_update = skip
        status_update = skip
        if now - last_raw_at >= HTML_COMPONENT_MIN_INTERVAL or done_batch:
            raw_update = gr.update(value=text, label=f"Output  {speed_info}" if speed_info else "Output")
            status_update = speed_info
            last_raw_at = now
        if any(update is not skip for update in updates) or raw_update is not skip or status_update is not skip:
            yield tuple([*updates, raw_update, final_token_counts, page_count, status_update])

    if final_text:
        pages = split_batch_output(final_text, page_count)
        scale = get_html_scale()
        scroll_seconds = get_scroll_seconds()
        final_previews = [
            render_preview(
                pages[i] if i < page_count else "",
                i,
                scale,
                i < page_count,
                prompt,
                final_token_counts[i] if i < len(final_token_counts) else None,
                scroll_seconds,
            )
            for i in range(MAX_HTML_PREVIEWS)
        ]
        yield tuple([
            *final_previews,
            gr.update(value=final_text, label=f"Output  {final_status}" if final_status else "Output"),
            final_token_counts,
            page_count,
            final_status,
        ])


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

    with gr.Tab("✨HTML生成", elem_classes="html-grid-tab"):
        with gr.Row(elem_classes="html-grid-main"):
            with gr.Column(scale=7, elem_classes="html-grid-controls"):
                html_api_url = gr.Textbox(label="Batch API URL", value=DEFAULT_BATCH_API_URL)
                html_password = gr.Textbox(label="Password", placeholder="输入password", value="", type="password")
                html_prompt_choice = gr.Dropdown(
                    choices=HTML_PROMPT_CHOICES,
                    value=HTML_PROMPT_CHOICES[0],
                    label=None,
                    show_label=False,
                    elem_classes="html-prompt-choice",
                )
                html_prompt = gr.Textbox(lines=6, label="Prompt", value=DEFAULT_HTML_PROMPT)
                with gr.Row():
                    html_submit = gr.Button("Generate HTML Grid", variant="primary")
                    html_stop = gr.Button("Stop", variant="secondary")
                    html_clear = gr.Button("Clear")
                html_token_count = gr.Slider(50, HTML_GEN_LIMIT, label="max_tokens", step=50, value=HTML_GEN_LIMIT)
                html_page_count = gr.Slider(1, MAX_HTML_PREVIEWS, label="batch_size", step=1, value=15)
                html_scale = gr.Slider(20, 100, label="Preview Scale %", step=5, value=35)
                html_scroll_seconds = gr.Slider(0, 10, label="Preview Scroll Seconds", step=1, value=5)
                html_chunk_size = gr.Slider(1, 128, value=8, step=1, label="chunk_size")
                html_temperature = gr.Slider(0.2, 2.0, label="temperature", step=0.1, value=1.0)
                html_top_k = gr.Slider(1, 1000, value=500, step=1, label="top_k")
                html_top_p = gr.Slider(0.0, 0.95, label="top_p", step=0.05, value=0.5)
                html_presence_penalty = gr.Slider(0.0, 2.0, label="alpha_presence", step=0.1, value=1.0)
                html_count_penalty = gr.Slider(0.0, 1.0, label="alpha_frequency", step=0.01, value=0.1)
                html_penalty_decay = gr.Slider(0.99, 0.999, label="alpha_decay", step=0.001, value=0.99)
                html_stop_tokens = gr.Textbox(label="stop_tokens(JSON)", value=DEFAULT_STOP_TOKENS_JSON)
                html_status = gr.Markdown("待命")
                html_token_counts = gr.State([0 for _ in range(MAX_HTML_PREVIEWS)])
                html_render_count = gr.State(15)
            with gr.Column(scale=33, elem_classes="html-grid-pages"):
                html_previews = []
                for _ in range(5):
                    with gr.Row(elem_classes="html-grid-preview-row"):
                        for _ in range(3):
                            html_previews.append(gr.HTML(render_preview(active=False), elem_classes="html-grid-preview"))
            with gr.Column(scale=7, elem_classes="html-grid-output"):
                html_raw_output = gr.Textbox(label="Output", lines=10, max_lines=40, **copy_button_kwargs(gr.Textbox))

        html_outputs = [*html_previews, html_raw_output, html_token_counts, html_render_count, html_status]
        html_inputs = [
            html_prompt,
            html_api_url,
            html_password,
            html_token_count,
            html_page_count,
            html_scale,
            html_scroll_seconds,
            html_temperature,
            html_top_k,
            html_top_p,
            html_presence_penalty,
            html_count_penalty,
            html_penalty_decay,
            html_chunk_size,
            html_stop_tokens,
        ]
        html_event = html_submit.click(
            evaluate_html_grid_api,
            html_inputs,
            html_outputs,
            show_progress="hidden",
            stream_every=0.5,
        )
        html_stop.click(
            lambda: "已停止",
            outputs=[html_status],
            cancels=[html_event],
        )
        html_clear.click(
            clear_html_grid,
            inputs=[html_page_count],
            outputs=html_outputs,
            queue=False,
        )
        html_scale.change(
            render_html_grid_from_raw,
            [html_prompt, html_raw_output, html_render_count, html_scale, html_scroll_seconds, html_token_counts],
            html_previews,
            queue=False,
            show_progress="hidden",
        )
        html_scroll_seconds.change(
            render_html_grid_from_raw,
            [html_prompt, html_raw_output, html_render_count, html_scale, html_scroll_seconds, html_token_counts],
            html_previews,
            queue=False,
            show_progress="hidden",
        )
        html_prompt_choice.change(
            html_prompt_from_choice,
            html_prompt_choice,
            html_prompt,
            queue=False,
            show_progress="hidden",
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=HTML_GRID_CSS, theme="ocean")
