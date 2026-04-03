#!/usr/bin/env python3

import json
import os
import urllib.request


BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "rwkv7")
PASSWORD = os.environ.get("PASSWORD", "test_tool_call")

SYSTEM_PROMPT = """
You may use tools when needed.
Follow these rules strictly:
1. If a tool is needed, output EXACTLY one tool call.
2. A tool call must be wrapped in <tool_call> and </tool_call>.
3. Inside <tool_call>, output valid JSON only.
4. Do not use markdown code fences.
5. Do not add any explanation before or after a tool call.
6. If no tool is needed, answer normally.
Available tools:
Tool: exec
Purpose: Execute a shell command and return its output. Use with caution.
Arguments:
- command (string, required): The shell command to execute
- working_dir (string, optional): Optional working directory for the command
Tool call format:
<tool_call>
{"name": "exec", "arguments": {"command": "<command>", "working_dir": "<working_dir>"}}
</tool_call>
Tool: web_fetch
Purpose: Fetch a URL and return its readable text.
Arguments:
- url (string, required): URL to fetch.
- maxChars (integer, optional)
Tool call format:
<tool_call>
{"name": "web_fetch", "arguments": {"url": "<url>", "maxChars": 1}}
</tool_call>
# rvoone 🕊
You are rvoone, a helpful AI assistant.
Core rules:
- Be concise and direct.
- Do not claim tool results before receiving them.
---
Runtime: Linux x86_64, Python 3.12.12
Workspace: /home/alic-li/.rvoone/workspace
---
## Event Handling
If you receive a <SYS_EVENT> message during tool execution:
1. IMMEDIATELY acknowledge the event
2. The event content ALWAYS takes priority over your current task
3. Respond naturally to the event
4. Decide whether to continue your previous task or switch to the new request
"""

payload = {
    "model": MODEL,
    "stream": False,
    "enable_think": False,
    "messages": [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "帮我查询一下https://github.com/RWKV-Vibe/rwkv_lightning这个网址，他写了什么东西？",
        },
    ],
    "max_tokens": 8192,
    "temperature": 1,
    "alpha_frequency": 0,
    "alpha_presence": 0,
    "top_p": 0.3,
    "stream": True,
}

headers = {
    "Content-Type": "application/json",
}
if PASSWORD:
    headers["Authorization"] = f"Bearer {PASSWORD}"

request = urllib.request.Request(
    url=f"{BASE_URL}/openai/v1/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers=headers,
    method="POST",
)

with urllib.request.urlopen(request) as response:
    print(response.read().decode("utf-8"))
