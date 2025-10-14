import requests
import json
import curses

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "contents": [
        "Chinese: 你会发现自己对这些故事产生了共鸣\n\nEnglish:",
        "Chinese: RWKV-8 ROSA 机制：超越注意力机制的神经符号无限范围无损信息传播器，使大语言模型(LLM)能够发明自己的内心独白语言。迈向可扩展后神经方法的第一步，开启人工智能的新时代 \n\nEnglish:",
        "Chinese: 他的脸上写满了痛苦和绝望\n\nEnglish:",
        "Chinese: 因为它涉及到了一个人的身份、性别认同和性取向\n\nEnglish:",
        "Chinese: 他的眼睛里充满了泪水，他的手在颤抖\n\nEnglish:",
        "Chinese: 这个宝藏是一个神秘的地方，充满了奇迹和神秘的事物\n\nEnglish:"
        "Chinese: 你会发现自己对这些故事产生了共鸣\n\nEnglish:",
        "Chinese: RWKV-8 ROSA 机制：超越注意力机制的神经符号无限范围无损信息传播器，使大语言模型(LLM)能够发明自己的内心独白语言。迈向可扩展后神经方法的第一步，开启人工智能的新时代 \n\nEnglish:",
        "Chinese: 他的脸上写满了痛苦和绝望\n\nEnglish:",
        "Chinese: 因为它涉及到了一个人的身份、性别认同和性取向\n\nEnglish:",
        "Chinese: 他的眼睛里充满了泪水，他的手在颤抖\n\nEnglish:",
        "Chinese: 这个宝藏是一个神秘的地方，充满了奇迹和神秘的事物\n\nEnglish:"
        "Chinese: 你会发现自己对这些故事产生了共鸣\n\nEnglish:",
        "Chinese: RWKV-8 ROSA 机制：超越注意力机制的神经符号无限范围无损信息传播器，使大语言模型(LLM)能够发明自己的内心独白语言。迈向可扩展后神经方法的第一步，开启人工智能的新时代 \n\nEnglish:",
        "Chinese: 他的脸上写满了痛苦和绝望\n\nEnglish:",
        "Chinese: 因为它涉及到了一个人的身份、性别认同和性取向\n\nEnglish:",
        "Chinese: 他的眼睛里充满了泪水，他的手在颤抖\n\nEnglish:",
        "Chinese: 这个宝藏是一个神秘的地方，充满了奇迹和神秘的事物\n\nEnglish:"
        "Chinese: 你会发现自己对这些故事产生了共鸣\n\nEnglish:",
        "Chinese: RWKV-8 ROSA 机制：超越注意力机制的神经符号无限范围无损信息传播器，使大语言模型(LLM)能够发明自己的内心独白语言。迈向可扩展后神经方法的第一步，开启人工智能的新时代 \n\nEnglish:",
    ],
    "max_tokens": 1024,
    "temperature": 1,
    "stream": True
}

def display_stream(stdscr):
    curses.start_color()
    curses.use_default_colors()
    
    curses.init_pair(1, -1, -1)  
    
    stdscr.nodelay(True)
    stdscr.clear()
    stdscr.refresh()
    
    with requests.post(url, json=payload, stream=True) as r:
        buffer = [""] * len(payload["contents"])
        
        stdscr.clear()
        stdscr.addstr(0, 0, "Streaming outputs:", curses.color_pair(1))
        stdscr.addstr(1, 0, "=" * 50, curses.color_pair(1))
        stdscr.refresh()
        
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                if "choices" in obj:
                    for choice in obj["choices"]:
                        idx = choice["index"]
                        delta = choice["delta"].get("content", "")
                        if delta:
                            buffer[idx] += delta
                    
                    stdscr.clear()
                    stdscr.addstr(0, 0, "Streaming outputs:", curses.color_pair(1))
                    stdscr.addstr(1, 0, "=" * 50, curses.color_pair(1))
                    for i in range(len(buffer)):
                        stdscr.addstr(i + 2, 0, f"[Sample {i}]: {buffer[i]}", curses.color_pair(1))
                    stdscr.refresh()
                    
            except json.JSONDecodeError:
                continue
    
    stdscr.clear()
    stdscr.addstr(0, 0, "[Final Outputs]:", curses.color_pair(1))
    for i, text in enumerate(buffer):
        stdscr.addstr(i + 1, 0, f"[Sample {i}] {text}", curses.color_pair(1))
    stdscr.refresh()
    stdscr.nodelay(False) 
    stdscr.getch() 

curses.wrapper(display_stream)