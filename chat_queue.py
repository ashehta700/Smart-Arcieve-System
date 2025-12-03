# chat_queue.py
import asyncio
from typing import Dict, Any
import httpx
import os

GPT4ALL_URL = os.getenv("GPT4ALL_URL", "http://localhost:4891")
MODEL = os.getenv("GPT4ALL_MODEL", "Llama 3.2 1B Instruct")

# --- Queue and worker ---
chat_queue = asyncio.Queue()

async def gpt4all_worker():
    async with httpx.AsyncClient(timeout=120) as client:  # 2 min timeout
        while True:
            task = await chat_queue.get()
            prompt = task["prompt"]
            fut = task["future"]
            try:
                resp = await client.post(
                    f"{GPT4ALL_URL}/v1/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 512
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                # extract the text safely
                content = data["choices"][0]["message"]["content"]
                fut.set_result(content)
            except Exception as e:
                fut.set_result(f"(GPT4All unavailable or timed out: {e})")
            finally:
                chat_queue.task_done()




# start the background worker
async def start_gpt4all_worker():
    asyncio.create_task(gpt4all_worker())

# --- Helper to submit prompt ---
async def submit_prompt(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await chat_queue.put({"prompt": prompt, "future": fut})
    return await fut
