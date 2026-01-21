import json
from typing import Any
import aiohttp
from pathlib import Path
from loguru import logger
from utils import load_config, remove_think_tags

config = load_config(str(Path(__file__).parent / "config.yaml"))


class Ollama:
    def __init__(self, model_name: str) -> None:
        self.url: str = config["base_url"]
        self.temperature: float = config["temperature"]
        self.max_tokens: int = config["max_tokens"]
        self.repeat_penalty: float = config["repeat_penalty"]
        self.model_name = model_name
        self.history: list[dict[str, str]] = []

    async def _generate(self, prompt: str):
        url = f"{self.url}/api/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                # "repeat_penalty": self.repeat_penalty,
            },
            "stream": True,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    raise Exception(f"API调用失败: {response.status}")

                full_response = ""
                final_context = {}

                # Process streaming response line by line
                async for line in response.content:
                    if not line:
                        continue
                    try:
                        # Decode and parse JSON
                        chunk = json.loads(line.decode("utf-8"))

                        # Concatenate model generated content
                        if "response" in chunk:
                            content_piece = chunk["response"]
                            full_response += content_piece

                            # Print to console in real time
                            print(content_piece, end="", flush=True)

                        # If it's the last part, save context information
                        if chunk.get("done", False):
                            final_context = chunk
                            break

                    except json.JSONDecodeError:
                        logger.error(f"Unparseable JSON line: {line}")

                # After streaming output ends, print newline
                print("\n")

                # Extract resource consumption information
                resource_consumption = {
                    "total_duration": final_context.get("total_duration", 0),
                    "prompt_eval_count": final_context.get("prompt_eval_count", 0),
                    "prompt_eval_duration": final_context.get(
                        "prompt_eval_duration", 0
                    ),
                    "eval_count": final_context.get("eval_count", 0),
                    "eval_duration": final_context.get("eval_duration", 0),
                }

                logger.info(
                    f"Ollama API call completed. Model: {self.model_name}, Time: {resource_consumption['total_duration']/1e9:.2f}s, Tokens: {resource_consumption['eval_count']}"
                )

                # Remove think tags and return
                result_text = remove_think_tags(full_response.strip())
                return result_text, resource_consumption

    async def _chat(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        _messages: list[dict[str, str]] = self.history + messages

        # logger.debug(f"Chat Messages: {_messages}")

        url: str = f"{self.url}/api/chat"
        data: dict[str, Any] = {
            "model": self.model_name,
            "messages": _messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                # "repeat_penalty": self.repeat_penalty,
            },
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    raise Exception(f"API call failed: {response.status}")

                full_response = ""
                final_context: dict[Any, Any] = {}

                # Process streaming response line by line
                async for line in response.content:
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line.decode("utf-8"))

                        # Chat API response format: { message: { role, content }, done: false }
                        content_piece = ""
                        if isinstance(chunk.get("message"), dict):
                            content_piece = chunk["message"].get("content", "")
                        elif "response" in chunk:  # Compatible with generate format
                            content_piece = chunk.get("response", "")

                        if content_piece:
                            full_response += content_piece
                            # print(content_piece, end="", flush=True)

                        if chunk.get("done", False):
                            final_context = chunk
                            break

                    except json.JSONDecodeError:
                        logger.error(f"Unparseable JSON line: {line}")

                print("\n")

                resource_consumption = {
                    "total_duration": final_context.get("total_duration", 0),
                    "prompt_eval_count": final_context.get("prompt_eval_count", 0),
                    "prompt_eval_duration": final_context.get(
                        "prompt_eval_duration", 0
                    ),
                    "eval_count": final_context.get("eval_count", 0),
                    "eval_duration": final_context.get("eval_duration", 0),
                }

                # Remove think tags and return
                result_text: str = remove_think_tags(full_response.strip())

                logger.info(
                    f"Chat call successful - Model: {self.model_name}, Time: {resource_consumption['total_duration']/1e9:.2f}s, Tokens: {resource_consumption['eval_count']}\nOutput content: {result_text}"
                )
                return result_text, resource_consumption

    def _add_history(self, role: str, content: str) -> None:
        self.history.append(
            {
                "role": role,
                "content": content,
            }
        )

    def _set_system_prompt(self, system_prompt: str) -> None:
        # if history has a system prompt, remove it.
        if len(self.history) > 0 and self.history[0]["role"] == "system":
            self.history.pop(0)
        self.history.insert(
            0,
            {
                "role": "system",
                "content": system_prompt,
            },
        )
