# mistral_client.py
import os
import requests
import httpx
import asyncio
from dotenv import load_dotenv
from typing import Iterator, AsyncIterator

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def ask_mistral(prompt: str, system_prompt: str = "") -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError("Set MISTRAL_API_KEY in your environment")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "mistral-medium-latest",  # or another Mistral model you prefer
        "messages": messages,
        "temperature": 0.3,
        "stream": False,
    }

    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def chat_mistral(
    messages: list,
    model: str = "mistral-small-latest",  # faster than mistral-medium, good quality
    temperature: float = 0.3,
    timeout: int = 30,
    max_tokens: int = 150,  # limit response length for speed
) -> str:
    """
    Chat-style call that preserves history. Expects messages in OpenAI/Mistral format:
    [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("Set MISTRAL_API_KEY in your environment")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,  # limit length for faster responses
        "stream": False,
    }

    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def chat_mistral_async(
    messages: list,
    model: str = "mistral-small-latest",
    temperature: float = 0.3,
    timeout: int = 30,
    max_tokens: int = 150,
) -> str:
    """
    Async version: non-blocking HTTP call that allows event loop to handle other tasks.
    This reduces perceived latency by not blocking the websocket handler.
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("Set MISTRAL_API_KEY in your environment")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def chat_mistral_stream(
    messages: list,
    model: str = "mistral-small-latest",
    temperature: float = 0.3,
    timeout: int = 30,
    max_tokens: int = 150,
) -> Iterator[str]:
    """
    Streaming version: yields text chunks as they arrive from Mistral.
    This allows TTS to start speaking before the full response is ready.
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("Set MISTRAL_API_KEY in your environment")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,  # Enable streaming
    }

    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
        stream=True,  # Important: enable streaming response
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data_str = line[6:]  # Remove "data: " prefix
            if data_str == "[DONE]":
                break
            try:
                import json
                data = json.loads(data_str)
                if "choices" in data and len(data["choices"]) > 0:
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
            except json.JSONDecodeError:
                continue


async def chat_mistral_stream_async(
    messages: list,
    model: str = "mistral-small-latest",
    temperature: float = 0.3,
    timeout: int = 30,
    max_tokens: int = 150,
) -> AsyncIterator[str]:
    """
    Async streaming version: yields text chunks as they arrive from Mistral.
    Non-blocking, allows TTS to start speaking before the full response is ready.
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("Set MISTRAL_API_KEY in your environment")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        timeout_config = httpx.Timeout(timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream(
                "POST",
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError as e:
                            # Skip malformed JSON lines
                            continue
                        except Exception as e:
                            print(f"Error parsing stream chunk: {e}")
                            continue
    except httpx.TimeoutException as e:
        raise RuntimeError(f"Mistral streaming timeout after {timeout}s") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Mistral API error: {e.response.status_code} - {e.response.text}") from e
    except Exception as e:
        raise RuntimeError(f"Mistral streaming error: {type(e).__name__}: {e}") from e
