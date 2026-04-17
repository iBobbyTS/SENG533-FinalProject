from __future__ import annotations

import json
import time
from typing import Any

import requests


class LMStudioClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.api_mode = "native" if self.base_url.endswith("/api/v1") else "openai_compat"

    def _post_with_retries(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.post(
                    f"{self.base_url}/{endpoint}",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LM Studio request failed: {last_error}") from last_error

    def _convert_messages_to_native_input(self, messages: list[dict[str, Any]]) -> tuple[str | list[dict[str, Any]], str | None]:
        system_prompt: str | None = None
        input_items: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                if isinstance(content, str):
                    system_prompt = content
                continue
            if role != "user":
                continue

            if isinstance(content, str):
                input_items.append({"type": "text", "content": content})
                continue

            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        input_items.append({"type": "text", "content": block.get("text", "")})
                    elif block.get("type") == "image_url":
                        image_url = block.get("image_url", {})
                        if isinstance(image_url, dict):
                            data_url = image_url.get("url", "")
                            input_items.append({"type": "image", "data_url": data_url})

        if len(input_items) == 1 and input_items[0].get("type") == "text":
            return input_items[0].get("content", ""), system_prompt
        return input_items, system_prompt

    def _chat_completion_openai_compat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
        stream: bool | None = None,
        stream_idle_timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        if extra_body:
            payload.update(extra_body)
        data = self._post_with_retries("chat/completions", payload)
        choice = data["choices"][0]
        content = choice.get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
            )
        return {
            "content": content or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage", {}),
            "raw": data,
        }

    def _consume_native_stream(
        self,
        payload: dict[str, Any],
        *,
        idle_timeout_seconds: int,
        stream_info: dict[str, Any],
    ) -> dict[str, Any]:
        def parse_event_data(event_type: str | None, raw_parts: list[bytes]) -> dict[str, Any]:
            if not raw_parts:
                return {}

            decode_errors = "strict"
            payload_variants: list[str] = []
            raw_joiners = [b"\n", b"", None]
            for joiner in raw_joiners:
                if joiner is None:
                    text = "\\n".join(part.decode("utf-8", errors="replace") for part in raw_parts)
                else:
                    text = joiner.join(raw_parts).decode("utf-8", errors=decode_errors)
                if text not in payload_variants:
                    payload_variants.append(text)

            last_error: Exception | None = None
            for variant in payload_variants:
                try:
                    return json.loads(variant)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc

            if event_type in {"message.delta", "reasoning.delta"}:
                warnings = stream_info.setdefault("parse_warnings", [])
                warnings.append(
                    {
                        "event_type": event_type,
                        "error": str(last_error),
                        "raw_preview": payload_variants[0][:200] if payload_variants else "",
                    }
                )
                return {}

            raise last_error if last_error is not None else RuntimeError("Failed to parse stream event data")

        with self.session.post(
            f"{self.base_url}/chat",
            json=payload,
            timeout=(30, idle_timeout_seconds),
            stream=True,
        ) as response:
            response.raise_for_status()
            current_event: str | None = None
            current_data: list[bytes] = []
            final_result: dict[str, Any] | None = None
            message_parts: list[str] = []
            reasoning_parts: list[str] = []
            stream_info.setdefault("event_counts", {})

            def process_event() -> None:
                nonlocal current_event, current_data, final_result
                if current_event is None and not current_data:
                    return
                event_type = current_event or "unknown"
                data = parse_event_data(current_event, current_data)
                if current_event is None:
                    event_type = data.get("type", "unknown")
                stream_info["last_event_type"] = event_type
                stream_info["last_event_at"] = time.time()
                counts = stream_info["event_counts"]
                counts[event_type] = counts.get(event_type, 0) + 1
                if isinstance(data, dict) and data.get("progress") is not None:
                    stream_info["last_progress"] = data.get("progress")
                if event_type == "message.delta":
                    message_parts.append(str(data.get("content", "")))
                elif event_type == "reasoning.delta":
                    reasoning_parts.append(str(data.get("content", "")))
                elif event_type == "error":
                    stream_info["error_event"] = data.get("error")
                elif event_type == "chat.end":
                    final_result = data.get("result", {})
                current_event = None
                current_data = []

            for raw_line in response.iter_lines(decode_unicode=False):
                if raw_line is None:
                    continue
                line = raw_line.rstrip(b"\r")
                if not line:
                    process_event()
                    continue
                if line.startswith(b":"):
                    continue
                if line.startswith(b"event:"):
                    current_event = line[len(b"event:") :].strip().decode("utf-8", errors="replace")
                elif line.startswith(b"data:"):
                    current_data.append(line[len(b"data:") :].lstrip())

            process_event()
            if final_result is None:
                raise RuntimeError("LM Studio native stream ended without chat.end")

        output_items = final_result.get("output", []) if isinstance(final_result, dict) else []
        content_parts = [
            item.get("content", "")
            for item in output_items
            if isinstance(item, dict) and item.get("type") == "message"
        ]
        stats = final_result.get("stats", {}) if isinstance(final_result, dict) else {}
        usage = {
            "prompt_tokens": stats.get("input_tokens"),
            "completion_tokens": stats.get("total_output_tokens"),
            "total_tokens": (
                (stats.get("input_tokens") or 0) + (stats.get("total_output_tokens") or 0)
                if stats.get("input_tokens") is not None and stats.get("total_output_tokens") is not None
                else None
            ),
            "reasoning_output_tokens": stats.get("reasoning_output_tokens"),
            "tokens_per_second": stats.get("tokens_per_second"),
            "time_to_first_token_seconds": stats.get("time_to_first_token_seconds"),
            "model_load_time_seconds": stats.get("model_load_time_seconds"),
        }
        stream_info["reasoning_chars"] = len("".join(reasoning_parts))
        stream_info["message_chars"] = len("".join(message_parts))
        return {
            "content": "".join(content_parts) or "".join(message_parts),
            "finish_reason": None,
            "usage": usage,
            "raw": final_result,
            "stream_debug": stream_info,
        }

    def _chat_completion_native_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
        stream_idle_timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        input_payload, system_prompt = self._convert_messages_to_native_input(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_payload,
            "temperature": temperature,
            "store": False,
            "stream": True,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        last_error: Exception | None = None
        last_context = ""
        idle_timeout = stream_idle_timeout_seconds or self.timeout_seconds
        for attempt in range(3):
            stream_info: dict[str, Any] = {}
            try:
                return self._consume_native_stream(
                    payload,
                    idle_timeout_seconds=idle_timeout,
                    stream_info=stream_info,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                context_parts = []
                last_event_type = stream_info.get("last_event_type")
                if last_event_type:
                    context_parts.append(f"last_stream_event={last_event_type}")
                if stream_info.get("last_progress") is not None:
                    context_parts.append(f"last_progress={stream_info['last_progress']}")
                if stream_info.get("last_event_at") is not None:
                    idle_for = time.time() - float(stream_info["last_event_at"])
                    context_parts.append(f"idle_after_last_event_seconds={idle_for:.3f}")
                if stream_info.get("event_counts"):
                    context_parts.append(f"event_counts={stream_info['event_counts']}")
                if stream_info.get("error_event"):
                    context_parts.append(f"stream_error={stream_info['error_event']}")
                last_context = "; ".join(context_parts)
                if attempt == 2:
                    break
                time.sleep(1.5 * (attempt + 1))
        if last_context:
            raise RuntimeError(f"LM Studio streaming request failed: {last_error}; {last_context}") from last_error
        raise RuntimeError(f"LM Studio streaming request failed: {last_error}") from last_error

    def _chat_completion_native(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
        stream: bool | None = None,
        stream_idle_timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        if stream is None:
            stream = True
        if stream:
            return self._chat_completion_native_stream(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                extra_body=extra_body,
                stream_idle_timeout_seconds=stream_idle_timeout_seconds,
            )
        input_payload, system_prompt = self._convert_messages_to_native_input(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_payload,
            "temperature": temperature,
            "store": False,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)
        data = self._post_with_retries("chat", payload)
        content_parts = []
        for item in data.get("output", []):
            if isinstance(item, dict) and item.get("type") == "message":
                content_parts.append(item.get("content", ""))
        stats = data.get("stats", {})
        usage = {
            "prompt_tokens": stats.get("input_tokens"),
            "completion_tokens": stats.get("total_output_tokens"),
            "total_tokens": (
                (stats.get("input_tokens") or 0) + (stats.get("total_output_tokens") or 0)
                if stats.get("input_tokens") is not None and stats.get("total_output_tokens") is not None
                else None
            ),
            "reasoning_output_tokens": stats.get("reasoning_output_tokens"),
            "tokens_per_second": stats.get("tokens_per_second"),
            "time_to_first_token_seconds": stats.get("time_to_first_token_seconds"),
            "model_load_time_seconds": stats.get("model_load_time_seconds"),
        }
        return {
            "content": "".join(content_parts),
            "finish_reason": None,
            "usage": usage,
            "raw": data,
        }

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
        stream: bool | None = None,
        stream_idle_timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        if self.api_mode == "native":
            return self._chat_completion_native(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                extra_body=extra_body,
                stream=stream,
                stream_idle_timeout_seconds=stream_idle_timeout_seconds,
            )
        return self._chat_completion_openai_compat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            extra_body=extra_body,
            stream=stream,
            stream_idle_timeout_seconds=stream_idle_timeout_seconds,
        )
