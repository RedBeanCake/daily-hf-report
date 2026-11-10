"""One retry budget for all LLM calls (SDK retries are disabled)."""

import logging
import time
from typing import Any, Callable

from http_utils import retry_call

LOGGER = logging.getLogger(__name__)
MODEL = "qwen3.7-plus"


def complete(client: Any, prompt: str, *, validate: Callable = str, json_mode: bool = False) -> Any:
    def call() -> Any:
        started = time.monotonic()
        options = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=180,
                **options,
            )
            content = response.choices[0].message.content
            if not content or response.choices[0].finish_reason == "length":
                raise ValueError("Empty or truncated LLM response")
            return validate(content)
        finally:
            LOGGER.info("LLM request finished in %.1fs", time.monotonic() - started)

    return retry_call(call, attempts=3, base_delay=2.0)
