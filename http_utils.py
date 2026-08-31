"""Small retry helpers shared by the external data sources."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence, Tuple

import requests


RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class RequestError(RuntimeError):
    """Raised when an external request cannot be completed."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def retry_call(
    operation: Callable[[], Any],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    retry_exceptions: Tuple[type[BaseException], ...] = (Exception,),
) -> Any:
    """Run an operation with bounded exponential backoff."""

    last_error: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return operation()
        except retry_exceptions as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(base_delay * (2**attempt))
    raise RequestError(f"operation failed after {attempts} attempts") from last_error


def request_with_retry(
    method: str,
    url: str,
    *,
    session: Optional[requests.Session] = None,
    attempts: int = 3,
    timeout: float = 15,
    **kwargs: Any,
) -> requests.Response:
    """Request a URL and retry transient network/server failures.

    Non-retryable client errors (4xx other than the transient 408/425/429) fail
    fast instead of burning the full retry budget, and the HTTP status code is
    preserved on the raised ``RequestError`` so callers can react to them -- for
    example, Hugging Face returns 400 for a date that is still in the future from
    its UTC perspective, and the caller can then fall back to an earlier date.
    """

    client = session or requests.Session()
    last_error: Optional[BaseException] = None

    for attempt in range(attempts):
        try:
            response = client.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(1.0 * (2**attempt))
                continue
            break

        if response.status_code in RETRYABLE_STATUS_CODES:
            last_error = RequestError(
                f"HTTP {response.status_code} from {url}",
                status_code=response.status_code,
            )
            if attempt < attempts - 1:
                time.sleep(1.0 * (2**attempt))
                continue
            break

        # Permanent client errors are not worth retrying; surface them immediately
        # with the status code attached so callers can branch on it.
        if 400 <= response.status_code < 500:
            raise RequestError(
                f"HTTP {response.status_code} from {url}: {response.text[:200]}",
                status_code=response.status_code,
            )

        response.raise_for_status()
        return response

    raise RequestError(
        f"request failed after {attempts} attempts: {url}",
        status_code=getattr(last_error, "status_code", None),
    ) from last_error


def request_json(url: str, **kwargs: Any) -> Any:
    response = request_with_retry("GET", url, **kwargs)
    try:
        return response.json()
    except ValueError as exc:
        raise RequestError(f"invalid JSON response from {url}") from exc


def request_text(url: str, **kwargs: Any) -> str:
    return request_with_retry("GET", url, **kwargs).text
