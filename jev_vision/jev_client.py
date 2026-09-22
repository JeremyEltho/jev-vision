"""Thin HTTP client for TypeSafe's System One API (Jev).

See https://docs.typesafe.ai/api for the current request/response contract.
Kept deliberately small: one method, one endpoint. The pipeline is
responsible for shaping questions; this module is only responsible for
getting them there and back.
"""

from __future__ import annotations

import os
from typing import Any

import requests

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"


class JevError(RuntimeError):
    """Raised when the Jev API returns an error or an unusable response."""


class JevClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 5.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError(
                "No TypeSafe API key found. Pass api_key= or set TYPESAFE_API_KEY."
            )
        self.model = model
        self.timeout = timeout
        self._session = session or requests.Session()

    def ask(self, state: Any, questions: dict[str, dict]) -> dict:
        """Send one batch of independent questions over shared state.

        Questions run in parallel server-side, so batching every ambiguous
        detection from a frame into a single call (the "speculative
        fan-out" pattern) costs about the same latency as asking one.
        """
        response = self._session.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "state": state, "questions": questions},
            timeout=self.timeout,
        )
        if not response.ok:
            raise JevError(
                f"Jev request failed ({response.status_code}): {response.text}"
            )

        body = response.json()
        if "answers" not in body:
            raise JevError(f"Unexpected Jev response shape: {body!r}")
        return body
