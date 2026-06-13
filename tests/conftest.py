# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from genkit._core._registry import Registry
from genkit.middleware import GenerateMiddlewareContext

try:
    from genkit._core._typing import Artifact
except ImportError:  # pragma: no cover - older published genkit builds
    Artifact = Any  # type: ignore[misc,assignment]


@dataclass
class _MockSessionState:
    session_id: str | None = 'test-session'


class MockSession:
    """Minimal session double for recipe tests (artifacts + session_id)."""

    def __init__(self, session_id: str = 'test-session') -> None:
        self._state = _MockSessionState(session_id=session_id)
        self._artifacts: list[Any] = []

    async def state(self) -> _MockSessionState:
        return self._state

    async def get_artifacts(self) -> list[Any]:
        return list(self._artifacts)

    async def add_artifacts(self, *artifacts: Any, _suppress_events: bool = False) -> None:
        for art in artifacts:
            name = getattr(art, 'name', None)
            if name:
                for i, existing in enumerate(self._artifacts):
                    if getattr(existing, 'name', None) == name:
                        self._artifacts[i] = art
                        break
                else:
                    self._artifacts.append(art)
            else:
                self._artifacts.append(art)


@pytest.fixture
def ctx() -> GenerateMiddlewareContext:
    return GenerateMiddlewareContext(registry=Registry(), abort_signal=asyncio.Event())
