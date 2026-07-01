# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Protocol

from nemo_relay import Json, JsonObject, LLMRequest
from nemo_relay.plugin import ConfigDiagnostic, PluginContext

class MemoryLifecyclePhase(StrEnum):
    BEFORE_LLM: "MemoryLifecyclePhase"
    AFTER_LLM: "MemoryLifecyclePhase"

class MemoryAttachment:
    reference: str
    content: str
    score: float | None
    def __init__(self, reference: str, content: str, score: float | None = None) -> None: ...

class MemoryAction:
    attachments: tuple[MemoryAttachment, ...]
    stored_references: tuple[str, ...]
    def __init__(
        self,
        attachments: tuple[MemoryAttachment, ...] = ...,
        stored_references: tuple[str, ...] = ...,
    ) -> None: ...

class MemoryLifecycleEvent:
    phase: MemoryLifecyclePhase
    call_name: str
    context: JsonObject
    request: LLMRequest
    response: Json | None
    attachments: tuple[MemoryAttachment, ...]
    def __init__(
        self,
        phase: MemoryLifecyclePhase,
        call_name: str,
        context: JsonObject,
        request: LLMRequest,
        response: Json | None = None,
        attachments: tuple[MemoryAttachment, ...] = ...,
    ) -> None: ...

class MemoryListener(Protocol):
    name: str
    async def on_event(self, event: MemoryLifecycleEvent) -> MemoryAction | None: ...

class MemoryPlugin:
    listener: MemoryListener
    def __init__(self, listener: MemoryListener) -> None: ...
    def validate(self, plugin_config: JsonObject) -> list[ConfigDiagnostic] | None: ...
    def register(self, plugin_config: JsonObject, context: PluginContext) -> None: ...

__all__: list[str]
