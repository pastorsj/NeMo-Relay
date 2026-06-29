# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import StrEnum
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from nemo_relay import Json, JsonObject, ScopeHandle

DEFAULT_SEARCH_LIMIT: int
MAX_SEARCH_LIMIT: int

class MemoryErrorCode(StrEnum):
    INVALID_REQUEST: MemoryErrorCode
    UNSUPPORTED: MemoryErrorCode
    DEADLINE_EXCEEDED: MemoryErrorCode
    CANCELLED: MemoryErrorCode
    CONFLICT: MemoryErrorCode
    PROVIDER_UNAVAILABLE: MemoryErrorCode
    INTERNAL: MemoryErrorCode

class MemoryOperationError:
    code: MemoryErrorCode
    message: str
    retryable: bool
    operation_id: str | None
    provider: str | None
    details: JsonObject
    def __init__(
        self,
        code: MemoryErrorCode,
        message: str,
        retryable: bool = False,
        operation_id: str | None = None,
        provider: str | None = None,
        details: JsonObject = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryOperationError: ...
    def to_dict(self) -> JsonObject: ...

class MemoryContractError(ValueError):
    error: MemoryOperationError
    def __init__(self, error: MemoryOperationError) -> None: ...

class MemoryNamespace:
    tenant_id: str
    subject_id: str
    session_id: str | None
    agent_id: str | None
    def __init__(
        self,
        tenant_id: str,
        subject_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> None: ...
    def validate(self) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryNamespace: ...
    def to_dict(self) -> JsonObject: ...

class MemorySearchScope(StrEnum):
    SUBJECT: MemorySearchScope
    AGENT: MemorySearchScope
    SESSION: MemorySearchScope
    EXACT: MemorySearchScope
    def validate(self, namespace: MemoryNamespace) -> None: ...

class FailurePolicy(StrEnum):
    FAIL_OPEN: FailurePolicy
    FAIL_CLOSED: FailurePolicy

class EvidenceMode(StrEnum):
    REFERENCES: EvidenceMode

class WriteProjection(StrEnum):
    USER: WriteProjection
    USER_AND_ASSISTANT: WriteProjection

class AutomaticMemoryConfig:
    namespace: MemoryNamespace | None
    search_scope: MemorySearchScope
    max_candidates: int
    max_items: int
    max_estimated_tokens: int
    operation_timeout_millis: int
    identity_policy: FailurePolicy
    retrieval_policy: FailurePolicy
    storage_policy: FailurePolicy
    write_projection: WriteProjection
    evidence_mode: EvidenceMode
    def __init__(
        self,
        namespace: MemoryNamespace | None = None,
        search_scope: MemorySearchScope = MemorySearchScope.SUBJECT,
        max_candidates: int = 20,
        max_items: int = 5,
        max_estimated_tokens: int = 512,
        operation_timeout_millis: int = 2_000,
        identity_policy: FailurePolicy = FailurePolicy.FAIL_CLOSED,
        retrieval_policy: FailurePolicy = FailurePolicy.FAIL_OPEN,
        storage_policy: FailurePolicy = FailurePolicy.FAIL_OPEN,
        write_projection: WriteProjection = WriteProjection.USER_AND_ASSISTANT,
        evidence_mode: EvidenceMode = EvidenceMode.REFERENCES,
    ) -> None: ...
    def to_dict(self) -> JsonObject: ...

class InMemoryAutomaticMemory:
    def __init__(self, config: AutomaticMemoryConfig | None = None) -> None: ...
    @property
    def active_turns(self) -> int: ...
    def install(
        self,
        *,
        name: str = "automatic_memory",
        priority: int = 0,
        scope: ScopeHandle | None = None,
    ) -> Self: ...
    def close(self) -> bool: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

class MemoryContent:
    kind: str
    text: str | None
    value: Json | None
    reference: str | None
    preview: str | None
    def __init__(
        self,
        kind: str,
        text: str | None = None,
        value: Json | None = None,
        reference: str | None = None,
        preview: str | None = None,
    ) -> None: ...
    @classmethod
    def text_content(cls, text: str) -> MemoryContent: ...
    @classmethod
    def json_content(cls, value: Json) -> MemoryContent: ...
    @classmethod
    def reference_content(cls, reference: str, preview: str | None = None) -> MemoryContent: ...
    def validate(self) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryContent: ...
    def to_dict(self) -> JsonObject: ...

class MemoryProvenance:
    source: str
    source_ids: tuple[str, ...]
    parent_memory_ids: tuple[str, ...]
    metadata: JsonObject
    def __init__(
        self,
        source: str,
        source_ids: tuple[str, ...] = (),
        parent_memory_ids: tuple[str, ...] = (),
        metadata: JsonObject = ...,
    ) -> None: ...
    def validate(self) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryProvenance: ...
    def to_dict(self) -> JsonObject: ...

class MemoryRecord:
    id: str
    provider: str
    namespace: MemoryNamespace
    content: MemoryContent
    event_timestamp: datetime
    ingested_at: datetime
    provenance: MemoryProvenance
    metadata: JsonObject
    provider_metadata: JsonObject
    def __init__(
        self,
        id: str,
        provider: str,
        namespace: MemoryNamespace,
        content: MemoryContent,
        event_timestamp: datetime,
        ingested_at: datetime,
        provenance: MemoryProvenance,
        metadata: JsonObject = ...,
        provider_metadata: JsonObject = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryRecord: ...
    def to_dict(self) -> JsonObject: ...

class MemoryMatch:
    record: MemoryRecord
    score: float
    rank: int
    def __init__(self, record: MemoryRecord, score: float, rank: int) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryMatch: ...
    def to_dict(self) -> JsonObject: ...

class MemoryFilter:
    metadata: JsonObject
    event_after: datetime | None
    event_before: datetime | None
    ingested_after: datetime | None
    ingested_before: datetime | None
    def __init__(
        self,
        metadata: JsonObject = ...,
        event_after: datetime | None = None,
        event_before: datetime | None = None,
        ingested_after: datetime | None = None,
        ingested_before: datetime | None = None,
    ) -> None: ...
    def validate(self) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryFilter: ...

class MemoryRequestContext:
    operation_id: str
    deadline: datetime | None
    def __init__(self, operation_id: str, deadline: datetime | None = None) -> None: ...
    def validate(self) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryRequestContext: ...

class MemorySearchRequest:
    context: MemoryRequestContext
    namespace: MemoryNamespace
    query: str
    scope: MemorySearchScope
    filter: MemoryFilter
    limit: int
    def __init__(
        self,
        context: MemoryRequestContext,
        namespace: MemoryNamespace,
        query: str,
        scope: MemorySearchScope = MemorySearchScope.SUBJECT,
        filter: MemoryFilter = ...,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> None: ...
    def validate(self) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemorySearchRequest: ...
    def to_dict(self) -> JsonObject: ...

class MemorySearchResult:
    matches: tuple[MemoryMatch, ...]
    partial_errors: tuple[MemoryOperationError, ...]
    def __init__(
        self,
        matches: tuple[MemoryMatch, ...] = (),
        partial_errors: tuple[MemoryOperationError, ...] = (),
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemorySearchResult: ...
    def to_dict(self) -> JsonObject: ...

class MemoryStoreRequest:
    context: MemoryRequestContext
    namespace: MemoryNamespace
    content: MemoryContent
    event_timestamp: datetime
    provenance: MemoryProvenance
    metadata: JsonObject
    idempotency_key: str | None
    def __init__(
        self,
        context: MemoryRequestContext,
        namespace: MemoryNamespace,
        content: MemoryContent,
        event_timestamp: datetime,
        provenance: MemoryProvenance,
        metadata: JsonObject = ...,
        idempotency_key: str | None = None,
    ) -> None: ...
    def validate(self) -> None: ...
    def to_dict(self) -> JsonObject: ...

class MemoryStoreDisposition(StrEnum):
    CREATED: MemoryStoreDisposition
    EXISTING: MemoryStoreDisposition

class MemoryStoreResult:
    record: MemoryRecord
    disposition: MemoryStoreDisposition
    def __init__(self, record: MemoryRecord, disposition: MemoryStoreDisposition) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryStoreResult: ...
    def to_dict(self) -> JsonObject: ...

class MemoryCapabilities:
    update: bool
    delete: bool
    batch_store: bool
    maintenance: bool
    feedback: bool
    health: bool
    def __init__(
        self,
        update: bool = False,
        delete: bool = False,
        batch_store: bool = False,
        maintenance: bool = False,
        feedback: bool = False,
        health: bool = False,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryCapabilities: ...
    def to_dict(self) -> JsonObject: ...

@runtime_checkable
class MemoryProvider(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def capabilities(self) -> MemoryCapabilities: ...
    async def search(self, request: MemorySearchRequest) -> MemorySearchResult: ...
    async def store(self, request: MemoryStoreRequest) -> MemoryStoreResult: ...
