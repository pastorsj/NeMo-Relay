# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral memory contracts and native automatic reference mode.

This module exposes the direct search/store contract and canonical wire
conversion helpers plus an in-memory native component for automatic recall,
write-back, and privacy-safe evidence around managed LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, NoReturn, Protocol, Self, cast, runtime_checkable

from nemo_relay import Json, JsonObject
from nemo_relay._native import _NativeInMemoryAutomaticMemory

if TYPE_CHECKING:
    from nemo_relay import ScopeHandle

DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 1_000


class MemoryErrorCode(StrEnum):
    """Stable machine-readable memory error codes."""

    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED = "unsupported"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CANCELLED = "cancelled"
    CONFLICT = "conflict"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class MemoryOperationError:
    """Serializable provider or contract failure."""

    code: MemoryErrorCode
    message: str
    retryable: bool = False
    operation_id: str | None = None
    provider: str | None = None
    details: JsonObject = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryOperationError:
        """Decode an operation error from canonical wire data."""
        return cls(
            code=MemoryErrorCode(_require_str(data, "code")),
            message=_require_str(data, "message"),
            retryable=bool(data.get("retryable", False)),
            operation_id=_optional_str(data.get("operation_id")),
            provider=_optional_str(data.get("provider")),
            details=_json_object(data.get("details", {})),
        )

    def to_dict(self) -> JsonObject:
        """Encode this error to canonical wire data."""
        return cast(JsonObject, _to_wire(self))


class MemoryContractError(ValueError):
    """Raised when a local contract value fails provider-neutral validation."""

    def __init__(self, error: MemoryOperationError) -> None:
        super().__init__(error.message)
        self.error = error


class MemoryProviderError(RuntimeError):
    """Raised when a provider operation returns a canonical memory failure."""

    def __init__(self, error: MemoryOperationError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass(frozen=True, slots=True)
class MemoryNamespace:
    """Tenant and subject partition plus optional origin context."""

    tenant_id: str
    subject_id: str
    session_id: str | None = None
    agent_id: str | None = None

    def validate(self) -> None:
        """Raise ``MemoryContractError`` when an identifier is empty."""
        _validate_identifier("tenant_id", self.tenant_id)
        _validate_identifier("subject_id", self.subject_id)
        _validate_optional_identifier("session_id", self.session_id)
        _validate_optional_identifier("agent_id", self.agent_id)

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryNamespace:
        """Decode and validate a namespace from canonical wire data."""
        namespace = cls(
            tenant_id=_require_str(data, "tenant_id"),
            subject_id=_require_str(data, "subject_id"),
            session_id=_optional_str(data.get("session_id")),
            agent_id=_optional_str(data.get("agent_id")),
        )
        namespace.validate()
        return namespace

    def to_dict(self) -> JsonObject:
        """Encode this namespace to canonical wire data."""
        self.validate()
        return cast(JsonObject, _to_wire(self))


class MemorySearchScope(StrEnum):
    """Namespace fields used to narrow a tenant and subject partition."""

    SUBJECT = "subject"
    AGENT = "agent"
    SESSION = "session"
    EXACT = "exact"

    def validate(self, namespace: MemoryNamespace) -> None:
        """Validate fields required by this scope."""
        namespace.validate()
        if self is MemorySearchScope.AGENT and namespace.agent_id is None:
            _raise_invalid("agent scope requires agent_id")
        if self is MemorySearchScope.SESSION and namespace.session_id is None:
            _raise_invalid("session scope requires session_id")


class FailurePolicy(StrEnum):
    """Failure behavior for one automatic-memory stage."""

    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"


class EvidenceMode(StrEnum):
    """Supported memory evidence capture mode."""

    REFERENCES = "references"


class WriteProjection(StrEnum):
    """Completed-turn content written by automatic memory."""

    USER = "user"
    USER_AND_ASSISTANT = "user_and_assistant"


class WriteDelivery(StrEnum):
    """Completed-turn storage delivery path."""

    INLINE = "inline"
    BACKGROUND = "background"


class MemoryBackpressurePolicy(StrEnum):
    """Admission behavior when the pending memory queue is full."""

    REJECT = "reject"
    WAIT = "wait"


@dataclass(frozen=True, slots=True)
class MemoryWorkQueueConfig:
    """Bounded local queue policy for background memory work."""

    capacity: int = 64
    backpressure: MemoryBackpressurePolicy = MemoryBackpressurePolicy.REJECT
    enqueue_timeout_millis: int = 250
    max_attempts: int = 3
    retry_initial_delay_millis: int = 25
    retry_max_delay_millis: int = 1_000
    attempt_timeout_millis: int = 2_000
    terminal_history_capacity: int = 1_024


@dataclass(frozen=True, slots=True)
class AutomaticMemoryConfig:
    """Validated configuration delegated to the native automatic component."""

    namespace: MemoryNamespace | None = None
    search_scope: MemorySearchScope = MemorySearchScope.SUBJECT
    max_candidates: int = 20
    max_items: int = 5
    max_estimated_tokens: int = 512
    operation_timeout_millis: int = 2_000
    identity_policy: FailurePolicy = FailurePolicy.FAIL_CLOSED
    retrieval_policy: FailurePolicy = FailurePolicy.FAIL_OPEN
    storage_policy: FailurePolicy = FailurePolicy.FAIL_OPEN
    write_projection: WriteProjection = WriteProjection.USER_AND_ASSISTANT
    write_delivery: WriteDelivery = WriteDelivery.INLINE
    background_queue: MemoryWorkQueueConfig = field(default_factory=MemoryWorkQueueConfig)
    evidence_mode: EvidenceMode = EvidenceMode.REFERENCES

    def to_dict(self) -> JsonObject:
        """Encode the native snake-case configuration shape."""
        return cast(JsonObject, _to_wire(self))


class InMemoryAutomaticMemory:
    """Dependency-free automatic memory for normal managed LLM calls.

    This reference implementation owns a native Rust in-memory provider. Python
    ``MemoryProvider`` implementations are not bridged into automatic execution
    yet; they remain the adapter contract for later provider work.
    """

    def __init__(self, config: AutomaticMemoryConfig | None = None) -> None:
        self._native = _NativeInMemoryAutomaticMemory(None if config is None else config.to_dict())
        self._installed = False

    @property
    def active_turns(self) -> int:
        """Return prepared calls still awaiting lifecycle completion."""
        return self._native.active_turns

    @property
    def background_status(self) -> JsonObject | None:
        """Return queue state, or ``None`` when write-back is inline."""
        return cast(JsonObject | None, self._native.background_status)

    def background_job_status(self, job_id: str) -> JsonObject | None:
        """Return retained state for one background job when available."""
        return cast(JsonObject | None, self._native.background_job_status(job_id))

    def install(
        self,
        *,
        name: str = "automatic_memory",
        priority: int = 0,
        scope: ScopeHandle | None = None,
    ) -> Self:
        """Install globally, or only within ``scope`` when supplied."""
        self._native.install(name, priority, scope)
        self._installed = True
        return self

    def close(self) -> bool:
        """Deregister once and return whether an active registration was removed."""
        try:
            return self._native.close()
        finally:
            self._installed = False

    async def flush(self, timeout_millis: int = 5_000) -> bool:
        """Wait for work accepted before this call without closing admission."""
        return cast(bool, await self._native.flush_background(timeout_millis))

    async def drain(self, timeout_millis: int = 5_000) -> bool:
        """Stop background admission and wait for all accepted work."""
        return cast(bool, await self._native.drain_background(timeout_millis))

    async def shutdown(self, timeout_millis: int = 5_000) -> bool:
        """Deregister, drain, and join the optional background worker."""
        self.close()
        return cast(bool, await self._native.shutdown_background(timeout_millis))

    def __enter__(self) -> Self:
        """Install globally on context entry when not already installed."""
        if not self._installed:
            self.install()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the installation on context exit."""
        self.close()


@dataclass(frozen=True, slots=True)
class MemoryContent:
    """Structured text, JSON content, or an opaque provider reference."""

    kind: str
    text: str | None = None
    value: Json | None = None
    reference: str | None = None
    preview: str | None = None

    @classmethod
    def text_content(cls, text: str) -> MemoryContent:
        """Create validated text content."""
        content = cls(kind="text", text=text)
        content.validate()
        return content

    @classmethod
    def json_content(cls, value: Json) -> MemoryContent:
        """Create JSON content."""
        return cls(kind="json", value=value)

    @classmethod
    def reference_content(cls, reference: str, preview: str | None = None) -> MemoryContent:
        """Create validated opaque-reference content."""
        content = cls(kind="reference", reference=reference, preview=preview)
        content.validate()
        return content

    def validate(self) -> None:
        """Validate the fields required by this content kind."""
        if self.kind == "text":
            _validate_identifier("memory text", self.text or "")
        elif self.kind == "json":
            pass
        elif self.kind == "reference":
            _validate_identifier("memory reference", self.reference or "")
        else:
            _raise_invalid(f"unknown memory content kind: {self.kind}")

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryContent:
        """Decode and validate content from canonical wire data."""
        content = cls(
            kind=_require_str(data, "kind"),
            text=_optional_str(data.get("text")),
            value=cast(Json | None, data.get("value")),
            reference=_optional_str(data.get("reference")),
            preview=_optional_str(data.get("preview")),
        )
        content.validate()
        return content

    def to_dict(self) -> JsonObject:
        """Encode this content to canonical wire data."""
        self.validate()
        return cast(JsonObject, _to_wire(self))


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    """Origin and derivation facts for one memory."""

    source: str
    source_ids: tuple[str, ...] = ()
    parent_memory_ids: tuple[str, ...] = ()
    metadata: JsonObject = field(default_factory=dict)

    def validate(self) -> None:
        """Validate that the source classification is present."""
        _validate_identifier("provenance.source", self.source)

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryProvenance:
        """Decode and validate provenance from canonical wire data."""
        provenance = cls(
            source=_require_str(data, "source"),
            source_ids=_string_tuple(data.get("source_ids", [])),
            parent_memory_ids=_string_tuple(data.get("parent_memory_ids", [])),
            metadata=_json_object(data.get("metadata", {})),
        )
        provenance.validate()
        return provenance

    def to_dict(self) -> JsonObject:
        """Encode this provenance to canonical wire data."""
        self.validate()
        return cast(JsonObject, _to_wire(self))


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """Durable provider-neutral memory record."""

    id: str
    provider: str
    namespace: MemoryNamespace
    content: MemoryContent
    event_timestamp: datetime
    ingested_at: datetime
    provenance: MemoryProvenance
    metadata: JsonObject = field(default_factory=dict)
    provider_metadata: JsonObject = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryRecord:
        """Decode a durable record from canonical wire data."""
        return cls(
            id=_require_str(data, "id"),
            provider=_require_str(data, "provider"),
            namespace=MemoryNamespace.from_dict(_require_object(data, "namespace")),
            content=MemoryContent.from_dict(_require_object(data, "content")),
            event_timestamp=_parse_datetime(_require_str(data, "event_timestamp")),
            ingested_at=_parse_datetime(_require_str(data, "ingested_at")),
            provenance=MemoryProvenance.from_dict(_require_object(data, "provenance")),
            metadata=_json_object(data.get("metadata", {})),
            provider_metadata=_json_object(data.get("provider_metadata", {})),
        )

    def to_dict(self) -> JsonObject:
        """Encode this record to canonical wire data."""
        return cast(JsonObject, _to_wire(self))


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    """Ranked memory search result."""

    record: MemoryRecord
    score: float
    rank: int

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryMatch:
        """Decode a ranked match from canonical wire data."""
        return cls(
            record=MemoryRecord.from_dict(_require_object(data, "record")),
            score=float(_require_number(data, "score")),
            rank=_require_int(data, "rank"),
        )

    def to_dict(self) -> JsonObject:
        """Encode this match to canonical wire data."""
        return cast(JsonObject, _to_wire(self))


@dataclass(frozen=True, slots=True)
class MemoryFilter:
    """Exact provider-neutral search filters."""

    metadata: JsonObject = field(default_factory=dict)
    event_after: datetime | None = None
    event_before: datetime | None = None
    ingested_after: datetime | None = None
    ingested_before: datetime | None = None

    def validate(self) -> None:
        """Validate timestamp range ordering."""
        for name, value in (
            ("event_after", self.event_after),
            ("event_before", self.event_before),
            ("ingested_after", self.ingested_after),
            ("ingested_before", self.ingested_before),
        ):
            if value is not None:
                _validate_datetime(name, value)
        _validate_range("event", self.event_after, self.event_before)
        _validate_range("ingested", self.ingested_after, self.ingested_before)

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryFilter:
        """Decode and validate filters from canonical wire data."""
        memory_filter = cls(
            metadata=_json_object(data.get("metadata", {})),
            event_after=_optional_datetime(data.get("event_after")),
            event_before=_optional_datetime(data.get("event_before")),
            ingested_after=_optional_datetime(data.get("ingested_after")),
            ingested_before=_optional_datetime(data.get("ingested_before")),
        )
        memory_filter.validate()
        return memory_filter


@dataclass(frozen=True, slots=True)
class MemoryRequestContext:
    """Correlation and deadline data shared by provider operations."""

    operation_id: str
    deadline: datetime | None = None

    def validate(self) -> None:
        """Validate the operation correlation identifier."""
        _validate_identifier("operation_id", self.operation_id)
        if self.deadline is not None:
            _validate_datetime("deadline", self.deadline)

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryRequestContext:
        """Decode and validate request context from canonical wire data."""
        context = cls(
            operation_id=_require_str(data, "operation_id"),
            deadline=_optional_datetime(data.get("deadline")),
        )
        context.validate()
        return context


@dataclass(frozen=True, slots=True)
class MemorySearchRequest:
    """Provider-neutral memory search request."""

    context: MemoryRequestContext
    namespace: MemoryNamespace
    query: str
    scope: MemorySearchScope = MemorySearchScope.SUBJECT
    filter: MemoryFilter = field(default_factory=MemoryFilter)
    limit: int = DEFAULT_SEARCH_LIMIT

    def validate(self) -> None:
        """Validate identity, query, filters, and result limit."""
        self.context.validate()
        self.scope.validate(self.namespace)
        _validate_identifier("memory search query", self.query)
        if not 1 <= self.limit <= MAX_SEARCH_LIMIT:
            _raise_invalid(f"memory search limit must be in 1..={MAX_SEARCH_LIMIT}")
        self.filter.validate()

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemorySearchRequest:
        """Decode and validate a search request from canonical wire data."""
        request = cls(
            context=MemoryRequestContext.from_dict(_require_object(data, "context")),
            namespace=MemoryNamespace.from_dict(_require_object(data, "namespace")),
            query=_require_str(data, "query"),
            scope=MemorySearchScope(str(data.get("scope", "subject"))),
            filter=MemoryFilter.from_dict(_json_object(data.get("filter", {}))),
            limit=_require_int(data, "limit"),
        )
        request.validate()
        return request

    def to_dict(self) -> JsonObject:
        """Encode this request to canonical wire data."""
        self.validate()
        return cast(JsonObject, _to_wire(self))


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """Successful matches plus non-fatal provider failures."""

    matches: tuple[MemoryMatch, ...] = ()
    partial_errors: tuple[MemoryOperationError, ...] = ()

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemorySearchResult:
        """Decode a search result from canonical wire data."""
        return cls(
            matches=tuple(MemoryMatch.from_dict(_json_object(item)) for item in _list(data.get("matches", []))),
            partial_errors=tuple(
                MemoryOperationError.from_dict(_json_object(item)) for item in _list(data.get("partial_errors", []))
            ),
        )

    def to_dict(self) -> JsonObject:
        """Encode this result to canonical wire data."""
        return cast(JsonObject, _to_wire(self))


@dataclass(frozen=True, slots=True)
class MemoryStoreRequest:
    """Provider-neutral memory store request."""

    context: MemoryRequestContext
    namespace: MemoryNamespace
    content: MemoryContent
    event_timestamp: datetime
    provenance: MemoryProvenance
    metadata: JsonObject = field(default_factory=dict)
    idempotency_key: str | None = None

    def validate(self) -> None:
        """Validate identity, content, provenance, and idempotency key."""
        self.context.validate()
        self.namespace.validate()
        self.content.validate()
        _validate_datetime("event_timestamp", self.event_timestamp)
        self.provenance.validate()
        _validate_optional_identifier("idempotency_key", self.idempotency_key)

    def to_dict(self) -> JsonObject:
        """Encode this request to canonical wire data."""
        self.validate()
        return cast(JsonObject, _to_wire(self))


class MemoryStoreDisposition(StrEnum):
    """Outcome of an idempotent store operation."""

    CREATED = "created"
    EXISTING = "existing"


@dataclass(frozen=True, slots=True)
class MemoryStoreResult:
    """Created or replayed record returned by store."""

    record: MemoryRecord
    disposition: MemoryStoreDisposition

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryStoreResult:
        """Decode a store result from canonical wire data."""
        return cls(
            record=MemoryRecord.from_dict(_require_object(data, "record")),
            disposition=MemoryStoreDisposition(_require_str(data, "disposition")),
        )

    def to_dict(self) -> JsonObject:
        """Encode this store result to canonical wire data."""
        return cast(JsonObject, _to_wire(self))


@dataclass(frozen=True, slots=True)
class MemoryDeleteRequest:
    """Delete request for a provider that advertises delete capability."""

    context: MemoryRequestContext
    namespace: MemoryNamespace
    id: str

    def validate(self) -> None:
        """Validate operation context, identity partition, and record ID."""
        self.context.validate()
        self.namespace.validate()
        _validate_identifier("memory id", self.id)

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryDeleteRequest:
        """Decode and validate a delete request from canonical wire data."""
        request = cls(
            context=MemoryRequestContext.from_dict(_require_object(data, "context")),
            namespace=MemoryNamespace.from_dict(_require_object(data, "namespace")),
            id=_require_str(data, "id"),
        )
        request.validate()
        return request

    def to_dict(self) -> JsonObject:
        """Encode this request to canonical wire data."""
        self.validate()
        return cast(JsonObject, _to_wire(self))


@dataclass(frozen=True, slots=True)
class MemoryDeleteResult:
    """Result of a provider delete operation."""

    id: str
    deleted: bool

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryDeleteResult:
        """Decode a delete result from canonical wire data."""
        return cls(id=_require_str(data, "id"), deleted=_require_bool(data, "deleted"))

    def to_dict(self) -> JsonObject:
        """Encode this result to canonical wire data."""
        return cast(JsonObject, _to_wire(self))


@dataclass(frozen=True, slots=True)
class MemoryCapabilities:
    """Optional capabilities advertised by a memory provider."""

    update: bool = False
    delete: bool = False
    batch_store: bool = False
    maintenance: bool = False
    feedback: bool = False
    health: bool = False

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryCapabilities:
        """Decode capabilities from canonical wire data."""
        return cls(
            update=bool(data.get("update", False)),
            delete=bool(data.get("delete", False)),
            batch_store=bool(data.get("batch_store", False)),
            maintenance=bool(data.get("maintenance", False)),
            feedback=bool(data.get("feedback", False)),
            health=bool(data.get("health", False)),
        )

    def to_dict(self) -> JsonObject:
        """Encode all capability flags to canonical wire data."""
        return cast(JsonObject, _to_wire(self, omit_empty=False))


class MemoryMaintenanceAction(StrEnum):
    """Provider maintenance operation."""

    REFLECT = "reflect"
    CONSOLIDATE = "consolidate"


@dataclass(frozen=True, slots=True)
class MemoryMaintenanceWindow:
    """Bounded source view consumed by one maintenance operation."""

    checkpoint_id: str
    query: str
    limit: int
    previous_checkpoint_id: str | None = None
    scope: MemorySearchScope = MemorySearchScope.SUBJECT
    filter: MemoryFilter = field(default_factory=MemoryFilter)

    def validate(self, namespace: MemoryNamespace) -> None:
        """Validate checkpoint identity and bounded source selection."""
        _validate_identifier("checkpoint_id", self.checkpoint_id)
        _validate_optional_identifier("previous_checkpoint_id", self.previous_checkpoint_id)
        if self.previous_checkpoint_id == self.checkpoint_id:
            _raise_invalid("previous_checkpoint_id must differ from checkpoint_id")
        self.scope.validate(namespace)
        _validate_identifier("maintenance window query", self.query)
        if not 1 <= self.limit <= MAX_SEARCH_LIMIT:
            _raise_invalid(f"maintenance window limit must be in 1..={MAX_SEARCH_LIMIT}")
        self.filter.validate()

    @classmethod
    def from_dict(cls, data: JsonObject, namespace: MemoryNamespace) -> MemoryMaintenanceWindow:
        """Decode and validate a maintenance window from canonical wire data."""
        window = cls(
            checkpoint_id=_require_str(data, "checkpoint_id"),
            previous_checkpoint_id=_optional_str(data.get("previous_checkpoint_id")),
            query=_require_str(data, "query"),
            scope=MemorySearchScope(str(data.get("scope", "subject"))),
            filter=MemoryFilter.from_dict(_json_object(data.get("filter", {}))),
            limit=_require_int(data, "limit"),
        )
        window.validate(namespace)
        return window


@dataclass(frozen=True, slots=True)
class MemoryMaintenanceRequest:
    """Request for a provider that advertises maintenance capability."""

    context: MemoryRequestContext
    namespace: MemoryNamespace
    action: MemoryMaintenanceAction
    window: MemoryMaintenanceWindow | None = None
    parameters: JsonObject = field(default_factory=dict)

    def validate(self) -> None:
        """Validate operation context, identity, and optional source window."""
        self.context.validate()
        self.namespace.validate()
        if self.window is not None:
            self.window.validate(self.namespace)

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryMaintenanceRequest:
        """Decode and validate a maintenance request from canonical wire data."""
        namespace = MemoryNamespace.from_dict(_require_object(data, "namespace"))
        raw_window = data.get("window")
        request = cls(
            context=MemoryRequestContext.from_dict(_require_object(data, "context")),
            namespace=namespace,
            action=MemoryMaintenanceAction(_require_str(data, "action")),
            window=None
            if raw_window is None
            else MemoryMaintenanceWindow.from_dict(_json_object(raw_window), namespace),
            parameters=_json_object(data.get("parameters", {})),
        )
        request.validate()
        return request

    def to_dict(self) -> JsonObject:
        """Encode this request to canonical wire data."""
        self.validate()
        return cast(JsonObject, _to_wire(self))


@dataclass(frozen=True, slots=True)
class MemoryMaintenanceResult:
    """Provider job or immediately committed derived-memory result."""

    job_id: str | None = None
    records: tuple[MemoryRecord, ...] = ()
    partial_errors: tuple[MemoryOperationError, ...] = ()

    @classmethod
    def from_dict(cls, data: JsonObject) -> MemoryMaintenanceResult:
        """Decode a maintenance result from canonical wire data."""
        return cls(
            job_id=_optional_str(data.get("job_id")),
            records=tuple(MemoryRecord.from_dict(_json_object(item)) for item in _list(data.get("records", []))),
            partial_errors=tuple(
                MemoryOperationError.from_dict(_json_object(item)) for item in _list(data.get("partial_errors", []))
            ),
        )

    def to_dict(self) -> JsonObject:
        """Encode this maintenance result to canonical wire data."""
        return cast(JsonObject, _to_wire(self))


@runtime_checkable
class MemoryProvider(Protocol):
    """Required async contract implemented by Python memory adapters."""

    @property
    def name(self) -> str:
        """Return the stable provider name."""
        ...

    @property
    def capabilities(self) -> MemoryCapabilities:
        """Return optional capabilities supported by this provider."""
        ...

    async def search(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Search for memories inside the request namespace."""
        ...

    async def store(self, request: MemoryStoreRequest) -> MemoryStoreResult:
        """Store or idempotently replay one memory."""
        ...


def _to_wire(value: object, *, omit_empty: bool = True) -> Json:
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, datetime):
        return _format_datetime(value)
    if is_dataclass(value) and not isinstance(value, type):
        output: JsonObject = {}
        for field_info in fields(value):
            field_value = getattr(value, field_info.name)
            if field_value is None:
                if isinstance(value, MemoryContent) and value.kind == "json" and field_info.name == "value":
                    output[field_info.name] = None
                continue
            if omit_empty and field_value in ((), [], {}):
                continue
            output[field_info.name] = _to_wire(field_value, omit_empty=omit_empty)
        return output
    if isinstance(value, tuple | list):
        return [_to_wire(item, omit_empty=omit_empty) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_wire(item, omit_empty=False) for key, item in value.items()}
    return cast(Json, value)


def _format_datetime(value: datetime) -> str:
    _validate_datetime("memory timestamp", value)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise MemoryContractError(
            MemoryOperationError(MemoryErrorCode.INVALID_REQUEST, f"invalid RFC3339 timestamp: {value}")
        ) from error
    if parsed.tzinfo is None:
        _raise_invalid("memory timestamps must include a timezone")
    return parsed


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _raise_invalid("memory timestamp must be a string")
    return _parse_datetime(value)


def _validate_identifier(name: str, value: str) -> None:
    if not value.strip():
        _raise_invalid(f"{name} must not be empty")


def _validate_optional_identifier(name: str, value: str | None) -> None:
    if value is not None:
        _validate_identifier(name, value)


def _validate_datetime(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        _raise_invalid(f"{name} must include a timezone")


def _validate_range(name: str, after: datetime | None, before: datetime | None) -> None:
    if after is not None and before is not None and after > before:
        _raise_invalid(f"{name}_after must not be later than {name}_before")


def _raise_invalid(message: str) -> NoReturn:
    raise MemoryContractError(MemoryOperationError(MemoryErrorCode.INVALID_REQUEST, message))


def _require_str(data: JsonObject, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        _raise_invalid(f"{key} must be a string")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _raise_invalid("optional identifier must be a string")
    return value


def _require_object(data: JsonObject, key: str) -> JsonObject:
    return _json_object(data.get(key))


def _json_object(value: object) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _raise_invalid("memory wire value must be an object with string keys")
    return cast(JsonObject, value)


def _list(value: object) -> list[Json]:
    if not isinstance(value, list):
        _raise_invalid("memory wire value must be a list")
    return cast(list[Json], value)


def _string_tuple(value: object) -> tuple[str, ...]:
    values = _list(value)
    if not all(isinstance(item, str) for item in values):
        _raise_invalid("memory identifier list must contain strings")
    return tuple(cast(list[str], values))


def _require_number(data: JsonObject, key: str) -> float | int:
    value = data.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        _raise_invalid(f"{key} must be a number")
    return value


def _require_int(data: JsonObject, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _raise_invalid(f"{key} must be an integer")
    return value


def _require_bool(data: JsonObject, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        _raise_invalid(f"{key} must be a boolean")
    return value


__all__ = [
    "AutomaticMemoryConfig",
    "DEFAULT_SEARCH_LIMIT",
    "EvidenceMode",
    "FailurePolicy",
    "InMemoryAutomaticMemory",
    "MAX_SEARCH_LIMIT",
    "MemoryCapabilities",
    "MemoryContent",
    "MemoryContractError",
    "MemoryDeleteRequest",
    "MemoryDeleteResult",
    "MemoryErrorCode",
    "MemoryFilter",
    "MemoryMaintenanceAction",
    "MemoryMaintenanceRequest",
    "MemoryMaintenanceResult",
    "MemoryMaintenanceWindow",
    "MemoryMatch",
    "MemoryBackpressurePolicy",
    "MemoryNamespace",
    "MemoryOperationError",
    "MemoryProvenance",
    "MemoryProvider",
    "MemoryProviderError",
    "MemoryRecord",
    "MemoryRequestContext",
    "MemorySearchRequest",
    "MemorySearchResult",
    "MemorySearchScope",
    "MemoryStoreDisposition",
    "MemoryStoreRequest",
    "MemoryStoreResult",
    "MemoryWorkQueueConfig",
    "WriteDelivery",
    "WriteProjection",
]
