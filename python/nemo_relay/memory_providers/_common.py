# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, dependency-free helpers for optional memory provider adapters."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from hashlib import sha256
from typing import NoReturn, TypeVar, cast

from nemo_relay import Json, JsonObject
from nemo_relay.memory import (
    MemoryContent,
    MemoryErrorCode,
    MemoryNamespace,
    MemoryOperationError,
    MemoryProviderError,
    MemoryRecord,
    MemorySearchRequest,
    MemorySearchScope,
    MemoryStoreDisposition,
    MemoryStoreRequest,
    MemoryStoreResult,
)

PARTITION_VERSION = "relay-memory-v1"
RECORD_ENVELOPE_VERSION = "0.1"
RECORD_METADATA_KEY = "_nemo_relay_record_v1"
RECORD_ID_METADATA_KEY = "_nemo_relay_record_id"
AGENT_TAG_PREFIX = "nemo-relay-agent-v1-"
SESSION_TAG_PREFIX = "nemo-relay-session-v1-"

_T = TypeVar("_T")


def vendor_partition(namespace: MemoryNamespace) -> str:
    """Return an opaque partition derived from both tenant and subject."""
    namespace.validate()
    material = _canonical_json([PARTITION_VERSION, namespace.tenant_id, namespace.subject_id])
    return f"{PARTITION_VERSION}-{sha256(material.encode()).hexdigest()}"


def scope_tags(namespace: MemoryNamespace) -> tuple[str, ...]:
    """Return opaque tags for optional agent and session narrowing."""
    namespace.validate()
    tags: list[str] = []
    if namespace.agent_id is not None:
        tags.append(_identity_tag(AGENT_TAG_PREFIX, namespace.agent_id))
    if namespace.session_id is not None:
        tags.append(_identity_tag(SESSION_TAG_PREFIX, namespace.session_id))
    return tuple(tags)


def required_scope_tags(request: MemorySearchRequest) -> tuple[str, ...]:
    """Return tags that can safely narrow the requested search scope."""
    request.validate()
    namespace = request.namespace
    if request.scope is MemorySearchScope.AGENT:
        return (_identity_tag(AGENT_TAG_PREFIX, cast(str, namespace.agent_id)),)
    if request.scope is MemorySearchScope.SESSION:
        return (_identity_tag(SESSION_TAG_PREFIX, cast(str, namespace.session_id)),)
    if request.scope is MemorySearchScope.EXACT:
        return scope_tags(namespace)
    return ()


def deterministic_record_id(provider: str, request: MemoryStoreRequest) -> str:
    """Create a stable Relay-side ID before a remote provider allocates one."""
    request.validate()
    seed: object = (
        ["idempotency", request.idempotency_key]
        if request.idempotency_key is not None
        else ["operation", request.context.operation_id, _store_fingerprint(request)]
    )
    material = _canonical_json([RECORD_ENVELOPE_VERSION, provider, vendor_partition(request.namespace), seed])
    return f"{provider}-{sha256(material.encode()).hexdigest()[:32]}"


def prepare_record(
    provider: str,
    request: MemoryStoreRequest,
    *,
    ingested_at: datetime | None = None,
) -> tuple[MemoryRecord, dict[str, str]]:
    """Build a canonical Relay record and its reserved vendor metadata."""
    request.validate()
    timestamp = ingested_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise_provider_error(
            provider,
            request.context.operation_id,
            MemoryErrorCode.INTERNAL,
            "adapter generated a timezone-naive ingestion timestamp",
        )
    record = MemoryRecord(
        id=deterministic_record_id(provider, request),
        provider=provider,
        namespace=request.namespace,
        content=request.content,
        event_timestamp=request.event_timestamp,
        ingested_at=timestamp,
        provenance=request.provenance,
        metadata=request.metadata,
    )
    # DTOs are frozen but their JSON containers are not. Round-trip once so a
    # caller cannot mutate the durable record through request-owned dictionaries.
    record = MemoryRecord.from_dict(record.to_dict())
    envelope = {
        "version": RECORD_ENVELOPE_VERSION,
        "record": record.to_dict(),
    }
    return record, {
        RECORD_METADATA_KEY: _canonical_json(envelope),
        RECORD_ID_METADATA_KEY: record.id,
    }


def reconstruct_record(
    provider: str,
    operation_id: str,
    metadata: Mapping[str, object] | None,
    *,
    provider_metadata: JsonObject | None = None,
) -> MemoryRecord:
    """Rebuild a Relay record from its reserved vendor metadata envelope."""
    raw = None if metadata is None else metadata.get(RECORD_METADATA_KEY)
    if not isinstance(raw, str):
        raise_provider_error(
            provider,
            operation_id,
            MemoryErrorCode.INTERNAL,
            "provider result is missing Relay record metadata",
        )
    try:
        envelope = json.loads(raw)
        if not isinstance(envelope, dict) or envelope.get("version") != RECORD_ENVELOPE_VERSION:
            raise ValueError("unsupported record envelope")
        record_data = envelope.get("record")
        if not isinstance(record_data, dict):
            raise ValueError("missing record")
        record = MemoryRecord.from_dict(cast(JsonObject, record_data))
    except (TypeError, ValueError) as error:
        raise_provider_error(
            provider,
            operation_id,
            MemoryErrorCode.INTERNAL,
            "provider returned invalid Relay record metadata",
            details={"error_type": type(error).__name__},
        )
    if record.provider != provider:
        raise_provider_error(
            provider,
            operation_id,
            MemoryErrorCode.INTERNAL,
            "provider result contains a record for another adapter",
        )
    return record_with_provider_metadata(record, provider_metadata or {})


def record_with_provider_metadata(record: MemoryRecord, provider_metadata: JsonObject) -> MemoryRecord:
    """Return a record with stable provider-owned identity facts attached."""
    return MemoryRecord(
        id=record.id,
        provider=record.provider,
        namespace=record.namespace,
        content=record.content,
        event_timestamp=record.event_timestamp,
        ingested_at=record.ingested_at,
        provenance=record.provenance,
        metadata=record.metadata,
        provider_metadata=provider_metadata,
    )


def content_search_text(content: MemoryContent) -> str:
    """Convert provider-neutral content to a deterministic searchable string."""
    content.validate()
    if content.kind == "text":
        return cast(str, content.text)
    if content.kind == "json":
        return _canonical_json(content.value)
    return content.preview or cast(str, content.reference)


def record_matches(record: MemoryRecord, request: MemorySearchRequest) -> bool:
    """Apply the exact Relay namespace, metadata, and temporal predicates."""
    request.validate()
    query = request.namespace
    actual = record.namespace
    if actual.tenant_id != query.tenant_id or actual.subject_id != query.subject_id:
        return False
    if request.scope is MemorySearchScope.AGENT and actual.agent_id != query.agent_id:
        return False
    if request.scope is MemorySearchScope.SESSION and actual.session_id != query.session_id:
        return False
    if request.scope is MemorySearchScope.EXACT:
        if query.agent_id is not None and actual.agent_id != query.agent_id:
            return False
        if query.session_id is not None and actual.session_id != query.session_id:
            return False
    memory_filter = request.filter
    if any(record.metadata.get(key) != value for key, value in memory_filter.metadata.items()):
        return False
    return (
        _after(record.event_timestamp, memory_filter.event_after)
        and _before(record.event_timestamp, memory_filter.event_before)
        and _after(record.ingested_at, memory_filter.ingested_after)
        and _before(record.ingested_at, memory_filter.ingested_before)
    )


async def await_vendor(
    awaitable: Awaitable[_T],
    *,
    provider: str,
    operation_id: str,
    deadline: datetime | None,
) -> _T:
    """Await a vendor call within the Relay deadline and normalize failures."""
    try:
        if deadline is None:
            return await awaitable
        if deadline.tzinfo is None:
            raise_provider_error(
                provider,
                operation_id,
                MemoryErrorCode.INVALID_REQUEST,
                "memory deadline must include a timezone",
            )
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            close = getattr(awaitable, "close", None)
            if callable(close):
                cast(Callable[[], object], close)()
            raise_provider_error(
                provider,
                operation_id,
                MemoryErrorCode.DEADLINE_EXCEEDED,
                "memory provider deadline exceeded",
                retryable=True,
            )
        async with asyncio.timeout(remaining):
            return await awaitable
    except asyncio.CancelledError:
        raise
    except MemoryProviderError:
        raise
    except TimeoutError:
        raise_provider_error(
            provider,
            operation_id,
            MemoryErrorCode.DEADLINE_EXCEEDED,
            "memory provider deadline exceeded",
            retryable=True,
        )
    except Exception as error:
        raise map_vendor_error(provider, operation_id, error) from error


def map_vendor_error(provider: str, operation_id: str, error: Exception) -> MemoryProviderError:
    """Map common SDK/HTTP failure facts without exposing response content."""
    status = _status_code(error)
    error_name = type(error).__name__
    if isinstance(error, ValueError) or error_name in {
        "CogneeValidationError",
        "ConfigurationError",
        "DatasetNotFoundError",
        "ValidationError",
    }:
        code, retryable = MemoryErrorCode.INVALID_REQUEST, False
    elif error_name == "RateLimitError":
        code, retryable = MemoryErrorCode.PROVIDER_UNAVAILABLE, True
    elif error_name in {
        "AuthenticationError",
        "PermissionDeniedError",
        "UnauthorizedDataAccessError",
        "UserNotFoundError",
    }:
        code, retryable = MemoryErrorCode.PROVIDER_UNAVAILABLE, False
    elif error_name in {
        "DatabaseError",
        "DatabaseNotCreatedError",
        "EmbeddingError",
        "LLMError",
        "NetworkError",
        "VectorSearchError",
        "VectorStoreError",
    }:
        code, retryable = MemoryErrorCode.PROVIDER_UNAVAILABLE, True
    elif status in {400, 404, 405, 422}:
        code, retryable = MemoryErrorCode.INVALID_REQUEST, False
    elif status in {401, 403}:
        code, retryable = MemoryErrorCode.PROVIDER_UNAVAILABLE, False
    elif status == 409:
        code, retryable = MemoryErrorCode.CONFLICT, False
    elif status in {408, 504}:
        code, retryable = MemoryErrorCode.DEADLINE_EXCEEDED, True
    elif status == 429 or (status is not None and status >= 500):
        code, retryable = MemoryErrorCode.PROVIDER_UNAVAILABLE, True
    else:
        code, retryable = MemoryErrorCode.INTERNAL, False
    details: JsonObject = {"error_type": error_name}
    if status is not None:
        details["status_code"] = status
    return MemoryProviderError(
        MemoryOperationError(
            code=code,
            message=f"{provider} provider operation failed",
            retryable=retryable,
            operation_id=operation_id,
            provider=provider,
            details=details,
        )
    )


def raise_provider_error(
    provider: str,
    operation_id: str,
    code: MemoryErrorCode,
    message: str,
    *,
    retryable: bool = False,
    details: JsonObject | None = None,
) -> NoReturn:
    """Raise one canonical provider error."""
    raise MemoryProviderError(
        MemoryOperationError(
            code=code,
            message=message,
            retryable=retryable,
            operation_id=operation_id,
            provider=provider,
            details=details or {},
        )
    )


class IdempotencyLedger:
    """Bounded exact replay ledger scoped to one adapter process lifetime."""

    def __init__(self, capacity: int = 1_024) -> None:
        if capacity < 1:
            raise ValueError("idempotency ledger capacity must be positive")
        self._capacity = capacity
        self._entries: OrderedDict[tuple[str, str], tuple[str, MemoryStoreResult]] = OrderedDict()
        self._locks: dict[tuple[str, str], tuple[asyncio.Lock, int]] = {}
        self._registry_lock = asyncio.Lock()

    async def run(
        self,
        request: MemoryStoreRequest,
        mutation: Callable[[], Awaitable[MemoryStoreResult]],
    ) -> MemoryStoreResult:
        """Run a mutation once per namespace/key and detect conflicting reuse."""
        request.validate()
        if request.idempotency_key is None:
            return await mutation()
        key = (vendor_partition(request.namespace), request.idempotency_key)
        fingerprint = _store_fingerprint(request)
        lock = await self._lock_for(key)
        try:
            async with lock:
                existing = self._entries.get(key)
                if existing is not None:
                    old_fingerprint, old_result = existing
                    self._entries.move_to_end(key)
                    if old_fingerprint != fingerprint:
                        raise_provider_error(
                            old_result.record.provider,
                            request.context.operation_id,
                            MemoryErrorCode.CONFLICT,
                            "idempotency key was reused with a different memory payload",
                        )
                    return MemoryStoreResult(old_result.record, MemoryStoreDisposition.EXISTING)
                result = await mutation()
                self._entries[key] = (fingerprint, result)
                self._entries.move_to_end(key)
                while len(self._entries) > self._capacity:
                    self._entries.popitem(last=False)
                return result
        finally:
            await self._release_lock(key, lock)

    async def invalidate_record(self, record_id: str) -> None:
        """Remove replay entries for a successfully deleted record.

        Existing and in-flight idempotent keys are pinned before inspection so
        an in-flight store cannot publish a stale replay entry after deletion.
        """
        if not record_id.strip():
            raise ValueError("record_id must not be empty")
        async with self._registry_lock:
            keys = tuple(dict.fromkeys((*self._entries.keys(), *self._locks.keys())))
            pinned: list[tuple[tuple[str, str], asyncio.Lock]] = []
            for key in keys:
                current = self._locks.get(key)
                if current is None:
                    lock = asyncio.Lock()
                    self._locks[key] = (lock, 1)
                else:
                    lock, users = current
                    self._locks[key] = (lock, users + 1)
                pinned.append((key, lock))
        try:
            for key, lock in pinned:
                async with lock:
                    entry = self._entries.get(key)
                    if entry is not None and entry[1].record.id == record_id:
                        self._entries.pop(key, None)
        finally:
            for key, lock in pinned:
                await self._release_lock(key, lock)

    async def _lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        async with self._registry_lock:
            current = self._locks.get(key)
            if current is None:
                lock = asyncio.Lock()
                self._locks[key] = (lock, 1)
                return lock
            lock, users = current
            self._locks[key] = (lock, users + 1)
            return lock

    async def _release_lock(self, key: tuple[str, str], lock: asyncio.Lock) -> None:
        async with self._registry_lock:
            current = self._locks.get(key)
            if current is None or current[0] is not lock:
                return
            users = current[1] - 1
            if users == 0:
                self._locks.pop(key, None)
            else:
                self._locks[key] = (lock, users)


def _store_fingerprint(request: MemoryStoreRequest) -> str:
    data = request.to_dict()
    payload = {
        "content": data["content"],
        "event_timestamp": data["event_timestamp"],
        "provenance": data["provenance"],
        "metadata": data.get("metadata", {}),
    }
    return sha256(_canonical_json(payload).encode()).hexdigest()


def _identity_tag(prefix: str, value: str) -> str:
    return f"{prefix}{sha256(_canonical_json([value]).encode()).hexdigest()}"


def _canonical_json(value: Json | object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _status_code(error: Exception) -> int | None:
    for name in ("status", "status_code"):
        value = getattr(error, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _after(value: datetime, boundary: datetime | None) -> bool:
    return boundary is None or value >= boundary


def _before(value: datetime, boundary: datetime | None) -> bool:
    return boundary is None or value <= boundary
