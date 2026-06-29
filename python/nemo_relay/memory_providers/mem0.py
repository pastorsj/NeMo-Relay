# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Mem0 OSS 2.x adapter for the Relay memory provider contract."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any, NoReturn, Protocol, cast

from nemo_relay import Json
from nemo_relay.memory import (
    MemoryCapabilities,
    MemoryErrorCode,
    MemoryMatch,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryStoreDisposition,
    MemoryStoreRequest,
    MemoryStoreResult,
)
from nemo_relay.memory_providers._common import (
    AGENT_TAG_PREFIX,
    SESSION_TAG_PREFIX,
    IdempotencyLedger,
    await_vendor,
    content_search_text,
    prepare_record,
    raise_provider_error,
    reconstruct_record,
    record_matches,
    record_with_provider_metadata,
    required_scope_tags,
    scope_tags,
    vendor_partition,
)

MEM0_MIN_VERSION = "2.0.8"
MEM0_MAX_VERSION = "3.0.0"
_FILTER_PREFIX = "_nemo_relay_filter_v1_"
_AGENT_METADATA_KEY = "_nemo_relay_agent_v1"
_SESSION_METADATA_KEY = "_nemo_relay_session_v1"


class Mem0AsyncClient(Protocol):
    """Subset of ``mem0.AsyncMemory`` consumed by the adapter."""

    async def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        metadata: dict[str, Any],
        infer: bool,
    ) -> object: ...

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any],
        threshold: float,
        explain: bool,
    ) -> object: ...


class Mem0MemoryProvider:
    """Relay provider profile for Mem0 OSS ``AsyncMemory`` 2.x.

    The profile uses ``infer=False`` so one Relay store maps to one raw Mem0
    ADD. Exact idempotency is bounded to this adapter instance's process.
    """

    name = "mem0"
    capabilities = MemoryCapabilities()

    def __init__(
        self,
        client: Mem0AsyncClient | None = None,
        *,
        config: Mapping[str, object] | None = None,
        idempotency_capacity: int = 1_024,
        overfetch_factor: int = 4,
    ) -> None:
        if client is not None and config is not None:
            raise ValueError("pass either a Mem0 client or config, not both")
        if overfetch_factor < 1:
            raise ValueError("overfetch_factor must be positive")
        self._client = client or _default_client(config)
        self._ledger = IdempotencyLedger(idempotency_capacity)
        self._overfetch_factor = overfetch_factor

    async def store(self, request: MemoryStoreRequest) -> MemoryStoreResult:
        """Store one raw Relay record in Mem0."""
        request.validate()

        async def mutation() -> MemoryStoreResult:
            record, reserved = prepare_record(self.name, request)
            metadata: dict[str, Any] = dict(reserved)
            tags = scope_tags(request.namespace)
            for tag in tags:
                if tag.startswith(AGENT_TAG_PREFIX):
                    metadata[_AGENT_METADATA_KEY] = tag
                elif tag.startswith(SESSION_TAG_PREFIX):
                    metadata[_SESSION_METADATA_KEY] = tag
            metadata.update(_mem0_filter_metadata(request.metadata))
            response = await await_vendor(
                self._client.add(
                    [{"role": "user", "content": _searchable_text(request)}],
                    user_id=vendor_partition(request.namespace),
                    metadata=metadata,
                    infer=False,
                ),
                provider=self.name,
                operation_id=request.context.operation_id,
                deadline=request.context.deadline,
            )
            results = _result_list(response, "add", request.context.operation_id)
            if len(results) != 1 or not isinstance(results[0].get("id"), str):
                _invalid_response(request.context.operation_id, "Mem0 add did not return exactly one memory ID")
            vendor_id = cast(str, results[0]["id"])
            record = record_with_provider_metadata(record, {"vendor_id": vendor_id})
            return MemoryStoreResult(record, MemoryStoreDisposition.CREATED)

        return await self._ledger.run(request, mutation)

    async def search(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Search one opaque Mem0 tenant/subject partition."""
        request.validate()
        filters: dict[str, Any] = {"user_id": vendor_partition(request.namespace)}
        required_tags = required_scope_tags(request)
        for tag in required_tags:
            if tag.startswith(AGENT_TAG_PREFIX):
                filters[_AGENT_METADATA_KEY] = tag
            elif tag.startswith(SESSION_TAG_PREFIX):
                filters[_SESSION_METADATA_KEY] = tag
        filters.update(_mem0_filter_metadata(request.filter.metadata))
        top_k = min(1_000, max(request.limit, request.limit * self._overfetch_factor))
        response = await await_vendor(
            self._client.search(
                request.query,
                top_k=top_k,
                filters=filters,
                threshold=0.0,
                explain=True,
            ),
            provider=self.name,
            operation_id=request.context.operation_id,
            deadline=request.context.deadline,
        )
        items = _result_list(response, "search", request.context.operation_id)
        matches: list[MemoryMatch] = []
        for item in items:
            nested = item.get("metadata")
            metadata: dict[str, object] = (
                dict(cast(Mapping[str, object], nested)) if isinstance(nested, Mapping) else {}
            )
            for key, value in item.items():
                if isinstance(key, str) and key.startswith("_nemo_relay_"):
                    metadata.setdefault(key, value)
            vendor_id = item.get("id")
            if not isinstance(vendor_id, str):
                _invalid_response(request.context.operation_id, "Mem0 search result is missing an ID")
            record = reconstruct_record(
                self.name,
                request.context.operation_id,
                metadata,
                provider_metadata={"vendor_id": vendor_id},
            )
            if not record_matches(record, request):
                continue
            score = item.get("score", 0.0)
            if not isinstance(score, int | float) or isinstance(score, bool):
                _invalid_response(request.context.operation_id, "Mem0 search score is not numeric")
            matches.append(MemoryMatch(record=record, score=float(cast(int | float, score)), rank=len(matches) + 1))
            if len(matches) == request.limit:
                break
        return MemorySearchResult(tuple(matches))


def _default_client(config: Mapping[str, object] | None) -> Mem0AsyncClient:
    try:
        from mem0 import AsyncMemory  # ty: ignore[unresolved-import]
    except ImportError as error:
        raise ImportError("Mem0 support requires `pip install 'nemo-relay[mem0]'`") from error
    if config is None:
        return cast(Mem0AsyncClient, AsyncMemory())
    return cast(Mem0AsyncClient, AsyncMemory.from_config(dict(config)))


def _searchable_text(request: MemoryStoreRequest) -> str:
    return content_search_text(request.content)


def _mem0_filter_metadata(metadata: Mapping[str, Json]) -> dict[str, Json]:
    output: dict[str, Json] = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool) or value is None:
            digest = sha256(key.encode()).hexdigest()
            output[f"{_FILTER_PREFIX}{digest}"] = value
    return output


def _result_list(response: object, operation: str, operation_id: str) -> list[dict[str, object]]:
    if not isinstance(response, Mapping):
        _invalid_response(operation_id, f"Mem0 {operation} response is not an object")
    response_mapping = cast(Mapping[str, object], response)
    raw = response_mapping.get("results")
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        _invalid_response(operation_id, f"Mem0 {operation} response has no results list")
    return [dict(cast(Mapping[str, object], item)) for item in raw]


def _invalid_response(operation_id: str, message: str) -> NoReturn:
    raise_provider_error("mem0", operation_id, MemoryErrorCode.INTERNAL, message)


__all__ = ["MEM0_MAX_VERSION", "MEM0_MIN_VERSION", "Mem0AsyncClient", "Mem0MemoryProvider"]
