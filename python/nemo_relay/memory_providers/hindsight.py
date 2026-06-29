# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Hindsight 0.8 adapter for Relay memory and Reflect maintenance."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import NoReturn, Protocol, cast

from nemo_relay.memory import (
    MemoryCapabilities,
    MemoryContent,
    MemoryErrorCode,
    MemoryFilter,
    MemoryMaintenanceAction,
    MemoryMaintenanceRequest,
    MemoryMaintenanceResult,
    MemoryMatch,
    MemoryProvenance,
    MemoryRequestContext,
    MemorySearchRequest,
    MemorySearchResult,
    MemorySearchScope,
    MemoryStoreDisposition,
    MemoryStoreRequest,
    MemoryStoreResult,
)
from nemo_relay.memory_providers._common import (
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

HINDSIGHT_MIN_VERSION = "0.8.3"
HINDSIGHT_MAX_VERSION = "0.9.0"
REFLECTION_ARTIFACT_VERSION = "0.1"


class HindsightAsyncClient(Protocol):
    """Subset of ``hindsight_client.Hindsight`` consumed by the adapter."""

    async def aretain(
        self,
        bank_id: str,
        content: str,
        *,
        timestamp: datetime | None,
        context: str | None,
        document_id: str | None,
        metadata: dict[str, str] | None,
        tags: list[str] | None,
        update_mode: str | None,
        retain_async: bool,
    ) -> object: ...

    async def arecall(
        self,
        bank_id: str,
        query: str,
        *,
        max_tokens: int,
        budget: str,
        trace: bool,
        include_source_facts: bool,
        tags: list[str] | None,
        tags_match: str,
        prefer_observations: bool,
    ) -> object: ...

    async def areflect(
        self,
        bank_id: str,
        query: str,
        *,
        budget: str,
        context: str | None,
        max_tokens: int | None,
        tags: list[str] | None,
        tags_match: str,
        include_facts: bool,
        include_tool_calls: bool,
        include_tool_call_output: bool,
    ) -> object: ...


class HindsightMemoryProvider:
    """Relay provider profile for the Hindsight 0.8 service client."""

    name = "hindsight"
    capabilities = MemoryCapabilities(maintenance=True)

    def __init__(
        self,
        client: HindsightAsyncClient | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        idempotency_capacity: int = 1_024,
        recall_tokens_per_item: int = 256,
    ) -> None:
        if client is not None and (base_url is not None or api_key is not None):
            raise ValueError("pass either a Hindsight client or connection settings, not both")
        if recall_tokens_per_item < 1:
            raise ValueError("recall_tokens_per_item must be positive")
        self._client = client if client is not None else _default_client(base_url, api_key, timeout)
        self._ledger = IdempotencyLedger(idempotency_capacity)
        self._recall_tokens_per_item = recall_tokens_per_item

    async def store(self, request: MemoryStoreRequest) -> MemoryStoreResult:
        """Retain one Relay record and wait for Hindsight processing."""
        request.validate()

        async def mutation() -> MemoryStoreResult:
            record, reserved = prepare_record(self.name, request)
            response = await await_vendor(
                self._client.aretain(
                    vendor_partition(request.namespace),
                    content_search_text(request.content),
                    timestamp=request.event_timestamp,
                    context=request.provenance.source,
                    document_id=record.id,
                    metadata=reserved,
                    tags=list(scope_tags(request.namespace)) or None,
                    update_mode="replace",
                    retain_async=False,
                ),
                provider=self.name,
                operation_id=request.context.operation_id,
                deadline=request.context.deadline,
            )
            if _field(response, "success") is not True:
                _invalid_response(request.context.operation_id, "Hindsight retain did not report success")
            if _field(response, "async", "var_async") is not False:
                _invalid_response(request.context.operation_id, "Hindsight retain unexpectedly queued work")
            record = record_with_provider_metadata(record, {"document_id": record.id})
            return MemoryStoreResult(record, MemoryStoreDisposition.CREATED)

        return await self._ledger.run(request, mutation)

    async def search(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Recall from one tenant/subject bank and enforce Relay filters."""
        request.validate()
        items = await self._recall(request)
        matches: list[MemoryMatch] = []
        seen_record_ids: set[str] = set()
        for item in items:
            metadata = _mapping_field(item, "metadata")
            document_id = _field(item, "document_id")
            if document_id is not None and not isinstance(document_id, str):
                _invalid_response(request.context.operation_id, "Hindsight document ID is not a string")
            record = reconstruct_record(
                self.name,
                request.context.operation_id,
                metadata,
                provider_metadata={"document_id": document_id or _record_id(metadata, request.context.operation_id)},
            )
            if not record_matches(record, request):
                continue
            if record.id in seen_record_ids:
                continue
            score = _final_score(item, request.context.operation_id)
            seen_record_ids.add(record.id)
            matches.append(MemoryMatch(record=record, score=score, rank=len(matches) + 1))
            if len(matches) == request.limit:
                break
        return MemorySearchResult(tuple(matches))

    async def maintain(self, request: MemoryMaintenanceRequest) -> MemoryMaintenanceResult:
        """Reflect over one bounded bank view and retain the derived memory."""
        request.validate()
        if request.action is not MemoryMaintenanceAction.REFLECT:
            raise_provider_error(
                self.name,
                request.context.operation_id,
                MemoryErrorCode.UNSUPPORTED,
                "Hindsight adapter supports reflect maintenance only",
            )
        if request.window is not None:
            raise_provider_error(
                self.name,
                request.context.operation_id,
                MemoryErrorCode.UNSUPPORTED,
                "Hindsight reflect cannot enforce a bounded Relay maintenance window",
            )
        query = request.parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            raise_provider_error(
                self.name,
                request.context.operation_id,
                MemoryErrorCode.INVALID_REQUEST,
                "Hindsight reflect requires a maintenance window or string query parameter",
            )
        search_request = MemorySearchRequest(
            context=request.context,
            namespace=request.namespace,
            query=query,
            scope=_maintenance_scope(request.parameters.get("scope", "subject"), request.context.operation_id),
            filter=MemoryFilter(),
            limit=_maintenance_limit(request.parameters.get("limit", 20), request.context.operation_id),
        )
        search_request.validate()
        tags = list(required_scope_tags(search_request)) or None
        budget = request.parameters.get("budget", "low")
        if not isinstance(budget, str) or budget not in {"low", "mid", "high"}:
            raise_provider_error(
                self.name,
                request.context.operation_id,
                MemoryErrorCode.INVALID_REQUEST,
                "Hindsight reflect budget must be low, mid, or high",
            )
        context = request.parameters.get("context")
        if context is not None and not isinstance(context, str):
            raise_provider_error(
                self.name,
                request.context.operation_id,
                MemoryErrorCode.INVALID_REQUEST,
                "Hindsight reflect context must be a string",
            )
        max_tokens = request.parameters.get("max_tokens")
        if max_tokens is not None and (
            not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1
        ):
            raise_provider_error(
                self.name,
                request.context.operation_id,
                MemoryErrorCode.INVALID_REQUEST,
                "Hindsight reflect max_tokens must be a positive integer",
            )
        response = await await_vendor(
            self._client.areflect(
                vendor_partition(request.namespace),
                query,
                budget=budget,
                context=context,
                max_tokens=cast(int | None, max_tokens),
                tags=tags,
                tags_match=_tags_match(tags),
                include_facts=True,
                include_tool_calls=False,
                include_tool_call_output=False,
            ),
            provider=self.name,
            operation_id=request.context.operation_id,
            deadline=request.context.deadline,
        )
        text = _field(response, "text")
        if not isinstance(text, str) or not text.strip():
            _invalid_response(request.context.operation_id, "Hindsight reflect response has no text")
        fact_ids = _reflect_fact_ids(response)
        parent_ids = await self._relay_parent_ids(search_request, set(fact_ids))
        checkpoint = request.parameters.get("checkpoint_id", request.context.operation_id)
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise_provider_error(
                self.name,
                request.context.operation_id,
                MemoryErrorCode.INVALID_REQUEST,
                "Hindsight reflect checkpoint_id must be a nonempty string",
            )
        stored = await self.store(
            MemoryStoreRequest(
                context=MemoryRequestContext(f"{request.context.operation_id}:store", request.context.deadline),
                namespace=request.namespace,
                content=MemoryContent.text_content(text),
                event_timestamp=datetime.now(timezone.utc),
                provenance=MemoryProvenance(
                    source="maintenance",
                    source_ids=(checkpoint, *(f"hindsight:{fact_id}" for fact_id in fact_ids)),
                    parent_memory_ids=tuple(parent_ids),
                    metadata={
                        "action": "reflect",
                        "artifact_version": REFLECTION_ARTIFACT_VERSION,
                        "maintainer": self.name,
                    },
                ),
                metadata={
                    "relay_derived": True,
                    "artifact_version": REFLECTION_ARTIFACT_VERSION,
                    "hindsight_evidence_count": len(fact_ids),
                },
                idempotency_key=f"hindsight-reflect:{checkpoint}:{REFLECTION_ARTIFACT_VERSION}",
            )
        )
        return MemoryMaintenanceResult(records=(stored.record,))

    async def _recall(self, request: MemorySearchRequest) -> list[object]:
        tags = list(required_scope_tags(request)) or None
        response = await await_vendor(
            self._client.arecall(
                vendor_partition(request.namespace),
                request.query,
                max_tokens=max(512, request.limit * self._recall_tokens_per_item),
                budget="mid",
                trace=False,
                include_source_facts=True,
                tags=tags,
                tags_match=_tags_match(tags),
                prefer_observations=False,
            ),
            provider=self.name,
            operation_id=request.context.operation_id,
            deadline=request.context.deadline,
        )
        results = _field(response, "results")
        if not isinstance(results, list):
            _invalid_response(request.context.operation_id, "Hindsight recall response has no results list")
        return cast(list[object], results)

    async def _relay_parent_ids(self, request: MemorySearchRequest, fact_ids: set[str]) -> list[str]:
        if not fact_ids:
            return []
        parents: list[str] = []
        for item in await self._recall(request):
            fact_id = _field(item, "id")
            if fact_id not in fact_ids:
                continue
            try:
                record = reconstruct_record(
                    self.name,
                    request.context.operation_id,
                    _mapping_field(item, "metadata"),
                )
            except Exception:
                continue
            if record.id not in parents:
                parents.append(record.id)
        return parents


def _default_client(base_url: str | None, api_key: str | None, timeout: float) -> HindsightAsyncClient:
    if not base_url:
        raise ValueError("base_url is required when a Hindsight client is not supplied")
    try:
        from hindsight_client import Hindsight
    except ImportError as error:
        raise ImportError("Hindsight support requires `pip install 'nemo-relay[hindsight]'`") from error
    return cast(
        HindsightAsyncClient,
        Hindsight(base_url=base_url, api_key=api_key, timeout=timeout, user_agent="nemo-relay-memory/0.1"),
    )


def _field(value: object, *names: str) -> object:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return cast(Mapping[object, object], value)[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _mapping_field(value: object, name: str) -> Mapping[str, object]:
    field = _field(value, name)
    if not isinstance(field, Mapping):
        return {}
    return cast(Mapping[str, object], field)


def _final_score(item: object, operation_id: str) -> float:
    scores = _field(item, "scores")
    score = _field(scores, "final")
    if not isinstance(score, int | float) or isinstance(score, bool):
        _invalid_response(operation_id, "Hindsight recall result has no numeric final score")
    return float(score)


def _reflect_fact_ids(response: object) -> list[str]:
    based_on = _field(response, "based_on")
    memories = _field(based_on, "memories")
    if not isinstance(memories, list):
        return []
    output: list[str] = []
    for memory in memories:
        fact_id = _field(memory, "id")
        if isinstance(fact_id, str) and fact_id not in output:
            output.append(fact_id)
    return output


def _record_id(metadata: Mapping[str, object], operation_id: str) -> str:
    value = metadata.get("_nemo_relay_record_id")
    if not isinstance(value, str):
        _invalid_response(operation_id, "Hindsight result has no Relay document identity")
    return value


def _maintenance_scope(value: object, operation_id: str) -> MemorySearchScope:
    try:
        return MemorySearchScope(str(value))
    except ValueError:
        raise_provider_error(
            "hindsight",
            operation_id,
            MemoryErrorCode.INVALID_REQUEST,
            "Hindsight reflect scope must be subject, agent, session, or exact",
        )


def _maintenance_limit(value: object, operation_id: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 1_000:
        raise_provider_error(
            "hindsight",
            operation_id,
            MemoryErrorCode.INVALID_REQUEST,
            "Hindsight reflect limit must be an integer in 1..=1000",
        )
    return value


def _tags_match(tags: list[str] | None) -> str:
    return "all_strict" if tags else "any"


def _invalid_response(operation_id: str, message: str) -> NoReturn:
    raise_provider_error("hindsight", operation_id, MemoryErrorCode.INTERNAL, message)


__all__ = [
    "HINDSIGHT_MAX_VERSION",
    "HINDSIGHT_MIN_VERSION",
    "HindsightAsyncClient",
    "HindsightMemoryProvider",
    "REFLECTION_ARTIFACT_VERSION",
]
