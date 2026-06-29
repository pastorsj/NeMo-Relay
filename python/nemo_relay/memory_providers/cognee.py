# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Cognee 1.2 adapter for remember, recall, forget, and improve."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import NoReturn, Protocol, cast
from uuid import UUID

from nemo_relay import JsonObject
from nemo_relay.memory import (
    MemoryCapabilities,
    MemoryDeleteRequest,
    MemoryDeleteResult,
    MemoryErrorCode,
    MemoryMaintenanceAction,
    MemoryMaintenanceRequest,
    MemoryMaintenanceResult,
    MemoryMatch,
    MemoryNamespace,
    MemorySearchRequest,
    MemorySearchResult,
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
)

COGNEE_MIN_VERSION = "1.2.2"
COGNEE_MAX_VERSION = "1.3.0"
COGNEE_SCORE_SEMANTICS = "reciprocal_rank"
_DATASET_VERSION = "nemo-relay-cognee-v1"
_ALLOWED_MAINTENANCE_PARAMETERS = frozenset({"checkpoint_id", "query"})


@dataclass(frozen=True, slots=True)
class CogneeDataItem:
    """Dependency-free mirror of the Cognee DataItem fields Relay uses."""

    data: str
    label: str
    external_metadata: dict[str, str]
    data_id: UUID


@dataclass(frozen=True, slots=True)
class CogneeRememberSnapshot:
    """Stable completion facts returned from Cognee remember."""

    status: str
    data_id: UUID


@dataclass(frozen=True, slots=True)
class CogneeRecallHit:
    """One normalized Cognee CHUNKS recall hit."""

    data_id: UUID
    text: str
    chunk_id: str | None = None
    vendor_score: float | None = None


@dataclass(frozen=True, slots=True)
class CogneeDataSnapshot:
    """Stored Cognee source-data identity and Relay external metadata."""

    data_id: UUID
    external_metadata: Mapping[str, object]


class CogneeAsyncClient(Protocol):
    """Subset of Cognee 1.2 consumed by the adapter."""

    async def remember(
        self,
        item: CogneeDataItem,
        *,
        dataset_name: str,
        run_in_background: bool,
        self_improvement: bool,
    ) -> CogneeRememberSnapshot: ...

    async def recall(
        self,
        query_text: str,
        *,
        query_type: str,
        datasets: list[str],
        top_k: int,
        auto_route: bool,
        scope: str,
        include_references: bool,
    ) -> Sequence[CogneeRecallHit]: ...

    async def list_data(
        self,
        dataset_name: str,
        data_ids: Sequence[UUID],
    ) -> Sequence[CogneeDataSnapshot]: ...

    async def forget(self, *, data_id: UUID, dataset: str) -> bool: ...

    async def improve(self, dataset: str, *, run_in_background: bool) -> object: ...


class CogneeMemoryProvider:
    """Relay memory provider profile for Cognee 1.2 permanent memory."""

    name = "cognee"
    capabilities = MemoryCapabilities(delete=True, maintenance=True)

    def __init__(
        self,
        client: CogneeAsyncClient | None = None,
        *,
        idempotency_capacity: int = 1_024,
        overfetch_factor: int = 4,
    ) -> None:
        if overfetch_factor < 1:
            raise ValueError("overfetch_factor must be positive")
        self._client = client if client is not None else _default_client()
        self._ledger = IdempotencyLedger(idempotency_capacity)
        self._overfetch_factor = overfetch_factor

    async def store(self, request: MemoryStoreRequest) -> MemoryStoreResult:
        """Remember one permanent Cognee source item without auto-improve."""
        request.validate()

        async def mutation() -> MemoryStoreResult:
            record, reserved = prepare_record(self.name, request)
            data_id = _record_uuid(record.id, request.context.operation_id)
            dataset_name = cognee_dataset(request.namespace)
            response = await await_vendor(
                self._client.remember(
                    CogneeDataItem(
                        data=content_search_text(request.content),
                        label=record.id,
                        external_metadata=reserved,
                        data_id=data_id,
                    ),
                    dataset_name=dataset_name,
                    run_in_background=False,
                    self_improvement=False,
                ),
                provider=self.name,
                operation_id=request.context.operation_id,
                deadline=request.context.deadline,
            )
            if not isinstance(response, CogneeRememberSnapshot):
                _invalid_response(request.context.operation_id, "Cognee remember returned an invalid result")
            if response.status != "completed" or response.data_id != data_id:
                raise_provider_error(
                    self.name,
                    request.context.operation_id,
                    MemoryErrorCode.PROVIDER_UNAVAILABLE,
                    "Cognee remember did not complete",
                    retryable=False,
                )
            record = reconstruct_record(
                self.name,
                request.context.operation_id,
                reserved,
                provider_metadata=_provider_metadata(data_id, dataset_name),
            )
            return MemoryStoreResult(record, MemoryStoreDisposition.CREATED)

        return await self._ledger.run(request, mutation)

    async def search(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Recall Cognee chunks and reconstruct their Relay source records."""
        request.validate()
        dataset_name = cognee_dataset(request.namespace)
        top_k = min(1_000, max(request.limit, request.limit * self._overfetch_factor))
        hits = await await_vendor(
            self._client.recall(
                request.query,
                query_type="CHUNKS",
                datasets=[dataset_name],
                top_k=top_k,
                auto_route=False,
                scope="graph",
                include_references=False,
            ),
            provider=self.name,
            operation_id=request.context.operation_id,
            deadline=request.context.deadline,
        )
        if not isinstance(hits, Sequence):
            _invalid_response(request.context.operation_id, "Cognee recall response is not a sequence")
        normalized_hits: list[CogneeRecallHit] = []
        data_ids: list[UUID] = []
        for hit in hits:
            if not isinstance(hit, CogneeRecallHit):
                _invalid_response(request.context.operation_id, "Cognee recall returned an invalid chunk")
            normalized_hits.append(hit)
            if hit.data_id not in data_ids:
                data_ids.append(hit.data_id)
        snapshots = await await_vendor(
            self._client.list_data(dataset_name, data_ids),
            provider=self.name,
            operation_id=request.context.operation_id,
            deadline=request.context.deadline,
        )
        by_id = _snapshot_map(snapshots, request.context.operation_id)
        matches: list[MemoryMatch] = []
        seen_record_ids: set[str] = set()
        for hit in normalized_hits:
            snapshot = by_id.get(hit.data_id)
            if snapshot is None:
                _invalid_response(request.context.operation_id, "Cognee recall referenced missing source data")
            record = reconstruct_record(
                self.name,
                request.context.operation_id,
                snapshot.external_metadata,
                provider_metadata=_provider_metadata(hit.data_id, dataset_name),
            )
            if _record_uuid(record.id, request.context.operation_id) != hit.data_id:
                _invalid_response(request.context.operation_id, "Cognee data identity does not match its Relay record")
            if record.id in seen_record_ids or not record_matches(record, request):
                continue
            seen_record_ids.add(record.id)
            rank = len(matches) + 1
            matches.append(MemoryMatch(record=record, score=1.0 / rank, rank=rank))
            if len(matches) == request.limit:
                break
        return MemorySearchResult(tuple(matches))

    async def delete(self, request: MemoryDeleteRequest) -> MemoryDeleteResult:
        """Forget one source item inside its exact tenant/subject dataset."""
        request.validate()
        data_id = _record_uuid(request.id, request.context.operation_id)
        dataset_name = cognee_dataset(request.namespace)
        snapshots = await await_vendor(
            self._client.list_data(dataset_name, [data_id]),
            provider=self.name,
            operation_id=request.context.operation_id,
            deadline=request.context.deadline,
        )
        by_id = _snapshot_map(snapshots, request.context.operation_id)
        snapshot = by_id.get(data_id)
        if snapshot is None:
            return MemoryDeleteResult(request.id, False)
        record = reconstruct_record(
            self.name,
            request.context.operation_id,
            snapshot.external_metadata,
            provider_metadata=_provider_metadata(data_id, dataset_name),
        )
        if record.id != request.id or record.namespace.tenant_id != request.namespace.tenant_id:
            _invalid_response(request.context.operation_id, "Cognee delete identity does not match its Relay record")
        if record.namespace.subject_id != request.namespace.subject_id:
            _invalid_response(request.context.operation_id, "Cognee delete crossed its Relay subject partition")
        deleted = await await_vendor(
            self._client.forget(data_id=data_id, dataset=dataset_name),
            provider=self.name,
            operation_id=request.context.operation_id,
            deadline=request.context.deadline,
        )
        if not isinstance(deleted, bool):
            _invalid_response(request.context.operation_id, "Cognee forget returned an invalid result")
        if deleted:
            await self._ledger.invalidate_record(request.id)
        return MemoryDeleteResult(request.id, bool(deleted))

    async def maintain(self, request: MemoryMaintenanceRequest) -> MemoryMaintenanceResult:
        """Run provider-native Cognee improve as Consolidate maintenance."""
        request.validate()
        if request.action is not MemoryMaintenanceAction.CONSOLIDATE:
            raise_provider_error(
                self.name,
                request.context.operation_id,
                MemoryErrorCode.UNSUPPORTED,
                "Cognee adapter supports consolidate maintenance only",
            )
        if request.window is not None:
            raise_provider_error(
                self.name,
                request.context.operation_id,
                MemoryErrorCode.UNSUPPORTED,
                "Cognee improve cannot enforce a bounded Relay maintenance window",
            )
        unsupported = sorted(set(request.parameters) - _ALLOWED_MAINTENANCE_PARAMETERS)
        if unsupported:
            raise_provider_error(
                self.name,
                request.context.operation_id,
                MemoryErrorCode.INVALID_REQUEST,
                f"unsupported Cognee improve parameters: {', '.join(unsupported)}",
            )
        await await_vendor(
            self._client.improve(cognee_dataset(request.namespace), run_in_background=False),
            provider=self.name,
            operation_id=request.context.operation_id,
            deadline=request.context.deadline,
        )
        return MemoryMaintenanceResult()


class _CogneeSdkClient:
    """Lazy wrapper around the current Cognee v1 memory-oriented API."""

    def __init__(self) -> None:
        import cognee
        from cognee.modules.search.types import SearchType
        from cognee.tasks.ingestion.data_item import DataItem

        self._cognee = cognee
        self._data_item_type = DataItem
        self._search_type = SearchType

    async def remember(
        self,
        item: CogneeDataItem,
        *,
        dataset_name: str,
        run_in_background: bool,
        self_improvement: bool,
    ) -> CogneeRememberSnapshot:
        result = await self._cognee.remember(
            self._data_item_type(
                data=item.data,
                label=item.label,
                external_metadata=item.external_metadata,
                data_id=item.data_id,
            ),
            dataset_name=dataset_name,
            run_in_background=run_in_background,
            self_improvement=self_improvement,
        )
        return CogneeRememberSnapshot(str(getattr(result, "status", "")), item.data_id)

    async def recall(
        self,
        query_text: str,
        *,
        query_type: str,
        datasets: list[str],
        top_k: int,
        auto_route: bool,
        scope: str,
        include_references: bool,
    ) -> Sequence[CogneeRecallHit]:
        if query_type != "CHUNKS":
            raise ValueError("Relay Cognee wrapper supports CHUNKS recall only")
        results = await self._cognee.recall(
            query_text,
            query_type=self._search_type.CHUNKS,
            datasets=datasets,
            top_k=top_k,
            auto_route=auto_route,
            scope=scope,
            include_references=include_references,
        )
        output: list[CogneeRecallHit] = []
        for result in results:
            metadata = _object_mapping(_object_field(result, "metadata", {}))
            raw = _object_mapping(_object_field(result, "raw", {}))
            raw_data_id = metadata.get("data_id", raw.get("document_id"))
            try:
                data_id = UUID(str(raw_data_id))
            except (TypeError, ValueError):
                raise ValueError("Cognee CHUNKS result has no valid data_id")
            score = _object_field(result, "score", None)
            output.append(
                CogneeRecallHit(
                    data_id=data_id,
                    text=str(_object_field(result, "text", "")),
                    chunk_id=_optional_string(metadata.get("chunk_id", raw.get("id"))),
                    vendor_score=float(score)
                    if isinstance(score, int | float) and not isinstance(score, bool)
                    else None,
                )
            )
        return tuple(output)

    async def list_data(
        self,
        dataset_name: str,
        data_ids: Sequence[UUID],
    ) -> Sequence[CogneeDataSnapshot]:
        if not data_ids:
            return ()
        datasets = await self._cognee.datasets.list_datasets()
        dataset = next((item for item in datasets if _object_field(item, "name", None) == dataset_name), None)
        if dataset is None:
            return ()
        wanted = set(data_ids)
        records = await self._cognee.datasets.list_data(_object_field(dataset, "id", None))
        output: list[CogneeDataSnapshot] = []
        for record in records:
            try:
                data_id = UUID(str(_object_field(record, "id", None)))
            except (TypeError, ValueError):
                continue
            if data_id not in wanted:
                continue
            metadata = _object_mapping(_object_field(record, "external_metadata", {}))
            output.append(CogneeDataSnapshot(data_id, metadata))
        return tuple(output)

    async def forget(self, *, data_id: UUID, dataset: str) -> bool:
        result = await self._cognee.forget(data_id=data_id, dataset=dataset)
        return (
            isinstance(result, Mapping)
            and result.get("status") == "success"
            and str(result.get("data_id")) == str(data_id)
        )

    async def improve(self, dataset: str, *, run_in_background: bool) -> object:
        return await self._cognee.improve(dataset=dataset, run_in_background=run_in_background)


def _default_client() -> CogneeAsyncClient:
    try:
        import cognee  # noqa: F401
    except ImportError as error:
        raise ImportError("Cognee support requires `pip install 'nemo-relay[cognee]'`") from error
    return _CogneeSdkClient()


def cognee_dataset(namespace: MemoryNamespace) -> str:
    """Return a short, opaque Cognee dataset for one tenant and subject."""
    namespace.validate()
    material = json.dumps(
        [_DATASET_VERSION, namespace.tenant_id, namespace.subject_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"nrelay_{sha256(material.encode()).hexdigest()[:48]}"


def _record_uuid(record_id: str, operation_id: str) -> UUID:
    prefix = "cognee-"
    try:
        if not record_id.startswith(prefix):
            raise ValueError("wrong provider prefix")
        return UUID(hex=record_id.removeprefix(prefix))
    except ValueError:
        raise_provider_error(
            "cognee",
            operation_id,
            MemoryErrorCode.INVALID_REQUEST,
            "Cognee record ID is invalid",
        )


def _provider_metadata(data_id: UUID, dataset_name: str) -> JsonObject:
    return {
        "data_id": str(data_id),
        "dataset_name": dataset_name,
        "score_semantics": COGNEE_SCORE_SEMANTICS,
    }


def _snapshot_map(
    snapshots: Sequence[CogneeDataSnapshot],
    operation_id: str,
) -> dict[UUID, CogneeDataSnapshot]:
    if not isinstance(snapshots, Sequence):
        _invalid_response(operation_id, "Cognee source-data response is not a sequence")
    output: dict[UUID, CogneeDataSnapshot] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, CogneeDataSnapshot):
            _invalid_response(operation_id, "Cognee returned invalid source data")
        output[snapshot.data_id] = snapshot
    return output


def _object_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _object_field(value: object, name: str, default: object) -> object:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value).get(name, default)
    return getattr(value, name, default)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _invalid_response(operation_id: str, message: str) -> NoReturn:
    raise_provider_error("cognee", operation_id, MemoryErrorCode.INTERNAL, message)


__all__ = [
    "COGNEE_MAX_VERSION",
    "COGNEE_MIN_VERSION",
    "COGNEE_SCORE_SEMANTICS",
    "CogneeAsyncClient",
    "CogneeDataItem",
    "CogneeDataSnapshot",
    "CogneeMemoryProvider",
    "CogneeRecallHit",
    "CogneeRememberSnapshot",
    "cognee_dataset",
]
