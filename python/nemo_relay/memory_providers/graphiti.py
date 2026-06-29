# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Graphiti OSS 0.29 temporal knowledge-graph memory adapter."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import NoReturn, Protocol, cast

from nemo_relay import Json, JsonObject
from nemo_relay.memory import (
    MemoryCapabilities,
    MemoryErrorCode,
    MemoryMatch,
    MemoryNamespace,
    MemoryRecord,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryStoreDisposition,
    MemoryStoreRequest,
    MemoryStoreResult,
)
from nemo_relay.memory_providers._common import (
    RECORD_METADATA_KEY,
    IdempotencyLedger,
    await_vendor,
    content_search_text,
    prepare_record,
    raise_provider_error,
    reconstruct_record,
    record_matches,
)

GRAPHITI_MIN_VERSION = "0.29.2"
GRAPHITI_MAX_VERSION = "0.30.0"
GRAPHITI_SCORE_SEMANTICS = "reciprocal_rank"
_EPISODE_NAME_PREFIX = "nemo-relay-record-v1:"
_GROUP_VERSION = "nemo-relay-graphiti-v1"


@dataclass(frozen=True, slots=True)
class GraphitiEdgeSnapshot:
    """Provider-neutral snapshot of one Graphiti entity edge."""

    uuid: str
    name: str
    fact: str
    episodes: tuple[str, ...]
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime
    expired_at: datetime | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    reference_time: datetime | None = None
    attributes: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class GraphitiEpisodeSnapshot:
    """Provider-neutral source episode returned by the Graphiti wrapper."""

    uuid: str
    name: str
    group_id: str
    content: str
    created_at: datetime
    valid_at: datetime
    entity_edge_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphitiEpisodeEvidence:
    """One episode and its current temporal entity-edge view."""

    episode: GraphitiEpisodeSnapshot
    edges: tuple[GraphitiEdgeSnapshot, ...]
    fact_rank: int | None = None


class GraphitiAsyncClient(Protocol):
    """Small async Graphiti surface consumed by the Relay adapter."""

    async def add_episode(
        self,
        *,
        name: str,
        body: str,
        source_description: str,
        reference_time: datetime,
        group_id: str,
        uuid: str,
    ) -> GraphitiEpisodeEvidence: ...

    async def search_episodes(
        self,
        query: str,
        *,
        group_id: str,
        num_results: int,
    ) -> Sequence[GraphitiEpisodeEvidence]: ...


class GraphitiMemoryProvider:
    """Relay memory profile for the Graphiti OSS temporal graph library.

    The Relay source record is stored as an episode. Graphiti-extracted facts
    remain provider evidence and never replace that source record.
    """

    name = "graphiti"
    capabilities = MemoryCapabilities()

    def __init__(
        self,
        client: GraphitiAsyncClient | None = None,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: object | None = None,
        embedder: object | None = None,
        cross_encoder: object | None = None,
        idempotency_capacity: int = 1_024,
        overfetch_factor: int = 4,
    ) -> None:
        connection_supplied = any(
            value is not None for value in (uri, user, password, llm_client, embedder, cross_encoder)
        )
        if client is not None and connection_supplied:
            raise ValueError("pass either a Graphiti client or connection settings, not both")
        if overfetch_factor < 1:
            raise ValueError("overfetch_factor must be positive")
        self._client = (
            client
            if client is not None
            else _default_client(
                uri=uri,
                user=user,
                password=password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
            )
        )
        self._ledger = IdempotencyLedger(idempotency_capacity)
        self._overfetch_factor = overfetch_factor

    async def store(self, request: MemoryStoreRequest) -> MemoryStoreResult:
        """Store one Relay source episode and await graph extraction."""
        request.validate()

        async def mutation() -> MemoryStoreResult:
            record, reserved = prepare_record(self.name, request)
            group_id = graphiti_group(request.namespace)
            evidence = await await_vendor(
                self._client.add_episode(
                    name=_encode_episode_name(reserved),
                    body=content_search_text(request.content),
                    source_description="NeMo Relay canonical memory episode",
                    reference_time=request.event_timestamp,
                    group_id=group_id,
                    uuid=record.id,
                ),
                provider=self.name,
                operation_id=request.context.operation_id,
                deadline=request.context.deadline,
            )
            rebuilt = _record_from_evidence(evidence, request.context.operation_id, group_id)
            if rebuilt.id != record.id:
                _invalid_response(request.context.operation_id, "Graphiti returned a different episode ID")
            return MemoryStoreResult(rebuilt, MemoryStoreDisposition.CREATED)

        return await self._ledger.run(request, mutation)

    async def search(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Search ranked facts and return their Relay source episodes."""
        request.validate()
        group_id = graphiti_group(request.namespace)
        num_results = min(1_000, max(request.limit, request.limit * self._overfetch_factor))
        evidences = await await_vendor(
            self._client.search_episodes(request.query, group_id=group_id, num_results=num_results),
            provider=self.name,
            operation_id=request.context.operation_id,
            deadline=request.context.deadline,
        )
        if not isinstance(evidences, Sequence):
            _invalid_response(request.context.operation_id, "Graphiti search response is not a sequence")
        matches: list[MemoryMatch] = []
        seen_record_ids: set[str] = set()
        for evidence in evidences:
            if not isinstance(evidence, GraphitiEpisodeEvidence):
                _invalid_response(request.context.operation_id, "Graphiti search returned invalid episode evidence")
            fact_rank = evidence.fact_rank
            if not isinstance(fact_rank, int) or isinstance(fact_rank, bool) or fact_rank < 1:
                _invalid_response(request.context.operation_id, "Graphiti search evidence has no positive fact rank")
            record = _record_from_evidence(evidence, request.context.operation_id, group_id)
            if record.id in seen_record_ids or not record_matches(record, request):
                continue
            seen_record_ids.add(record.id)
            matches.append(
                MemoryMatch(
                    record=record,
                    score=1.0 / fact_rank,
                    rank=len(matches) + 1,
                )
            )
            if len(matches) == request.limit:
                break
        return MemorySearchResult(tuple(matches))


class _GraphitiSdkClient:
    """Lazy Graphiti SDK wrapper that contains mutable driver selection."""

    def __init__(
        self,
        *,
        uri: str,
        user: str | None,
        password: str | None,
        llm_client: object | None,
        embedder: object | None,
        cross_encoder: object | None,
    ) -> None:
        from graphiti_core import Graphiti
        from graphiti_core.edges import EntityEdge
        from graphiti_core.nodes import EpisodeType, EpisodicNode

        self._graph = Graphiti(
            uri,
            user,
            password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )
        self._entity_edge_type = EntityEdge
        self._episode_type = EpisodeType
        self._episodic_node_type = EpisodicNode
        self._lock = asyncio.Lock()

    async def add_episode(
        self,
        *,
        name: str,
        body: str,
        source_description: str,
        reference_time: datetime,
        group_id: str,
        uuid: str,
    ) -> GraphitiEpisodeEvidence:
        async with self._lock:
            self._select_group(group_id)
            result = await self._graph.add_episode(
                name,
                body,
                source_description,
                reference_time,
                source=self._episode_type.text,
                group_id=group_id,
                uuid=uuid,
            )
            episode = _sdk_episode(result.episode)
            edges = tuple(sorted((_sdk_edge(edge) for edge in result.edges), key=lambda edge: edge.uuid))
            return GraphitiEpisodeEvidence(episode, edges)

    async def search_episodes(
        self,
        query: str,
        *,
        group_id: str,
        num_results: int,
    ) -> Sequence[GraphitiEpisodeEvidence]:
        async with self._lock:
            self._select_group(group_id)
            ranked_edges = await self._graph.search(
                query,
                group_ids=[group_id],
                num_results=num_results,
            )
            episode_ranks: OrderedDict[str, int] = OrderedDict()
            for rank, edge in enumerate(ranked_edges, 1):
                for episode_id in edge.episodes:
                    episode_ranks.setdefault(episode_id, rank)
            if not episode_ranks:
                return ()
            episodes = await self._episodic_node_type.get_by_uuids(
                self._graph.driver,
                list(episode_ranks),
            )
            edge_ids = list(dict.fromkeys(edge_id for episode in episodes for edge_id in episode.entity_edges))
            graph_edges = await self._entity_edge_type.get_by_uuids(self._graph.driver, edge_ids) if edge_ids else []
            edge_snapshots = tuple(_sdk_edge(edge) for edge in graph_edges)
            normalized_edges = {edge.uuid: edge for edge in edge_snapshots}
            episode_by_id = {episode.uuid: episode for episode in episodes}
            output: list[GraphitiEpisodeEvidence] = []
            for episode_id, rank in episode_ranks.items():
                episode = episode_by_id.get(episode_id)
                if episode is None:
                    continue
                edges = tuple(
                    sorted(
                        (normalized_edges[edge_id] for edge_id in episode.entity_edges if edge_id in normalized_edges),
                        key=lambda edge: edge.uuid,
                    )
                )
                output.append(GraphitiEpisodeEvidence(_sdk_episode(episode), edges, rank))
            return tuple(output)

    def _select_group(self, group_id: str) -> None:
        driver = self._graph.driver.clone(database=group_id)
        self._graph.driver = driver
        self._graph.clients.driver = driver


def _default_client(
    *,
    uri: str | None,
    user: str | None,
    password: str | None,
    llm_client: object | None,
    embedder: object | None,
    cross_encoder: object | None,
) -> GraphitiAsyncClient:
    if not uri:
        raise ValueError("uri is required when a Graphiti client is not supplied")
    try:
        import graphiti_core  # noqa: F401
    except ImportError as error:
        raise ImportError("Graphiti support requires `pip install 'nemo-relay[graphiti]'`") from error
    return _GraphitiSdkClient(
        uri=uri,
        user=user,
        password=password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )


def graphiti_group(namespace: MemoryNamespace) -> str:
    """Return a short, opaque Graphiti-safe tenant/subject group key."""
    namespace.validate()
    material = json.dumps(
        [_GROUP_VERSION, namespace.tenant_id, namespace.subject_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"nrelay_{sha256(material.encode()).hexdigest()[:48]}"


def _encode_episode_name(reserved: Mapping[str, str]) -> str:
    payload = json.dumps(dict(reserved), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    encoded = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"{_EPISODE_NAME_PREFIX}{encoded}"


def _decode_episode_name(name: str, operation_id: str) -> Mapping[str, object]:
    if not name.startswith(_EPISODE_NAME_PREFIX):
        _invalid_response(operation_id, "Graphiti episode is missing Relay record metadata")
    encoded = name.removeprefix(_EPISODE_NAME_PREFIX)
    try:
        padding = "=" * (-len(encoded) % 4)
        value = json.loads(base64.urlsafe_b64decode(encoded + padding))
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise_provider_error(
            "graphiti",
            operation_id,
            MemoryErrorCode.INTERNAL,
            "Graphiti episode contains invalid Relay record metadata",
            details={"error_type": type(error).__name__},
        )
    if not isinstance(value, Mapping) or not isinstance(value.get(RECORD_METADATA_KEY), str):
        _invalid_response(operation_id, "Graphiti episode contains invalid Relay record metadata")
    return cast(Mapping[str, object], value)


def _record_from_evidence(
    evidence: GraphitiEpisodeEvidence,
    operation_id: str,
    expected_group: str,
) -> MemoryRecord:
    if evidence.episode.group_id != expected_group:
        _invalid_response(operation_id, "Graphiti episode crossed its Relay subject partition")
    try:
        provider_metadata = _provider_metadata(evidence)
    except ValueError as error:
        raise_provider_error(
            "graphiti",
            operation_id,
            MemoryErrorCode.INTERNAL,
            "Graphiti episode contains invalid temporal metadata",
            details={"error_type": type(error).__name__},
        )
    record = reconstruct_record(
        "graphiti",
        operation_id,
        _decode_episode_name(evidence.episode.name, operation_id),
        provider_metadata=provider_metadata,
    )
    if record.id != evidence.episode.uuid:
        _invalid_response(operation_id, "Graphiti episode identity does not match its Relay record")
    if content_search_text(record.content) != evidence.episode.content:
        _invalid_response(operation_id, "Graphiti episode content does not match its Relay record")
    return record


def _provider_metadata(evidence: GraphitiEpisodeEvidence) -> JsonObject:
    episode = evidence.episode
    return {
        "episode_uuid": episode.uuid,
        "episode_created_at": _format_datetime(episode.created_at),
        "episode_valid_at": _format_datetime(episode.valid_at),
        "graph_edges": [_edge_wire(edge) for edge in sorted(evidence.edges, key=lambda item: item.uuid)],
        "graph_snapshot_semantics": "current_episode_edges",
        "score_semantics": GRAPHITI_SCORE_SEMANTICS,
    }


def _edge_wire(edge: GraphitiEdgeSnapshot) -> JsonObject:
    return {
        "uuid": edge.uuid,
        "name": edge.name,
        "fact": edge.fact,
        "episodes": list(edge.episodes),
        "source_node_uuid": edge.source_node_uuid,
        "target_node_uuid": edge.target_node_uuid,
        "created_at": _format_datetime(edge.created_at),
        "expired_at": _optional_datetime_wire(edge.expired_at),
        "valid_at": _optional_datetime_wire(edge.valid_at),
        "invalid_at": _optional_datetime_wire(edge.invalid_at),
        "reference_time": _optional_datetime_wire(edge.reference_time),
        "attributes": _json_value(edge.attributes or {}),
    }


def _sdk_episode(value: object) -> GraphitiEpisodeSnapshot:
    return GraphitiEpisodeSnapshot(
        uuid=cast(str, getattr(value, "uuid")),
        name=cast(str, getattr(value, "name")),
        group_id=cast(str, getattr(value, "group_id")),
        content=cast(str, getattr(value, "content")),
        created_at=cast(datetime, getattr(value, "created_at")),
        valid_at=cast(datetime, getattr(value, "valid_at")),
        entity_edge_ids=tuple(cast(list[str], getattr(value, "entity_edges", []))),
    )


def _sdk_edge(value: object) -> GraphitiEdgeSnapshot:
    attributes = _json_value(getattr(value, "attributes", {}))
    if not isinstance(attributes, dict):
        attributes = {}
    return GraphitiEdgeSnapshot(
        uuid=cast(str, getattr(value, "uuid")),
        name=cast(str, getattr(value, "name")),
        fact=cast(str, getattr(value, "fact")),
        episodes=tuple(cast(list[str], getattr(value, "episodes", []))),
        source_node_uuid=cast(str, getattr(value, "source_node_uuid")),
        target_node_uuid=cast(str, getattr(value, "target_node_uuid")),
        created_at=cast(datetime, getattr(value, "created_at")),
        expired_at=cast(datetime | None, getattr(value, "expired_at", None)),
        valid_at=cast(datetime | None, getattr(value, "valid_at", None)),
        invalid_at=cast(datetime | None, getattr(value, "invalid_at", None)),
        reference_time=cast(datetime | None, getattr(value, "reference_time", None)),
        attributes=cast(JsonObject, attributes),
    )


def _json_value(value: object) -> Json:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return _format_datetime(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item) for item in value]
    return str(value)


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Graphiti returned a timezone-naive timestamp")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _optional_datetime_wire(value: datetime | None) -> Json:
    return None if value is None else _format_datetime(value)


def _invalid_response(operation_id: str, message: str) -> NoReturn:
    raise_provider_error("graphiti", operation_id, MemoryErrorCode.INTERNAL, message)


__all__ = [
    "GRAPHITI_MAX_VERSION",
    "GRAPHITI_MIN_VERSION",
    "GRAPHITI_SCORE_SEMANTICS",
    "GraphitiAsyncClient",
    "GraphitiEdgeSnapshot",
    "GraphitiEpisodeEvidence",
    "GraphitiEpisodeSnapshot",
    "GraphitiMemoryProvider",
    "graphiti_group",
]
