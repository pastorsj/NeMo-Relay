# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reusable Python supported-profile conformance runner."""

from __future__ import annotations

from datetime import datetime, timezone

from nemo_relay.memory import (
    MemoryCapabilities,
    MemoryMatch,
    MemoryProvider,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryStoreDisposition,
    MemoryStoreRequest,
    MemoryStoreResult,
)
from nemo_relay.memory_providers._common import IdempotencyLedger, prepare_record, record_matches
from nemo_relay.memory_providers.conformance import run_provider_conformance


class ReferenceAdapter:
    name = "python-reference"
    capabilities = MemoryCapabilities()

    def __init__(self) -> None:
        self._ledger = IdempotencyLedger()
        self._records = []

    async def store(self, request: MemoryStoreRequest) -> MemoryStoreResult:
        request.validate()

        async def mutation() -> MemoryStoreResult:
            record, _ = prepare_record(self.name, request, ingested_at=datetime.now(timezone.utc))
            self._records.append(record)
            return MemoryStoreResult(record, MemoryStoreDisposition.CREATED)

        return await self._ledger.run(request, mutation)

    async def search(self, request: MemorySearchRequest) -> MemorySearchResult:
        request.validate()
        query_tokens = set(request.query.lower().split())
        scored = []
        for record in self._records:
            if not record_matches(record, request):
                continue
            content_tokens = set((record.content.text or "").lower().split())
            score = len(query_tokens & content_tokens) / max(1, len(query_tokens))
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return MemorySearchResult(
            tuple(
                MemoryMatch(record=record, score=score, rank=rank)
                for rank, (score, record) in enumerate(scored[: request.limit], 1)
            )
        )


async def test_reference_adapter_passes_supported_conformance():
    provider: MemoryProvider = ReferenceAdapter()

    report = await run_provider_conformance(provider, "unit-reference")

    assert report.passed, [(case.name, case.message) for case in report.failures]
    assert [case.name for case in report.cases] == [
        "required_store",
        "required_search",
        "record_preservation",
        "subject_isolation_cross_session",
        "deterministic_order",
        "idempotency",
        "metadata_time_filters",
        "typed_invalid_request",
        "capability_agreement",
    ]


async def test_empty_run_id_is_reported_without_mutating_provider():
    provider = ReferenceAdapter()

    report = await run_provider_conformance(provider, " ")

    assert not report.passed
    assert report.failures[0].name == "valid_run_id"
    assert provider._records == []
