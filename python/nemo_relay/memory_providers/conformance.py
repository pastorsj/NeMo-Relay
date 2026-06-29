# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable supported-profile conformance for Python memory adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from nemo_relay.memory import (
    MemoryCapabilities,
    MemoryContent,
    MemoryContractError,
    MemoryErrorCode,
    MemoryFilter,
    MemoryMaintenanceAction,
    MemoryMaintenanceRequest,
    MemoryMaintenanceResult,
    MemoryMaintenanceWindow,
    MemoryNamespace,
    MemoryProvenance,
    MemoryProvider,
    MemoryProviderError,
    MemoryRequestContext,
    MemorySearchRequest,
    MemorySearchResult,
    MemorySearchScope,
    MemoryStoreDisposition,
    MemoryStoreRequest,
)

@dataclass(frozen=True, slots=True)
class ConformanceCase:
    """Result of one named provider invariant."""

    name: str
    passed: bool
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    """Structured outcome of one isolated provider conformance run."""

    provider: str
    run_id: str
    cases: tuple[ConformanceCase, ...]

    @property
    def passed(self) -> bool:
        """Return whether every supported case passed."""
        return all(case.passed for case in self.cases)

    @property
    def failures(self) -> tuple[ConformanceCase, ...]:
        """Return only failed cases."""
        return tuple(case for case in self.cases if not case.passed)


async def run_provider_conformance(provider: MemoryProvider, run_id: str) -> ConformanceReport:
    """Run required and advertised Python adapter cases in an isolated namespace."""
    cases: list[ConformanceCase] = []
    if not run_id.strip():
        return ConformanceReport(provider.name, run_id, (ConformanceCase("valid_run_id", False, "run_id is empty"),))

    origin = _namespace(run_id, "subject", "session-a", "assistant")
    cross_session = _namespace(run_id, "subject", "session-b", "assistant")
    other_subject = _namespace(run_id, "other-subject", "session-b", "assistant")
    stored_request = _store_request(run_id, "seed", origin, "conformance solarized editor preference", "seed-key")

    try:
        stored = await provider.store(stored_request)
        _require(stored.disposition is MemoryStoreDisposition.CREATED, "initial store did not report created")
        cases.append(ConformanceCase("required_store", True))
    except Exception as error:
        cases.append(_failed("required_store", error))
        for name in (
            "required_search",
            "record_preservation",
            "subject_isolation_cross_session",
            "deterministic_order",
            "idempotency",
            "metadata_time_filters",
            "typed_invalid_request",
            "capability_agreement",
        ):
            cases.append(ConformanceCase(name, False, "required seed store failed"))
        return ConformanceReport(provider.name, run_id, tuple(cases))

    search_request = _search_request(run_id, "required-search", cross_session, "solarized editor preference")
    try:
        searched = await provider.search(search_request)
        _require(searched.matches, "search returned no matches")
        cases.append(ConformanceCase("required_search", True))
    except Exception as error:
        searched = MemorySearchResult()
        cases.append(_failed("required_search", error))

    try:
        match = next((item for item in searched.matches if item.record.id == stored.record.id), None)
        _require(match is not None, "stored record was not returned")
        assert match is not None
        _require(match.record == stored.record, "stored record fields changed")
        _require(match.rank > 0 and match.score > 0, "rank or score is not positive")
        cases.append(ConformanceCase("record_preservation", True))
    except Exception as error:
        cases.append(_failed("record_preservation", error))

    try:
        isolated = await provider.search(
            _search_request(run_id, "isolated-search", other_subject, "solarized editor preference")
        )
        _require(
            any(item.record.id == stored.record.id for item in searched.matches),
            "subject search did not cross sessions",
        )
        _require(not isolated.matches, "search crossed the subject boundary")
        cases.append(ConformanceCase("subject_isolation_cross_session", True))
    except Exception as error:
        cases.append(_failed("subject_isolation_cross_session", error))

    try:
        first = await provider.search(search_request)
        second = await provider.search(search_request)
        _require(first == second, "repeated search changed results or order")
        cases.append(ConformanceCase("deterministic_order", True))
    except Exception as error:
        cases.append(_failed("deterministic_order", error))

    try:
        replay = await provider.store(
            MemoryStoreRequest(
                context=_context(run_id, "replay"),
                namespace=stored_request.namespace,
                content=stored_request.content,
                event_timestamp=stored_request.event_timestamp,
                provenance=stored_request.provenance,
                metadata=stored_request.metadata,
                idempotency_key=stored_request.idempotency_key,
            )
        )
        _require(replay.disposition is MemoryStoreDisposition.EXISTING, "replay did not report existing")
        _require(replay.record == stored.record, "replay did not return original record")
        conflict_request = MemoryStoreRequest(
            context=_context(run_id, "conflict"),
            namespace=stored_request.namespace,
            content=MemoryContent.text_content("different conformance payload"),
            event_timestamp=stored_request.event_timestamp,
            provenance=stored_request.provenance,
            metadata=stored_request.metadata,
            idempotency_key=stored_request.idempotency_key,
        )
        try:
            await provider.store(conflict_request)
        except MemoryProviderError as error:
            _require(error.error.code is MemoryErrorCode.CONFLICT, "conflict returned the wrong code")
        else:
            raise AssertionError("conflicting replay unexpectedly succeeded")
        cases.append(ConformanceCase("idempotency", True))
    except Exception as error:
        cases.append(_failed("idempotency", error))

    try:
        filtered = await provider.search(
            MemorySearchRequest(
                context=_context(run_id, "filtered"),
                namespace=cross_session,
                query="solarized editor preference",
                filter=MemoryFilter(
                    metadata={"conformance_kind": "preference"},
                    event_after=stored.record.event_timestamp - timedelta(seconds=1),
                    event_before=stored.record.event_timestamp + timedelta(seconds=1),
                    ingested_before=stored.record.ingested_at + timedelta(seconds=1),
                ),
                limit=4,
            )
        )
        rejected = await provider.search(
            MemorySearchRequest(
                context=_context(run_id, "filtered-empty"),
                namespace=cross_session,
                query="solarized editor preference",
                filter=MemoryFilter(metadata={"conformance_kind": "other"}),
                limit=4,
            )
        )
        _require(any(item.record.id == stored.record.id for item in filtered.matches), "matching filter lost record")
        _require(not rejected.matches, "nonmatching metadata filter returned records")
        cases.append(ConformanceCase("metadata_time_filters", True))
    except Exception as error:
        cases.append(_failed("metadata_time_filters", error))

    try:
        invalid = _search_request(run_id, "invalid", cross_session, " ")
        try:
            await provider.search(invalid)
        except MemoryContractError as error:
            _require(error.error.code is MemoryErrorCode.INVALID_REQUEST, "invalid request returned wrong code")
        except MemoryProviderError as error:
            _require(error.error.code is MemoryErrorCode.INVALID_REQUEST, "invalid request returned wrong code")
        else:
            raise AssertionError("invalid search unexpectedly succeeded")
        cases.append(ConformanceCase("typed_invalid_request", True))
    except Exception as error:
        cases.append(_failed("typed_invalid_request", error))

    try:
        await _check_capabilities(provider, run_id, origin)
        cases.append(ConformanceCase("capability_agreement", True))
    except Exception as error:
        cases.append(_failed("capability_agreement", error))

    return ConformanceReport(provider.name, run_id, tuple(cases))


async def _check_capabilities(provider: MemoryProvider, run_id: str, namespace: MemoryNamespace) -> None:
    capabilities: MemoryCapabilities = provider.capabilities
    unsupported_python_profile = {
        "update": capabilities.update,
        "delete": capabilities.delete,
        "batch_store": capabilities.batch_store,
        "feedback": capabilities.feedback,
        "health": capabilities.health,
    }
    enabled = [name for name, advertised in unsupported_python_profile.items() if advertised]
    _require(not enabled, f"Python adapter profile cannot verify advertised capabilities: {', '.join(enabled)}")
    maintain = getattr(provider, "maintain", None)
    if not capabilities.maintenance:
        return
    if not callable(maintain):
        raise AssertionError("maintenance is advertised but maintain is absent")
    request = MemoryMaintenanceRequest(
        context=_context(run_id, "maintenance"),
        namespace=namespace,
        action=MemoryMaintenanceAction.REFLECT,
        window=MemoryMaintenanceWindow(
            checkpoint_id=f"{run_id}-checkpoint",
            query="Reflect on the conformance memory",
            limit=5,
        ),
    )
    result = await maintain(request)
    _require(isinstance(result, MemoryMaintenanceResult), "maintain returned the wrong result type")


def _namespace(run_id: str, subject: str, session: str, agent: str) -> MemoryNamespace:
    return MemoryNamespace(f"relay-conformance-{run_id}", subject, session, agent)


def _context(run_id: str, suffix: str) -> MemoryRequestContext:
    return MemoryRequestContext(f"{run_id}-{suffix}")


def _store_request(
    run_id: str,
    suffix: str,
    namespace: MemoryNamespace,
    text: str,
    idempotency_key: str,
) -> MemoryStoreRequest:
    return MemoryStoreRequest(
        context=_context(run_id, suffix),
        namespace=namespace,
        content=MemoryContent.text_content(text),
        event_timestamp=datetime(2026, 6, 29, 12, tzinfo=timezone.utc),
        provenance=MemoryProvenance("conformance", (f"{run_id}-turn",)),
        metadata={"conformance_kind": "preference"},
        idempotency_key=f"{run_id}-{idempotency_key}",
    )


def _search_request(
    run_id: str,
    suffix: str,
    namespace: MemoryNamespace,
    query: str,
) -> MemorySearchRequest:
    return MemorySearchRequest(
        context=_context(run_id, suffix),
        namespace=namespace,
        query=query,
        scope=MemorySearchScope.SUBJECT,
        limit=10,
    )


def _require(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _failed(name: str, error: Exception) -> ConformanceCase:
    return ConformanceCase(name, False, f"{type(error).__name__}: {error}")


__all__ = ["ConformanceCase", "ConformanceReport", "run_provider_conformance"]
