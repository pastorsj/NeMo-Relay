# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-language fixture and protocol tests for ``nemo_relay.memory``."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from nemo_relay import JsonObject, memory

FIXTURE_PATH = Path(__file__).parents[2] / "crates" / "types" / "tests" / "fixtures" / "memory_contract_v0_1.json"


@pytest.fixture(name="contract_fixture")
def contract_fixture_fixture() -> JsonObject:
    import json

    return json.loads(FIXTURE_PATH.read_text())


def test_canonical_fixture_round_trips_without_semantic_loss(contract_fixture: JsonObject):
    search_request_data = cast(JsonObject, contract_fixture["search_request"])
    search_result_data = cast(JsonObject, contract_fixture["search_result"])
    store_result_data = cast(JsonObject, contract_fixture["store_result"])
    capabilities_data = cast(JsonObject, contract_fixture["capabilities"])

    assert isinstance(search_request_data, dict)
    assert isinstance(search_result_data, dict)
    assert isinstance(store_result_data, dict)
    assert isinstance(capabilities_data, dict)

    request = memory.MemorySearchRequest.from_dict(search_request_data)
    result = memory.MemorySearchResult.from_dict(search_result_data)
    store_result = memory.MemoryStoreResult.from_dict(store_result_data)
    capabilities = memory.MemoryCapabilities.from_dict(capabilities_data)

    assert request.to_dict() == search_request_data
    assert result.to_dict() == search_result_data
    assert store_result.to_dict() == store_result_data
    assert capabilities.to_dict() == capabilities_data
    assert result.matches[0].record.namespace.session_id == "session-a"
    assert result.matches[0].record.provider_metadata == {"native": {"collection": "demo", "revision": 7}}


@pytest.mark.parametrize(
    ("namespace", "message"),
    [
        (memory.MemoryNamespace("", "subject"), "tenant_id"),
        (memory.MemoryNamespace("tenant", " "), "subject_id"),
        (memory.MemoryNamespace("tenant", "subject", session_id=""), "session_id"),
    ],
)
def test_namespace_rejects_empty_identity(namespace: memory.MemoryNamespace, message: str):
    with pytest.raises(memory.MemoryContractError, match=message) as error:
        namespace.to_dict()
    assert error.value.error.code is memory.MemoryErrorCode.INVALID_REQUEST


def test_narrow_scope_requires_matching_namespace_identifier():
    namespace = memory.MemoryNamespace("tenant", "subject")
    with pytest.raises(memory.MemoryContractError, match="agent_id"):
        memory.MemorySearchScope.AGENT.validate(namespace)
    with pytest.raises(memory.MemoryContractError, match="session_id"):
        memory.MemorySearchScope.SESSION.validate(namespace)


def test_json_and_reference_content_keep_nested_provider_values():
    json_content = memory.MemoryContent.json_content(
        {"nested": {"provider_key": [1, True, None]}, "raw_name": "bank_id"}
    )
    reference_content = memory.MemoryContent.reference_content("opaque:item:1", "safe preview")

    assert memory.MemoryContent.from_dict(json_content.to_dict()) == json_content
    assert memory.MemoryContent.from_dict(reference_content.to_dict()) == reference_content


async def test_runtime_checkable_provider_protocol_executes_required_methods(contract_fixture: JsonObject):
    result_data = cast(JsonObject, contract_fixture["search_result"])
    store_data = cast(JsonObject, contract_fixture["store_result"])
    request_data = cast(JsonObject, contract_fixture["search_request"])
    assert isinstance(result_data, dict)
    assert isinstance(store_data, dict)
    assert isinstance(request_data, dict)

    search_result = memory.MemorySearchResult.from_dict(result_data)
    store_result = memory.MemoryStoreResult.from_dict(store_data)
    provider = SimpleNamespace(
        name="fixture",
        capabilities=memory.MemoryCapabilities(),
        search=AsyncMock(return_value=search_result),
        store=AsyncMock(return_value=store_result),
    )

    assert isinstance(provider, memory.MemoryProvider)
    request = memory.MemorySearchRequest.from_dict(request_data)
    assert await provider.search(request) == search_result

    store_request = memory.MemoryStoreRequest(
        context=memory.MemoryRequestContext("store-1"),
        namespace=store_result.record.namespace,
        content=store_result.record.content,
        event_timestamp=datetime(2026, 6, 29, 11, 55, tzinfo=timezone.utc),
        provenance=store_result.record.provenance,
        metadata=store_result.record.metadata,
        idempotency_key="preference-1",
    )
    assert await provider.store(store_request) == store_result
    provider.search.assert_awaited_once_with(request)
    provider.store.assert_awaited_once_with(store_request)
