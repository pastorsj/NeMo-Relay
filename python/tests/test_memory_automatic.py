# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Python proof for native reference automatic memory."""

import json

import pytest

from nemo_relay import LLMRequest, ScopeType, codecs, llm, memory, scope, subscribers


def _request(user: str) -> LLMRequest:
    return LLMRequest(
        {},
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": user}],
            "preserved": {"provider": True},
        },
    )


def _response(assistant: str):
    return {
        "id": "response-1",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


async def _turn(session_id: str, user: str, assistant: str, seen: list[dict], *, enabled: bool = True):
    namespace = {
        "tenant_id": "tenant-python",
        "subject_id": "subject-alex",
        "session_id": session_id,
        "agent_id": "assistant",
    }

    async def provider(request: LLMRequest):
        seen.append(request.content)
        return _response(assistant)

    return await llm.execute(
        "python-memory-agent",
        _request(user),
        provider,
        metadata={"memory": {"enabled": enabled, "namespace": namespace}},
        codec=codecs.OpenAIChatCodec(),
        response_codec=codecs.OpenAIChatCodec(),
    )


@pytest.mark.asyncio
async def test_two_sessions_recall_write_and_emit_private_evidence_without_memory_tools():
    events = []
    subscribers.register("python-automatic-memory-events", events.append)
    component = memory.InMemoryAutomaticMemory().install(name="python-automatic-memory")
    try:
        first_seen: list[dict] = []
        await _turn(
            "session-a",
            "SENTINEL_USER prefers solarized dark editor theme",
            "SENTINEL_ASSISTANT acknowledged the preference",
            first_seen,
        )
        assert "<relay_memory" not in json.dumps(first_seen[0])

        second_seen: list[dict] = []
        await _turn(
            "session-b",
            "Which editor theme does SENTINEL_USER prefer?",
            "The preference is solarized dark.",
            second_seen,
        )
        injected = json.dumps(second_seen[0])
        assert '<relay_memory version=\\"0.1\\">' in injected
        assert "solarized dark editor theme" in injected
        assert second_seen[0]["preserved"] == {"provider": True}

        subscribers.flush()
        memory_events = [event for event in events if event.category == "memory"]
        assert {event.name for event in memory_events} >= {
            "memory.retrieval",
            "memory.injection",
            "memory.storage",
        }
        evidence = json.dumps([event.data for event in memory_events], sort_keys=True)
        assert "SENTINEL_USER" not in evidence
        assert "SENTINEL_ASSISTANT" not in evidence
        assert "solarized dark editor theme" not in evidence
        assert "content_hash" in evidence
        assert component.active_turns == 0
    finally:
        assert component.close() is True
        assert component.close() is False
        subscribers.flush()
        subscribers.deregister("python-automatic-memory-events")


@pytest.mark.asyncio
async def test_opt_out_and_scope_cleanup_preserve_the_normal_request():
    config = memory.AutomaticMemoryConfig(
        namespace=memory.MemoryNamespace(
            "tenant-python",
            "subject-alex",
            session_id="fallback",
            agent_id="assistant",
        )
    )
    parent = scope.push("python-memory-scope", ScopeType.Agent)
    component = memory.InMemoryAutomaticMemory(config).install(
        name="python-scope-memory",
        scope=parent,
    )
    try:
        seen: list[dict] = []
        await _turn(
            "session-opt-out",
            "do not remember this sentinel",
            "not stored",
            seen,
            enabled=False,
        )
        assert seen == [_request("do not remember this sentinel").content]
    finally:
        scope.pop(parent)

    assert component.close() is False
    assert component.active_turns == 0


def test_invalid_automatic_config_is_rejected_by_native_validation():
    with pytest.raises(ValueError, match="max_items"):
        memory.InMemoryAutomaticMemory(memory.AutomaticMemoryConfig(max_candidates=1, max_items=2))


@pytest.mark.asyncio
async def test_background_write_back_flush_status_and_shutdown_preserve_automatic_recall():
    events = []
    subscribers.register("python-background-memory-events", events.append)
    component = memory.InMemoryAutomaticMemory(
        memory.AutomaticMemoryConfig(
            write_delivery=memory.WriteDelivery.BACKGROUND,
            background_queue=memory.MemoryWorkQueueConfig(
                capacity=2,
                max_attempts=2,
                retry_initial_delay_millis=1,
                retry_max_delay_millis=1,
            ),
        )
    ).install(name="python-background-memory")
    shut_down = False
    try:
        first_seen: list[dict] = []
        await _turn(
            "background-a",
            "BACKGROUND_USER prefers a nord editor theme",
            "Preference acknowledged",
            first_seen,
        )
        assert "<relay_memory" not in json.dumps(first_seen[0])
        assert component.background_status is not None
        assert component.background_status["accepted_total"] == 1
        assert await component.flush(timeout_millis=5_000) is True

        second_seen: list[dict] = []
        await _turn(
            "background-b",
            "Which editor theme does BACKGROUND_USER prefer?",
            "Nord.",
            second_seen,
        )
        injected = json.dumps(second_seen[0])
        assert '<relay_memory version=\\"0.1\\">' in injected
        assert "nord editor theme" in injected
        assert await component.flush() is True

        subscribers.flush()
        storage_events = [event for event in events if event.name == "memory.storage"]
        statuses = {event.data["status"] for event in storage_events}
        assert statuses >= {"queued", "running", "stored"}
        first_job_id = storage_events[0].data["job_id"]
        job = component.background_job_status(first_job_id)
        assert job is not None
        assert job["state"] == "succeeded"
        assert job["attempts"] == 1
        evidence = json.dumps([event.data for event in storage_events], sort_keys=True)
        assert "BACKGROUND_USER" not in evidence
        assert "nord editor theme" not in evidence

        with pytest.raises(ValueError, match="timeout_millis"):
            await component.flush(timeout_millis=0)
        assert await component.shutdown() is True
        shut_down = True
        assert component.background_status["accepting"] is False
    finally:
        if not shut_down:
            component.close()
            await component.shutdown()
        subscribers.flush()
        subscribers.deregister("python-background-memory-events")
