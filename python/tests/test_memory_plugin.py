# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import nemo_relay
from nemo_relay import Json, LLMRequest, llm, plugin, scope, subscribers
from nemo_relay.memory import (
    MemoryAction,
    MemoryAttachment,
    MemoryLifecycleEvent,
    MemoryLifecyclePhase,
    MemoryPlugin,
)


class RecordingListener:
    name = "recording-memory"

    def __init__(self) -> None:
        self.events: list[MemoryLifecycleEvent] = []

    async def on_event(self, event: MemoryLifecycleEvent) -> MemoryAction | None:
        self.events.append(event)
        if event.phase is MemoryLifecyclePhase.BEFORE_LLM:
            return MemoryAction(attachments=(MemoryAttachment("memory-1", "Ada prefers solarized mode.", 0.91),))
        return MemoryAction(stored_references=("memory-2",))


async def test_memory_plugin_forwards_events_injects_context_and_emits_marks():
    listener = RecordingListener()
    kind = "python.test_memory_lifecycle"
    subscriber_name = "python_test_memory_events"
    observed = []
    seen_request: LLMRequest | None = None
    plugin.register(kind, MemoryPlugin(listener))
    subscribers.register(subscriber_name, observed.append)
    try:
        await plugin.initialize(plugin.PluginConfig(components=[plugin.ComponentSpec(kind=kind)]))

        async def model(request: LLMRequest) -> Json:
            nonlocal seen_request
            seen_request = request
            return {"answer": "You prefer solarized mode."}

        request = LLMRequest({}, {"messages": [{"role": "user", "content": "What do I prefer?"}]})
        with scope.scope(
            "plain-agent",
            nemo_relay.ScopeType.Agent,
            data={"user_id": "ada", "session_id": "session-b", "correlation_id": "turn-1"},
        ):
            response = await llm.execute("demo-model", request, model)
    finally:
        subscribers.flush()
        plugin.clear()
        subscribers.deregister(subscriber_name)
        plugin.deregister(kind)

    assert response == {"answer": "You prefer solarized mode."}
    assert seen_request is not None
    messages = seen_request.content["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert "memory-1" in messages[0]["content"]
    assert "Ada prefers solarized mode." in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "What do I prefer?"}

    assert [event.phase for event in listener.events] == [
        MemoryLifecyclePhase.BEFORE_LLM,
        MemoryLifecyclePhase.AFTER_LLM,
    ]
    assert listener.events[0].context == {
        "user_id": "ada",
        "session_id": "session-b",
        "correlation_id": "turn-1",
    }
    assert listener.events[1].request.content == request.content
    assert listener.events[1].response == response
    assert listener.events[1].attachments[0].reference == "memory-1"

    marks = {event.name: event for event in observed if event.kind == "mark"}
    assert marks["memory.retrieved"].data == {
        "listener": "recording-memory",
        "count": 1,
        "references": ["memory-1"],
        "correlation_id": "turn-1",
    }
    assert marks["memory.injected"].data == marks["memory.retrieved"].data
    assert marks["memory.stored"].data == {
        "listener": "recording-memory",
        "count": 1,
        "references": ["memory-2"],
        "correlation_id": "turn-1",
    }
    assert "solarized" not in str([event.data for event in marks.values()])


async def test_listener_failure_is_observable_and_does_not_break_the_agent():
    class FailingListener:
        name = "failing-memory"

        async def on_event(self, event: MemoryLifecycleEvent) -> MemoryAction | None:
            raise RuntimeError(str(event.phase))

    kind = "python.test_failing_memory"
    subscriber_name = "python_test_memory_failure"
    observed = []
    plugin.register(kind, MemoryPlugin(FailingListener()))
    subscribers.register(subscriber_name, observed.append)
    try:
        await plugin.initialize(plugin.PluginConfig(components=[plugin.ComponentSpec(kind=kind)]))
        with scope.scope("plain-agent", nemo_relay.ScopeType.Agent):
            response = await llm.execute(
                "demo-model",
                LLMRequest({}, {"messages": [{"role": "user", "content": "Hello"}]}),
                lambda request: {"message_count": len(cast(list[object], request.content["messages"]))},
            )
    finally:
        subscribers.flush()
        plugin.clear()
        subscribers.deregister(subscriber_name)
        plugin.deregister(kind)

    assert response == {"message_count": 1}
    errors = [event for event in observed if event.kind == "mark" and event.name == "memory.listener_error"]
    assert [event.data["phase"] for event in errors] == ["llm.before", "llm.after"]
    assert all(event.data["error_type"] == "RuntimeError" for event in errors)


async def test_memory_plugin_injects_serialized_langchain_messages():
    listener = RecordingListener()
    kind = "python.test_memory_langchain_shape"
    seen: LLMRequest | None = None
    plugin.register(kind, MemoryPlugin(listener))
    try:
        await plugin.initialize(plugin.PluginConfig(components=[plugin.ComponentSpec(kind=kind)]))

        async def model(request: LLMRequest):
            nonlocal seen
            seen = request
            return {"answer": "ok"}

        request = LLMRequest(
            {},
            {
                "messages": [
                    {
                        "type": "human",
                        "data": {
                            "content": "What do I prefer?",
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "type": "human",
                            "name": None,
                            "id": None,
                        },
                    }
                ]
            },
        )
        with scope.scope("langchain-agent", nemo_relay.ScopeType.Agent):
            await llm.execute("demo-model", request, model)
    finally:
        subscribers.flush()
        plugin.clear()
        plugin.deregister(kind)

    assert seen is not None
    messages = cast(list[dict], seen.content["messages"])
    assert messages[0]["type"] == "system"
    assert "memory-1" in messages[0]["data"]["content"]
    assert messages[1]["type"] == "human"


def test_memory_attachment_rejects_blank_values():
    for reference, content in (("", "content"), ("memory-1", " ")):
        try:
            MemoryAttachment(reference, content)
        except ValueError as error:
            assert "must not be blank" in str(error)
        else:
            raise AssertionError("blank memory attachment should fail")
