"""Stand-ins for model endpoints, shared by the cortex tests.

Everything here mimics the *shape* of a real client rather than its behaviour,
so the router can be exercised end to end — failover, breaker, cache, schema
repair — without a network, an API key, or a model.
"""
from __future__ import annotations

import types

from aegis.cortex.providers import (
    AnthropicProvider, CallParams, Completion, OpenAICompatibleProvider, Provider,
)


class _Usage:
    def __init__(self, prompt_tokens=None, completion_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeOpenAIClient:
    """Minimal ``AsyncOpenAI`` look-alike.

    ``responses`` is consumed one per call; an entry that is an exception is
    raised instead of returned, which is how provider failure is simulated.
    """

    def __init__(self, responses, usage=_Usage(3, 7)):
        self._responses = list(responses)
        self._usage = usage
        self.requests: list[dict] = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self._responses.pop(0) if self._responses else ""
        if isinstance(item, Exception):
            raise item
        message = types.SimpleNamespace(content=item)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice], usage=self._usage)


#: Distinguishes "use the default usage object" from "this server reports no
#: usage at all" — passing None had to mean the second, not the first.
DEFAULT_USAGE = object()


class FakeAnthropicClient:
    def __init__(self, responses, usage=DEFAULT_USAGE):
        self._responses = list(responses)
        self._usage = (types.SimpleNamespace(input_tokens=5, output_tokens=9)
                       if usage is DEFAULT_USAGE else usage)
        self.requests: list[dict] = []
        self.messages = types.SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self._responses.pop(0) if self._responses else ""
        if isinstance(item, Exception):
            raise item
        blocks = [types.SimpleNamespace(text=part) for part in
                  (item if isinstance(item, list) else [item])]
        return types.SimpleNamespace(content=blocks, usage=self._usage)


def openai_provider(name="kimi", model="kimi-test", responses=("ok",),
                    usage=_Usage(3, 7), **kwargs) -> OpenAICompatibleProvider:
    client = FakeOpenAIClient(responses, usage)
    provider = OpenAICompatibleProvider(
        name, model, api_key="k", base_url="https://example/v1",
        client=client, **kwargs)
    provider.fake_client = client
    return provider


def anthropic_provider(name="claude", model="claude-test",
                       responses=("ok",)) -> AnthropicProvider:
    client = FakeAnthropicClient(responses)
    provider = AnthropicProvider(name, model, api_key="k", client=client)
    provider.fake_client = client
    return provider


class ScriptedProvider(Provider):
    """A provider that returns exactly what a test tells it to.

    Used where the point under test is the router's behaviour, not any wire
    format: how many times it was called, in what order, with what excluded.
    """

    kind = "scripted"

    def __init__(self, name="scripted", model="scripted-1", responses=("ok",),
                 available=True, fail=False):
        super().__init__(name, model)
        self._responses = list(responses)
        self._available = available
        self._fail = fail
        self.invocations: list[list[dict]] = []

    @property
    def available(self) -> bool:
        return self._available

    def unavailable_reason(self) -> str:
        return f"{self.name}: switched off by the test"

    async def _invoke(self, messages: list[dict], params: CallParams) -> Completion:
        self.invocations.append(list(messages))
        if self._fail:
            return Completion.failure(self.name, self.model, "scripted failure")
        text = self._responses.pop(0) if self._responses else ""
        if isinstance(text, Exception):
            raise text
        return Completion(text=text, provider=self.name, model=self.model,
                          tokens_in=10, tokens_out=20)


class FakeLease:
    """The lease shape the cortex checks for (spec M4.3).

    ``tokens`` is the allowance the lease actually holds; None means the lease
    does not declare one and the estimate is not checked against it.
    """

    def __init__(self, active=True, tokens=None):
        self.active = active
        self.tokens = tokens
        self.committed: list[tuple[int, int]] = []


class FakeResources:
    """Just enough resource manager for the budget gate."""

    def __init__(self):
        self.commits: list[tuple[int, int]] = []

    def commit_tokens(self, lease, tokens, calls=1):
        self.commits.append((tokens, calls))
        if hasattr(lease, "committed"):
            lease.committed.append((tokens, calls))
