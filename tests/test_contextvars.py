# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

import asyncio
import inspect
import secrets

import pytest

import structlog

from structlog.contextvars import (
    _CONTEXT_VARS,
    bind_contextvars,
    bound_contextvars,
    clear_contextvars,
    get_contextvars,
    get_merged_contextvars,
    merge_contextvars,
    reset_contextvars,
    unbind_contextvars,
)


@pytest.fixture(autouse=True)
def _clear_contextvars():
    """
    Make sure all tests start with a clean slate.
    """
    clear_contextvars()


class TestContextvars:
    async def test_bind(self):
        """
        Binding a variable causes it to be included in the result of
        merge_contextvars.
        """
        event_loop = asyncio.get_running_loop()

        async def coro():
            bind_contextvars(a=1)
            return merge_contextvars(None, None, {"b": 2})

        assert {"a": 1, "b": 2} == await event_loop.create_task(coro())

    async def test_multiple_binds(self):
        """
        Multiple calls to bind_contextvars accumulate values instead of
        replacing them. But they override redefined ones.
        """
        event_loop = asyncio.get_running_loop()

        async def coro():
            bind_contextvars(a=1, c=3)
            bind_contextvars(c=333, d=4)
            return merge_contextvars(None, None, {"b": 2})

        assert {
            "a": 1,
            "b": 2,
            "c": 333,
            "d": 4,
        } == await event_loop.create_task(coro())

    async def test_reset(self):
        """
        reset_contextvars allows resetting contexvars to
        previously-set values.
        """
        event_loop = asyncio.get_running_loop()

        async def coro():
            bind_contextvars(a=1)

            assert {"a": 1} == get_contextvars()

            await event_loop.create_task(nested_coro())

        async def nested_coro():
            tokens = bind_contextvars(a=2, b=3)

            assert {"a": 2, "b": 3} == get_contextvars()

            reset_contextvars(**tokens)

            assert {"a": 1} == get_contextvars()

        await event_loop.create_task(coro())

    async def test_nested_async_bind(self):
        """
        Context is passed correctly between "nested" concurrent operations.
        """
        event_loop = asyncio.get_running_loop()

        async def coro():
            bind_contextvars(a=1)
            return await event_loop.create_task(nested_coro())

        async def nested_coro():
            bind_contextvars(c=3)
            return merge_contextvars(None, None, {"b": 2})

        assert {"a": 1, "b": 2, "c": 3} == await event_loop.create_task(coro())

    async def test_merge_works_without_bind(self):
        """
        merge_contextvars returns values as normal even when there has
        been no previous calls to bind_contextvars.
        """
        event_loop = asyncio.get_running_loop()

        async def coro():
            return merge_contextvars(None, None, {"b": 2})

        assert {"b": 2} == await event_loop.create_task(coro())

    async def test_merge_overrides_bind(self):
        """
        Variables included in merge_contextvars override previously
        bound variables.
        """
        event_loop = asyncio.get_running_loop()

        async def coro():
            bind_contextvars(a=1)
            return merge_contextvars(None, None, {"a": 111, "b": 2})

        assert {"a": 111, "b": 2} == await event_loop.create_task(coro())

    async def test_clear(self):
        """
        The context-local context can be cleared, causing any previously bound
        variables to not be included in merge_contextvars's result.
        """
        event_loop = asyncio.get_running_loop()

        async def coro():
            bind_contextvars(a=1)
            clear_contextvars()
            return merge_contextvars(None, None, {"b": 2})

        assert {"b": 2} == await event_loop.create_task(coro())

    async def test_clear_without_bind(self):
        """
        The context-local context can be cleared, causing any previously bound
        variables to not be included in merge_contextvars's result.
        """
        event_loop = asyncio.get_running_loop()

        async def coro():
            clear_contextvars()
            return merge_contextvars(None, None, {})

        assert {} == await event_loop.create_task(coro())

    async def test_unbind(self):
        """
        Unbinding a previously bound variable causes it to be removed from the
        result of merge_contextvars.
        """
        event_loop = asyncio.get_running_loop()

        async def coro():
            bind_contextvars(a=1)
            unbind_contextvars("a")
            return merge_contextvars(None, None, {"b": 2})

        assert {"b": 2} == await event_loop.create_task(coro())

    async def test_unbind_not_bound(self):
        """
        Unbinding a not bound variable causes doesn't raise an exception.
        """
        event_loop = asyncio.get_running_loop()

        async def coro():
            # Since unbinding means "setting to Ellipsis", we have to make
            # some effort to ensure that the ContextVar never existed.
            unbind_contextvars("a" + secrets.token_hex())

            return merge_contextvars(None, None, {"b": 2})

        assert {"b": 2} == await event_loop.create_task(coro())

    async def test_parallel_binds(self):
        """
        Binding a variable causes it to be included in the result of
        merge_contextvars.
        """
        event_loop = asyncio.get_running_loop()
        coro1_bind = asyncio.Event()
        coro2_bind = asyncio.Event()

        bind_contextvars(c=3)

        async def coro1():
            bind_contextvars(a=1)

            coro1_bind.set()
            await coro2_bind.wait()

            return merge_contextvars(None, None, {"b": 2})

        async def coro2():
            bind_contextvars(a=2)

            await coro1_bind.wait()
            coro2_bind.set()

            return merge_contextvars(None, None, {"b": 2})

        coro1_task = event_loop.create_task(coro1())
        coro2_task = event_loop.create_task(coro2())

        assert {"a": 1, "b": 2, "c": 3} == await coro1_task
        assert {"a": 2, "b": 2, "c": 3} == await coro2_task

    def test_get_only_gets_structlog_without_deleted(self):
        """
        get_contextvars returns only the structlog-specific key-values with
        the prefix removed. Deleted keys (= Ellipsis) are ignored.
        """
        bind_contextvars(a=1, b=2)
        unbind_contextvars("b")
        _CONTEXT_VARS["foo"] = "bar"

        assert {"a": 1} == get_contextvars()

    def test_get_merged_merges_context(self):
        """
        get_merged_contextvars merges a bound context into the copy.
        """
        bind_contextvars(x=1)
        log = structlog.get_logger().bind(y=2)

        assert {"x": 1, "y": 2} == get_merged_contextvars(log)


class TestBoundContextvars:
    def test_cleanup(self):
        """
        Bindings are cleaned up
        """
        with bound_contextvars(x=42, y="foo"):
            assert {"x": 42, "y": "foo"} == get_contextvars()

        assert {} == get_contextvars()

    def test_cleanup_conflict(self):
        """
        Overwritten keys are restored after the clean up
        """
        bind_contextvars(x="original", z="unrelated")
        with bound_contextvars(x=42, y="foo"):
            assert {"x": 42, "y": "foo", "z": "unrelated"} == get_contextvars()

        assert {"x": "original", "z": "unrelated"} == get_contextvars()

    def test_preserve_independent_bind(self):
        """
        New bindings inside bound_contextvars are preserved after the clean up
        """
        with bound_contextvars(x=42):
            bind_contextvars(y="foo")
            assert {"x": 42, "y": "foo"} == get_contextvars()

        assert {"y": "foo"} == get_contextvars()

    def test_nesting_works(self):
        """
        bound_contextvars binds and unbinds even when nested
        """
        with bound_contextvars(l1=1):
            assert {"l1": 1} == get_contextvars()

            with bound_contextvars(l2=2):
                assert {"l1": 1, "l2": 2} == get_contextvars()

            assert {"l1": 1} == get_contextvars()

        assert {} == get_contextvars()

    def test_as_decorator(self):
        """
        bound_contextvars can be used as a decorator and it preserves the
        name, signature and documentation of the wrapped function.
        """

        @bound_contextvars(x=42)
        def wrapped(arg1):
            """Wrapped documentation"""
            bind_contextvars(y=arg1)
            assert {"x": 42, "y": arg1} == get_contextvars()

        wrapped(23)

        assert "wrapped" == wrapped.__name__
        assert "(arg1)" == str(inspect.signature(wrapped))
        assert "Wrapped documentation" == wrapped.__doc__


class TestAsyncioTaskIsolation:
    """
    Pin down how context-local context behaves across asyncio tasks:

    ``asyncio.create_task`` copies the current context at creation time, so
    binds are task-local and never leak into concurrently running siblings.
    """

    async def test_sibling_tasks_do_not_leak(self):
        """
        Keys bound inside one task are invisible to a concurrently running
        sibling task, while keys bound in the parent before task creation
        are visible in both.
        """
        bind_contextvars(request_id="req-1")

        async def worker(name):
            bind_contextvars(worker=name)
            # Yield so both tasks bind before either merges.
            await asyncio.sleep(0.01)
            return merge_contextvars(None, None, {})

        results = await asyncio.gather(
            asyncio.create_task(worker("one")),
            asyncio.create_task(worker("two")),
        )

        assert [
            {"request_id": "req-1", "worker": "one"},
            {"request_id": "req-1", "worker": "two"},
        ] == results

    async def test_bind_after_task_creation_is_invisible(self):
        """
        A task sees the context as of its creation: keys bound in the
        parent *after* create_task do not propagate into the task.
        """

        async def worker():
            await asyncio.sleep(0.01)
            return get_contextvars()

        task = asyncio.create_task(worker())
        bind_contextvars(late=1)

        assert {} == await task

    async def test_child_binds_do_not_propagate_to_parent(self):
        """
        Keys bound inside a task are gone from the parent once the task
        has finished.
        """

        async def worker():
            bind_contextvars(inner=1)

        await asyncio.create_task(worker())

        assert {} == get_contextvars()


class TestBindUnbindResetSemantics:
    """
    Pin down what nested bind_contextvars / unbind_contextvars /
    reset_contextvars actually merge.
    """

    def test_unbind_does_not_restore_outer_value(self):
        """
        unbind_contextvars is *not* scoped: after a nested re-bind,
        unbinding removes the key entirely (an Ellipsis tombstone) instead
        of restoring the outer value.

        NOTE: a stack-like "unbind restores the previous value" mental
        model is a misunderstanding on our side, not an implementation
        defect. Use the tokens returned by bind_contextvars together with
        reset_contextvars for scope-like restoration.
        """
        bind_contextvars(a=1)
        bind_contextvars(a=2)
        unbind_contextvars("a")

        assert {} == get_contextvars()
        assert {"b": 2} == merge_contextvars(None, None, {"b": 2})

    def test_reset_restores_value_from_token_creation_time(self):
        """
        reset_contextvars restores each key to the value it had when its
        token was created -- even if the key was unbound in the meantime.

        Note that the token must come from the *second* bind: a token
        restores the state from *before its own bind_contextvars call*,
        not the value that bind installed.
        """
        bind_contextvars(a=1)
        tokens = bind_contextvars(a=2)
        unbind_contextvars("a")

        reset_contextvars(**tokens)

        assert {"a": 1} == get_contextvars()

    def test_reset_to_absent_removes_key(self):
        """
        A token created while the key was unbound resets the key back to
        "absent" (the Ellipsis tombstone), so it disappears from merges.
        """
        tokens = bind_contextvars(a=1)

        assert {"a": 1} == get_contextvars()

        reset_contextvars(**tokens)

        assert {} == get_contextvars()

    def test_reset_only_touches_passed_tokens(self):
        """
        reset_contextvars leaves keys without a passed token alone.
        """
        bind_contextvars(a=1)
        tokens = bind_contextvars(a=2)
        bind_contextvars(b=3)

        reset_contextvars(**tokens)

        assert {"a": 1, "b": 3} == get_contextvars()

    def test_merge_mutates_and_returns_same_dict(self):
        """
        merge_contextvars merges into the passed event_dict in place and
        returns the identical object. The merge is a snapshot: binds that
        happen afterwards don't affect the returned dict.
        """
        bind_contextvars(a=1)
        event_dict = {"b": 2}

        result = merge_contextvars(None, None, event_dict)

        assert result is event_dict
        assert {"a": 1, "b": 2} == result

        bind_contextvars(c=3)

        assert {"a": 1, "b": 2} == result

    def test_event_dict_wins_over_context(self):
        """
        merge_contextvars uses setdefault: keys already present in the
        event_dict (e.g. from BoundLogger.bind or log-call kwargs) are
        *not* overwritten by the context-local context.

        NOTE: the naive guess "bind_contextvars overrides per-call values"
        is a misunderstanding on our side; the precedence is intentional.
        """
        bind_contextvars(a=1)

        assert {"a": 111} == merge_contextvars(None, None, {"a": 111})
