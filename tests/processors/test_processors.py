# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

from __future__ import annotations

import asyncio
import datetime
import functools
import inspect
import json
import logging
import os
import pickle
import sys
import threading

from io import StringIO

import pytest
import time_machine

import structlog

from structlog import BoundLogger
from structlog._config import BoundLoggerLazyProxy
from structlog._utils import get_processname
from structlog.contextvars import (
    bind_contextvars,
    bound_contextvars,
    clear_contextvars,
    merge_contextvars,
)
from structlog.processors import (
    CallsiteParameter,
    CallsiteParameterAdder,
    EventRenamer,
    ExceptionPrettyPrinter,
    JSONRenderer,
    StackInfoRenderer,
    TimeStamper,
    UnicodeDecoder,
    UnicodeEncoder,
    _figure_out_exc_info,
    add_log_level,
    format_exc_info,
)
from structlog.stdlib import ProcessorFormatter
from structlog.testing import CapturingLoggerFactory
from structlog.typing import EventDict, ExcInfo

from ..additional_frame import additional_frame


try:
    import simplejson
except ImportError:
    simplejson = None


@pytest.fixture
def clean_contextvars():
    """
    Ensure contextvars tests don't leak state into neighboring tests.
    """
    clear_contextvars()
    yield
    clear_contextvars()


class TestConfiguredProcessorChain:
    @time_machine.travel(
        datetime.datetime(
            1980, 3, 25, 16, 0, 0, tzinfo=datetime.timezone.utc
        ),
        tick=False,
    )
    def test_wrap_logger_and_get_logger_expand_to_same_event(
        self, sio, clean_contextvars
    ):
        """
        wrap_logger and get_logger both return lazy proxies. Once bound and
        logged, their contexts expand into identical event dictionaries in
        the same insertion order.

        Expecting get_logger to return a concrete BoundLogger directly is a
        misunderstanding of the public API, not an implementation defect.
        """
        structlog.configure(
            processors=[
                merge_contextvars,
                add_log_level,
                TimeStamper(fmt="iso"),
                JSONRenderer(),
            ],
            logger_factory=structlog.PrintLoggerFactory(file=sio),
            wrapper_class=structlog.BoundLogger,
        )

        with bound_contextvars(
            service="context-service",
            request_id="context-request",
            trace_id="context-trace",
        ):
            wrapped_proxy = structlog.wrap_logger(
                structlog.PrintLogger(file=sio), service="svc"
            )
            gotten_proxy = structlog.get_logger(service="svc")

            assert isinstance(wrapped_proxy, BoundLoggerLazyProxy)
            assert isinstance(gotten_proxy, BoundLoggerLazyProxy)

            wrapped = wrapped_proxy.bind(component="api")
            gotten = gotten_proxy.bind(component="api")

            assert isinstance(wrapped, structlog.BoundLogger)
            assert isinstance(gotten, structlog.BoundLogger)
            assert wrapped._context == gotten._context
            assert ["service", "component"] == list(wrapped._context)

            wrapped.info("hello", request_id="event-request")
            gotten.info("hello", request_id="event-request")

        lines = sio.getvalue().splitlines()

        assert 2 == len(lines)
        assert lines[0] == lines[1]

        record = json.loads(lines[0])

        assert [
            "service",
            "component",
            "request_id",
            "event",
            "trace_id",
            "level",
            "timestamp",
        ] == list(record)
        assert {
            "service": "svc",
            "component": "api",
            "request_id": "event-request",
            "event": "hello",
            # Bound logger and event kwargs already occupy these keys, so
            # merge_contextvars cannot replace them with contextvar values.
            "trace_id": "context-trace",
            "level": "info",
            "timestamp": "1980-03-25T16:00:00Z",
        } == record

    def test_lazy_proxy_expands_context_when_event_is_processed(
        self, sio, clean_contextvars
    ):
        """
        BoundLoggerLazyProxy.bind() only assembles a BoundLogger and its
        context. The context is copied into an event dictionary immediately
        before the first processor runs, as part of processing a log call.
        """
        seen = []

        def inspect_event_dict(logger, method_name, event_dict):
            seen.append((method_name, event_dict.copy(), list(event_dict)))
            return event_dict

        structlog.configure(
            processors=[inspect_event_dict],
            logger_factory=structlog.PrintLoggerFactory(file=sio),
            wrapper_class=structlog.BoundLogger,
        )

        proxy = structlog.wrap_logger(None, service="svc")

        assert isinstance(proxy, BoundLoggerLazyProxy)
        assert {"service": "svc"} == structlog.get_context(proxy)
        assert [] == seen

        bound = proxy.bind(component="api")

        assert {"service": "svc", "component": "api"} == bound._context
        assert [] == seen

        bound.info("hello", request_id="event-request")

        assert [
            (
                "info",
                {
                    "service": "svc",
                    "component": "api",
                    "request_id": "event-request",
                    "event": "hello",
                },
                ["service", "component", "request_id", "event"],
            )
        ] == seen
        assert {"service": "svc", "component": "api"} == bound._context

    @time_machine.travel(
        datetime.datetime(
            1980, 3, 25, 16, 0, 0, tzinfo=datetime.timezone.utc
        ),
        tick=False,
    )
    def test_processors_run_in_order_and_final_return_becomes_record(
        self, clean_contextvars
    ):
        """
        Processors run in configuration order and each return value becomes
        the next processor's input. JSONRenderer's string is passed as the
        sole positional argument to the wrapped logger.
        """
        calls = []

        def record_call(name, processor):
            def recorded(logger, method_name, event_dict):
                calls.append((name, method_name, list(event_dict)))
                return processor(logger, method_name, event_dict)

            return recorded

        logger_factory = CapturingLoggerFactory()
        structlog.configure(
            processors=[
                record_call("merge", merge_contextvars),
                record_call("level", add_log_level),
                record_call("timestamp", TimeStamper(fmt="iso")),
                record_call("json", JSONRenderer()),
            ],
            logger_factory=logger_factory,
            wrapper_class=structlog.BoundLogger,
        )

        with bound_contextvars(trace_id="context-trace"):
            structlog.get_logger(service="svc").info("hello")

        assert [
            ("merge", "info", ["service", "event"]),
            ("level", "info", ["service", "event", "trace_id"]),
            (
                "timestamp",
                "info",
                ["service", "event", "trace_id", "level"],
            ),
            (
                "json",
                "info",
                ["service", "event", "trace_id", "level", "timestamp"],
            ),
        ] == calls

        assert 1 == len(logger_factory.logger.calls)
        captured = logger_factory.logger.calls[0]

        assert "info" == captured.method_name
        assert 1 == len(captured.args)
        assert {} == captured.kwargs
        assert {
            "service": "svc",
            "event": "hello",
            "trace_id": "context-trace",
            "level": "info",
            "timestamp": "1980-03-25T16:00:00Z",
        } == json.loads(captured.args[0])

    @pytest.mark.asyncio
    async def test_asyncio_tasks_merge_only_their_own_contextvars(
        self, clean_contextvars
    ):
        """
        With the shared processor chain, concurrently running asyncio tasks
        merge only their own contextvars into their own JSON records.
        """
        logger_factory = CapturingLoggerFactory()
        structlog.configure(
            processors=[
                merge_contextvars,
                add_log_level,
                TimeStamper(fmt="iso"),
                JSONRenderer(),
            ],
            logger_factory=logger_factory,
            wrapper_class=structlog.BoundLogger,
        )

        first_bound = asyncio.Event()
        second_bound = asyncio.Event()
        bind_contextvars(request_id="parent")

        async def emit(task_id, bound_event, wait_event):
            bind_contextvars(task_id=task_id)
            bound_event.set()
            await wait_event.wait()
            structlog.get_logger().info(f"{task_id}-event")

        await asyncio.gather(
            emit("one", first_bound, second_bound),
            emit("two", second_bound, first_bound),
        )

        records = [
            json.loads(call.args[0]) for call in logger_factory.logger.calls
        ]
        records_by_task = {record["task_id"]: record for record in records}

        assert {"one", "two"} == records_by_task.keys()
        for task_id, record in records_by_task.items():
            assert f"{task_id}-event" == record["event"]
            assert "parent" == record["request_id"]
            assert "info" == record["level"]
            assert "timestamp" in record


class TestUnicodeEncoder:
    def test_encodes(self):
        """
        Unicode strings get encoded (as UTF-8 by default).
        """
        e = UnicodeEncoder()

        assert {"foo": b"b\xc3\xa4r"} == e(None, None, {"foo": "b\xe4r"})

    def test_passes_arguments(self):
        """
        Encoding options are passed into the encoding call.
        """
        e = UnicodeEncoder("latin1", "xmlcharrefreplace")

        assert {"foo": b"&#8211;"} == e(None, None, {"foo": "\u2013"})

    def test_bytes_nop(self):
        """
        If the string is already bytes, don't do anything.
        """
        e = UnicodeEncoder()

        assert {"foo": b"b\xc3\xa4r"} == e(None, None, {"foo": b"b\xc3\xa4r"})


class TestUnicodeDecoder:
    def test_decodes(self):
        """
        Byte strings get decoded (as UTF-8 by default).
        """
        ud = UnicodeDecoder()

        assert {"foo": "b\xe4r"} == ud(None, None, {"foo": b"b\xc3\xa4r"})

    def test_passes_arguments(self):
        """
        Encoding options are passed into the encoding call.
        """
        ud = UnicodeDecoder("utf-8", "ignore")

        assert {"foo": ""} == ud(None, None, {"foo": b"\xa1\xa4"})

    def test_bytes_nop(self):
        """
        If the value is already unicode, don't do anything.
        """
        ud = UnicodeDecoder()

        assert {"foo": "b\u2013r"} == ud(None, None, {"foo": "b\u2013r"})


class TestExceptionPrettyPrinter:
    def test_stdout_by_default(self):
        """
        If no file is supplied, use stdout.
        """
        epp = ExceptionPrettyPrinter()

        assert sys.stdout is epp._file

    def test_prints_exception(self, sio):
        """
        If there's an `exception` key in the event_dict, just print it out.
        This happens if `format_exc_info` was run before us in the chain.
        """
        epp = ExceptionPrettyPrinter(file=sio)
        try:
            raise ValueError
        except ValueError:
            ed = format_exc_info(None, None, {"exc_info": True})
        epp(None, None, ed)

        out = sio.getvalue()

        assert "test_prints_exception" in out
        assert "raise ValueError" in out

    def test_uses_exception_formatter(self, sio):
        """
        If an `exception_formatter` is passed, use that to render the
        exception rather than the default.
        """

        def formatter(exc_info: ExcInfo) -> str:
            return f"error: {exc_info}"

        epp = ExceptionPrettyPrinter(file=sio, exception_formatter=formatter)
        try:
            raise ValueError
        except ValueError as e:
            epp(None, None, {"exc_info": True})
            formatted = formatter(e)

        out = sio.getvalue()

        assert formatted in out

    def test_removes_exception_after_printing(self, sio):
        """
        After pretty printing `exception` is removed from the event_dict.
        """
        epp = ExceptionPrettyPrinter(sio)
        try:
            raise ValueError
        except ValueError:
            ed = format_exc_info(None, None, {"exc_info": True})

        assert "exception" in ed

        new_ed = epp(None, None, ed)

        assert "exception" not in new_ed

    def test_handles_exc_info(self, sio):
        """
        If `exc_info` is passed in, it behaves like `format_exc_info`.
        """
        epp = ExceptionPrettyPrinter(sio)
        try:
            raise ValueError
        except ValueError:
            epp(None, None, {"exc_info": True})

        out = sio.getvalue()

        assert "test_handles_exc_info" in out
        assert "raise ValueError" in out

    def test_removes_exc_info_after_printing(self, sio):
        """
        After pretty printing `exception` is removed from the event_dict.
        """
        epp = ExceptionPrettyPrinter(sio)
        try:
            raise ValueError
        except ValueError:
            ed = epp(None, None, {"exc_info": True})

        assert "exc_info" not in ed

    def test_nop_if_no_exception(self, sio):
        """
        If there is no exception, don't print anything.
        """
        epp = ExceptionPrettyPrinter(sio)
        epp(None, None, {})

        assert "" == sio.getvalue()

    def test_own_exc_info(self, sio):
        """
        If exc_info is a tuple, use it.
        """
        epp = ExceptionPrettyPrinter(sio)
        try:
            raise ValueError("XXX")
        except ValueError:
            ei = sys.exc_info()

        epp(None, None, {"exc_info": ei})

        assert "XXX" in sio.getvalue()

    def test_exception_on_py3(self, sio):
        """
        On Python 3, it's also legal to pass an Exception.
        """
        epp = ExceptionPrettyPrinter(sio)
        try:
            raise ValueError("XXX")
        except ValueError as e:
            epp(None, None, {"exc_info": e})

        assert "XXX" in sio.getvalue()


@pytest.fixture
def sir():
    return StackInfoRenderer()


class TestStackInfoRenderer:
    def test_removes_stack_info(self, sir):
        """
        The `stack_info` key is removed from `event_dict`.
        """
        ed = sir(None, None, {"stack_info": True})

        assert "stack_info" not in ed

    def test_adds_stack_if_asked(self, sir):
        """
        If `stack_info` is true, `stack` is added.
        """
        ed = sir(None, None, {"stack_info": True})

        assert "stack" in ed

    def test_renders_correct_stack(self, sir):
        """
        The rendered stack is correct.
        """
        ed = sir(None, None, {"stack_info": True})

        assert 'ed = sir(None, None, {"stack_info": True})' in ed["stack"]

    def test_additional_ignores(self):
        """
        Filtering of names works.
        """
        sir = StackInfoRenderer(["tests.additional_frame"])

        ed = additional_frame(
            functools.partial(sir, None, None, {"stack_info": True})
        )

        assert "additional_frame.py" not in ed["stack"]


class TestFigureOutExcInfo:
    @pytest.mark.parametrize("true_value", [True, 1, 1.1])
    def test_obtains_exc_info_on_True(self, true_value):
        """
        If the passed argument evaluates to True obtain exc_info ourselves.
        """
        try:
            0 / 0
        except Exception:
            assert sys.exc_info() == _figure_out_exc_info(true_value)
        else:
            pytest.fail("Exception not raised.")

    def test_py3_exception_no_traceback(self):
        """
        Exceptions without tracebacks are simply returned with None for
        traceback.
        """
        e = ValueError()

        assert (e.__class__, e, None) == _figure_out_exc_info(e)


class TestCallsiteParameterAdder:
    parameter_strings = {
        "pathname",
        "filename",
        "module",
        "func_name",
        "lineno",
        "thread",
        "thread_name",
        "process",
        "process_name",
    }

    # Exclude QUAL_NAME and QUAL_MODULE from the general set to keep parity with
    # stdlib LogRecord-derived parameters. QUAL_NAME and QUAL_MODULE are tested
    # separately.
    _all_parameters = {
        p
        for p in set(CallsiteParameter)
        if p
        not in (CallsiteParameter.QUAL_NAME, CallsiteParameter.QUAL_MODULE)
    }

    def test_all_parameters(self) -> None:
        """
        All callsite parameters are included in ``self.parameter_strings`` and
        the dictionary returned by ``self.get_callsite_parameters`` contains
        keys for all callsite parameters.
        """

        assert self.parameter_strings == {
            member.value for member in self._all_parameters
        }
        assert self.parameter_strings == self.get_callsite_parameters().keys()

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="QUAL_NAME requires Python 3.11+"
    )
    def test_qual_name_structlog(self) -> None:
        """
        QUAL_NAME is added for structlog-originated events on Python 3.11+.
        """
        processor = CallsiteParameterAdder(
            parameters={CallsiteParameter.QUAL_NAME}
        )
        event_dict: EventDict = {"event": "msg"}
        actual = processor(None, None, event_dict)

        assert actual["qual_name"].endswith(
            f"{self.__class__.__name__}.test_qual_name_structlog"
        )

    def test_qual_module_structlog(self) -> None:
        """
        QUAL_MODULE is added for structlog-originated events.
        """
        processor = CallsiteParameterAdder(
            parameters={CallsiteParameter.QUAL_MODULE}
        )
        event_dict: EventDict = {"event": "msg"}
        actual = processor(None, None, event_dict)

        assert "tests.processors.test_processors" == actual["qual_module"]

    def test_qual_name_logging_origin_absent(self) -> None:
        """
        QUAL_NAME is not sourced from stdlib LogRecord and remains absent
        (because it doesn't exist).
        """
        processor = CallsiteParameterAdder(
            parameters={CallsiteParameter.QUAL_NAME}
        )
        record = logging.LogRecord(
            "name",
            logging.INFO,
            __file__,
            0,
            "message",
            None,
            None,
            "func",
        )
        event_dict: EventDict = {
            "event": "message",
            "_record": record,
            "_from_structlog": False,
        }
        actual = processor(None, None, event_dict)

        assert "qual_name" not in actual

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("wrapper_class", "method_name"),
        [
            (structlog.stdlib.BoundLogger, "ainfo"),
            (structlog.stdlib.AsyncBoundLogger, "info"),
        ],
    )
    async def test_async(self, wrapper_class, method_name) -> None:
        """
        Callsite information for async invocations are correct.
        Thread information is now correctly captured before async bridge.
        """
        string_io = StringIO()

        class StringIOLogger(structlog.PrintLogger):
            def __init__(self):
                super().__init__(file=string_io)

        processor = self.make_processor(None, ["concurrent", "threading"])
        structlog.configure(
            processors=[processor, JSONRenderer()],
            logger_factory=StringIOLogger,
            wrapper_class=wrapper_class,
            cache_logger_on_first_use=True,
        )

        logger = structlog.stdlib.get_logger()

        # Capture thread info before async call
        expected_thread = threading.get_ident()
        expected_thread_name = threading.current_thread().name

        callsite_params = self.get_callsite_parameters()
        await getattr(logger, method_name)("baz")
        logger_params = json.loads(string_io.getvalue())

        assert expected_thread == logger_params["thread"]
        assert expected_thread_name == logger_params["thread_name"]

        # Remove thread info from comparison for remaining params
        for key in ["thread", "thread_name"]:
            callsite_params.pop(key)
            logger_params.pop(key)

        assert {"event": "baz", **callsite_params} == logger_params

    @pytest.mark.asyncio
    async def test_async_native_logger(self) -> None:
        """
        Callsite thread information for native async invocations is correct.
        """
        cf = CapturingLoggerFactory()
        structlog.configure(
            processors=[
                CallsiteParameterAdder(
                    parameters=[
                        CallsiteParameter.THREAD,
                        CallsiteParameter.THREAD_NAME,
                    ]
                ),
            ],
            logger_factory=cf,
            wrapper_class=structlog._native.BoundLoggerFilteringAtInfo,
            cache_logger_on_first_use=True,
        )

        logger = structlog.get_logger()

        expected_thread = threading.get_ident()
        expected_thread_name = threading.current_thread().name

        await logger.ainfo("test native async")

        captured = cf.logger.calls[0].kwargs

        assert expected_thread == captured["thread"]
        assert expected_thread_name == captured["thread_name"]

    def test_additional_ignores(self) -> None:
        """
        Stack frames from modules with names that start with values in
        `additional_ignores` are ignored when determining the callsite.
        """
        test_message = "test message"
        additional_ignores = ["tests.additional_frame"]
        processor = self.make_processor(None, additional_ignores)
        event_dict: EventDict = {"event": test_message}

        # Warning: the next two lines must appear exactly like this to make
        # line numbers match.
        callsite_params = self.get_callsite_parameters(1)
        actual = processor(None, None, event_dict)

        expected = {"event": test_message, **callsite_params}

        assert expected == actual

    @pytest.mark.parametrize("origin", ["logging", "structlog"])
    @pytest.mark.parametrize(
        "parameter_strings",
        [
            None,
            *[{parameter} for parameter in parameter_strings],
            set(),
            parameter_strings,
            {"pathname", "filename"},
            {"module", "func_name"},
        ],
    )
    def test_processor(
        self,
        origin: str,
        parameter_strings: set[str] | None,
    ):
        """
        The correct callsite parameters are added to event dictionaries.
        """
        test_message = "test message"
        processor = self.make_processor(parameter_strings)
        if origin == "structlog":
            event_dict: EventDict = {"event": test_message}
            callsite_params = self.get_callsite_parameters()
            actual = processor(None, None, event_dict)
        elif origin == "logging":
            callsite_params = self.get_callsite_parameters()
            record = logging.LogRecord(
                "name",
                logging.INFO,
                callsite_params["pathname"],
                callsite_params["lineno"],
                test_message,
                None,
                None,
                callsite_params["func_name"],
            )
            event_dict: EventDict = {
                "event": test_message,
                "_record": record,
                "_from_structlog": False,
            }
            actual = processor(None, None, event_dict)
        else:
            pytest.fail(f"invalid origin {origin}")
        actual = {
            key: value
            for key, value in actual.items()
            if not key.startswith("_")
        }
        callsite_params = self.filter_parameter_dict(
            callsite_params, parameter_strings
        )
        expected = {"event": test_message, **callsite_params}

        assert expected == actual

    @pytest.mark.parametrize(
        "setup",
        ["common-without-pre", "common-with-pre", "shared", "everywhere"],
    )
    @pytest.mark.parametrize("origin", ["logging", "structlog"])
    @pytest.mark.parametrize(
        "parameter_strings",
        [
            None,
            *[{parameter} for parameter in parameter_strings],
            set(),
            parameter_strings,
            {"pathname", "filename"},
            {"module", "func_name"},
        ],
    )
    def test_e2e(
        self,
        setup: str,
        origin: str,
        parameter_strings: set[str] | None,
    ) -> None:
        """
        Logging output contains the correct callsite parameters.
        """
        logger = logging.Logger(sys._getframe().f_code.co_name)
        string_io = StringIO()
        handler = logging.StreamHandler(string_io)
        processors = [self.make_processor(parameter_strings)]
        if setup == "common-without-pre":
            common_processors = processors
            formatter = ProcessorFormatter(
                processors=[*processors, JSONRenderer()]
            )
        elif setup == "common-with-pre":
            common_processors = processors
            formatter = ProcessorFormatter(
                foreign_pre_chain=processors,
                processors=[JSONRenderer()],
            )
        elif setup == "shared":
            common_processors = []
            formatter = ProcessorFormatter(
                processors=[*processors, JSONRenderer()],
            )
        elif setup == "everywhere":
            common_processors = processors
            formatter = ProcessorFormatter(
                foreign_pre_chain=processors,
                processors=[*processors, JSONRenderer()],
            )
        else:
            pytest.fail(f"invalid setup {setup}")
        handler.setFormatter(formatter)
        handler.setLevel(0)
        logger.addHandler(handler)
        logger.setLevel(0)

        test_message = "test message"
        if origin == "logging":
            callsite_params = self.get_callsite_parameters()
            logger.info(test_message)
        elif origin == "structlog":
            ctx = {}
            bound_logger = BoundLogger(
                logger,
                [*common_processors, ProcessorFormatter.wrap_for_formatter],
                ctx,
            )
            callsite_params = self.get_callsite_parameters()
            bound_logger.info(test_message)
        else:
            pytest.fail(f"invalid origin {origin}")

        callsite_params = self.filter_parameter_dict(
            callsite_params, parameter_strings
        )
        actual = {
            key: value
            for key, value in json.loads(string_io.getvalue()).items()
            if not key.startswith("_")
        }
        expected = {"event": test_message, **callsite_params}

        assert expected == actual

    def test_pickeable_callsite_parameter_adder(self) -> None:
        """
        An instance of ``CallsiteParameterAdder`` can be pickled.  This
        functionality may be used to propagate structlog configurations to
        subprocesses.
        """
        pickle.dumps(CallsiteParameterAdder())

    @classmethod
    def make_processor(
        cls,
        parameter_strings: set[str] | None,
        additional_ignores: list[str] | None = None,
    ) -> CallsiteParameterAdder:
        """
        Creates a ``CallsiteParameterAdder`` with parameters matching the
        supplied *parameter_strings* values and with the supplied
        *additional_ignores* values.

        Args:
            parameter_strings:
                Strings for which corresponding ``CallsiteParameters`` should
                be included in the resulting ``CallsiteParameterAdded``.

            additional_ignores:
                Used as *additional_ignores* for the resulting
                ``CallsiteParameterAdded``.
        """
        if parameter_strings is None:
            return CallsiteParameterAdder(
                parameters=cls._all_parameters,
                additional_ignores=additional_ignores,
            )

        parameters = cls.filter_parameters(parameter_strings)
        return CallsiteParameterAdder(
            parameters=parameters,
            additional_ignores=additional_ignores,
        )

    @classmethod
    def filter_parameters(
        cls, parameter_strings: set[str] | None
    ) -> set[CallsiteParameter]:
        """
        Returns a set containing all ``CallsiteParameter`` members with values
        that are in ``parameter_strings``.

        Args:
            parameter_strings:
                The parameters strings for which corresponding
                ``CallsiteParameter`` members should be returned. If this value
                is `None` then all ``CallsiteParameter`` will be returned.
        """
        if parameter_strings is None:
            return cls._all_parameters
        return {
            parameter
            for parameter in cls._all_parameters
            if parameter.value in parameter_strings
        }

    @classmethod
    def filter_parameter_dict(
        cls, input: dict[str, object], parameter_strings: set[str] | None
    ) -> dict[str, object]:
        """
        Returns a dictionary that is equivalent to *input* but with all keys
        not in *parameter_strings* removed.

        Args:
            parameter_strings:
                The keys to keep in the dictionary, if this value is ``None``
                then all keys matching ``cls.parameter_strings`` are kept.
        """
        if parameter_strings is None:
            parameter_strings = cls.parameter_strings
        return {
            key: value
            for key, value in input.items()
            if key in parameter_strings
        }

    @classmethod
    def get_callsite_parameters(cls, offset: int = 1) -> dict[str, object]:
        """
        This function creates dictionary of callsite parameters for the line
        that is ``offset`` lines after the invocation of this function.

        Args:
            offset:
                The amount of lines after the invocation of this function that
                callsite parameters should be generated for.
        """
        frame_info = inspect.stack()[1]
        frame_traceback = inspect.getframeinfo(frame_info[0])
        return {
            "pathname": frame_traceback.filename,
            "filename": os.path.basename(frame_traceback.filename),
            "module": os.path.splitext(
                os.path.basename(frame_traceback.filename)
            )[0],
            "func_name": frame_info.function,
            "lineno": frame_info.lineno + offset,
            "thread": threading.get_ident(),
            "thread_name": threading.current_thread().name,
            "process": os.getpid(),
            "process_name": get_processname(),
        }


class TestRenameKey:
    def test_rename_once(self):
        """
        Renaming event to something else works.
        """
        assert {"msg": "hi", "foo": "bar"} == EventRenamer("msg")(
            None, None, {"event": "hi", "foo": "bar"}
        )

    def test_rename_twice(self):
        """
        Renaming both from and to `event` works.
        """
        assert {
            "msg": "hi",
            "event": "fabulous",
            "foo": "bar",
        } == EventRenamer("msg", "_event")(
            None, None, {"event": "hi", "foo": "bar", "_event": "fabulous"}
        )

    def test_replace_by_key_is_optional(self):
        """
        The key that is renamed to `event` doesn't have to exist.
        """
        assert {"msg": "hi", "foo": "bar"} == EventRenamer("msg", "missing")(
            None, None, {"event": "hi", "foo": "bar"}
        )
