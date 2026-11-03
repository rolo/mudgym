"""Concurrent WASM failures preserve the original exceptions."""

from concurrent.futures import Future

import pytest

from mudgym.connections.wasm.futures import InlineExecutor, ordered_future_results


@pytest.mark.parametrize("first_error", [KeyboardInterrupt(), ValueError("first")])
def test_future_error_collection_preserves_interruptions(first_error):
    first, second = Future(), Future()
    second_error = RuntimeError("second")
    first.set_exception(first_error)
    second.set_exception(second_error)
    with pytest.raises(BaseExceptionGroup) as raised:
        ordered_future_results([first, second], "test operation")
    assert raised.value.exceptions == (first_error, second_error)
    assert raised.value.__cause__ is first_error
    assert isinstance(raised.value, ExceptionGroup) == isinstance(first_error, Exception)


def test_submission_interruption_preserves_a_submitted_worker_failure():
    worker = Future()
    worker_error = RuntimeError("worker failed")
    interruption = KeyboardInterrupt("submission interrupted")
    worker.set_exception(worker_error)

    def submissions():
        yield worker
        raise interruption

    with pytest.raises(BaseException) as raised:
        ordered_future_results(submissions(), "test operation")
    assert isinstance(raised.value, BaseExceptionGroup)
    assert raised.value.exceptions == (interruption, worker_error)
    assert raised.value.__cause__ is interruption


def test_inline_work_settles_its_future_before_submit_returns():
    executor = InlineExecutor()
    order = []

    done = executor.submit(lambda value: order.append(value) or value * 2, 21)
    order.append("submitted")

    assert order == [21, "submitted"]
    assert done.result() == 42


def test_inline_work_that_raises_reaches_the_collector_through_its_future():
    def fail():
        raise KeyboardInterrupt

    failed = InlineExecutor().submit(fail)

    assert failed.done()
    with pytest.raises(KeyboardInterrupt):
        ordered_future_results([failed], "inline operation")
