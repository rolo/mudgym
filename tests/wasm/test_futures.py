"""Concurrent WASM failures preserve the original exceptions."""

from concurrent.futures import Future

import pytest

from mudgym.connections.wasm.futures import ordered_future_results


@pytest.mark.parametrize("first_error", [KeyboardInterrupt(), SystemExit(17), ValueError("first")])
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
