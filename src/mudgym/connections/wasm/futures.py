"""Collect concurrent WASM work without losing failures or interruptions."""

from collections.abc import Callable, Iterable
from concurrent.futures import Future, wait


def ordered_future_results[Result](
    futures: Iterable[Future[Result]], operation: str, *, submitted: list[Future[Result]] | None = None
) -> list[Result]:
    """Submit and collect in order, settling returned futures before raising.

    ``submitted`` retains each returned future so the caller can roll back successful work after a failure.
    """
    if submitted is None:
        submitted = []
    try:
        for future in futures:
            submitted.append(future)
        return [future.result() for future in submitted]
    except BaseException as first_error:
        for future in submitted:
            future.cancel()
        wait(submitted)
        additional_errors = []
        for future in submitted:
            if future.cancelled():
                continue
            try:
                future.result()
            except BaseException as error:  # noqa: BLE001 - retain interruptions in the group raised below
                if error is not first_error:
                    additional_errors.append(error)
        if additional_errors:
            raise BaseExceptionGroup(operation, [first_error, *additional_errors]) from first_error
        raise


class InlineExecutor:
    """Runs each piece of work as it is submitted, for a Python that cannot start threads, such as Pyodide's."""

    def submit[Result](self, work: Callable[..., Result], /, *arguments: object) -> Future[Result]:
        future: Future[Result] = Future()
        try:
            future.set_result(work(*arguments))
        except BaseException as error:  # noqa: BLE001 - the caller reads every failure back through the future
            future.set_exception(error)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Nothing is ever left running, so there is nothing to wait for or cancel."""
