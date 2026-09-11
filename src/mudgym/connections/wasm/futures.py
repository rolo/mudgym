"""Collect concurrent WASM work without losing failures or interruptions."""

from collections.abc import Iterable
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
            except BaseException as error:
                if error is not first_error:
                    additional_errors.append(error)
        if additional_errors:
            raise BaseExceptionGroup(operation, [first_error, *additional_errors]) from first_error
        raise
