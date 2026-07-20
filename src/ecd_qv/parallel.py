"""
Shared process-pool helpers used by the pipeline stages.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

_MP_CONTEXT = multiprocessing.get_context("spawn")


def parallel_map(function, jobs, n_jobs: int):
    """Map ``function`` over ``jobs``, using a spawn pool when ``n_jobs > 1``."""
    if n_jobs > 1:
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=_MP_CONTEXT,
        ) as pool:
            yield from pool.map(function, jobs)
    else:
        yield from map(function, jobs)


def parallel_map_unordered(function, keyed_jobs: dict, n_jobs: int):
    """Yield ``(key, result)`` as each keyed job finishes (order not preserved)."""
    if n_jobs > 1:
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=_MP_CONTEXT,
        ) as pool:
            futures = {
                pool.submit(function, arguments): key
                for key, arguments in keyed_jobs.items()
            }
            for future in as_completed(futures):
                yield futures[future], future.result()
    else:
        for key, arguments in keyed_jobs.items():
            yield key, function(arguments)
