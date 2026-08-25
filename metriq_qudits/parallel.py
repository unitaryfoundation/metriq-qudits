"""
Shared process-pool helpers used by the pipeline stages.
"""

from __future__ import annotations

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

_MP_CONTEXT = multiprocessing.get_context("spawn")


def milestone_progress(label: str):
    """Return a callback that prints at roughly 25% completion increments."""
    reported: set[int] = set()

    def report(completed: int, total: int) -> None:
        if total <= 0:
            return
        milestones = {max(1, math.ceil(total * fraction))
                      for fraction in (0.25, 0.5, 0.75, 1.0)}
        reached = {point for point in milestones if completed >= point}
        new = reached - reported
        if not new:
            return
        reported.update(new)
        print(f"      {label}: {completed}/{total} ({completed / total:.0%})",
              flush=True)

    return report


def parallel_map(function, jobs, n_jobs: int, on_progress=None):
    """Map ``function`` over ``jobs`` and optionally report completed items.

    Results preserve input order even with multiple workers, while progress is
    reported in actual completion order so one slow early job cannot make the
    terminal appear stalled.
    """
    jobs = list(jobs)
    total = len(jobs)
    if n_jobs > 1:
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=_MP_CONTEXT,
        ) as pool:
            futures = {pool.submit(function, job): i for i, job in enumerate(jobs)}
            results = [None] * total
            for completed, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if on_progress is not None:
                    on_progress(completed, total)
            yield from results
    else:
        for completed, job in enumerate(jobs, start=1):
            yield function(job)
            if on_progress is not None:
                on_progress(completed, total)


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
