from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BenchmarkResult:
    seconds: float
    value: Any


def benchmark_callable(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 1,
    repeats: int = 1,
    **kwargs: Any,
) -> BenchmarkResult:
    for _ in range(max(warmup, 0)):
        fn(*args, **kwargs)

    start = time.perf_counter()
    value = None
    for _ in range(max(repeats, 1)):
        value = fn(*args, **kwargs)
    seconds = (time.perf_counter() - start) / max(repeats, 1)
    return BenchmarkResult(seconds=seconds, value=value)
