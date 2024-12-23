from dataclasses import dataclass
from typing import Optional, Sequence

from cupy.cuda import Event, get_elapsed_time


@dataclass
class Perf:
    """Performance information."""

    total_ns: int
    gpu_ms: Optional[float] = None
    buffer_size: Optional[int] = None
    buffer_size_is_tight: bool = False

    @staticmethod
    def from_measurements(total: tuple[int, int],
                          gpu: Optional[tuple[Event, Event]],
                          buffer_size: Optional[int],
                          buffer_size_is_tight: bool = False) -> "Perf":
        return Perf(total_ns=total[1] - total[0],
                    gpu_ms=get_elapsed_time(gpu[0], gpu[1])
                    if gpu is not None else None,
                    buffer_size=buffer_size,
                    buffer_size_is_tight=buffer_size_is_tight)


def average_performance(perfs: Sequence[Perf]) -> Perf:
    """Calculate the average of multiple Perfs."""
    total_ns = 0
    gpu_ms: Optional[float] = 0.
    buffer_size: Optional[int] = 0
    buffer_size_is_tight = True
    for perf in perfs:
        total_ns += perf.total_ns
        if gpu_ms is not None:
            if perf.gpu_ms is not None:
                gpu_ms += perf.gpu_ms
            else:
                gpu_ms = None
        if buffer_size is not None:
            if perf.buffer_size is not None:
                buffer_size += perf.buffer_size
            else:
                buffer_size = None
        if not perf.buffer_size_is_tight:
            buffer_size_is_tight = False
    if gpu_ms is not None:
        gpu_ms = gpu_ms / len(perfs)
    if buffer_size is not None:
        buffer_size = buffer_size // len(perfs)
    return Perf(total_ns=total_ns // len(perfs),
                gpu_ms=gpu_ms,
                buffer_size=buffer_size,
                buffer_size_is_tight=buffer_size_is_tight)
