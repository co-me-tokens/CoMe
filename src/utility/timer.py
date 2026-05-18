from __future__ import annotations

import numpy as np
import typing as tp
from dataclasses import dataclass
from contextlib import contextmanager

from rich import box
from rich.console import Console
from rich.table import Table

from torch.cuda import Event, synchronize, default_stream


class Timer:
    
    @dataclass(kw_only=True, frozen=True, eq=False, slots=True)
    class Entry:
        name : str
        count: int
        P50  : float
        P90  : float
        P99  : float
    
    _is_activate: tp.ClassVar[bool] = False
    _timer_queue: tp.ClassVar[dict[str, list[tuple[Event, Event]]]] = dict()
    
    @classmethod
    def is_active(cls) -> bool: return cls._is_activate
    
    @classmethod
    def activate(cls, mode: bool = True): cls._is_activate = mode

    @classmethod
    def reset(cls) -> None:
        cls._timer_queue.clear()
    
    @classmethod
    @contextmanager
    def range(cls, name: str) -> tp.Iterator[None]:
        if cls._is_activate:
            if name not in cls._timer_queue:
                cls._timer_queue[name] = []
            
            start_event = Event(enable_timing=True)
            end_event   = Event(enable_timing=True)
            start_event.record(stream=default_stream())
            try:
                yield
            finally:
                end_event.record(stream=default_stream())
                # In torch <= 2.7.0 there are some typing issue w/ _CudaEventBase and Event class.
                t_start_event = tp.cast(Event, start_event)
                t_end_event   = tp.cast(Event, end_event)
                cls._timer_queue[name].append((t_start_event, t_end_event))
            
        else:
            try: yield
            finally: pass

    @classmethod
    def results(cls) -> list[Timer.Entry]:
        results: list[Timer.Entry] = []
        for name, events in cls._timer_queue.items():
            elapsed = np.array([s.elapsed_time(e) for s, e in events])
            results.append(Timer.Entry(
                name=name, count=len(events), 
                P50=np.median(elapsed).item(),
                P90=np.percentile(elapsed, 90).item(),
                P99=np.percentile(elapsed, 99).item(),
            ))
        return results

    @classmethod
    def report(cls) -> str:
        synchronize()

        table = Table(box=box.SIMPLE_HEAD, header_style="bold", show_edge=False)
        table.add_column("Name")
        table.add_column("Count", justify="right")
        table.add_column("Median (ms)", justify="right")
        table.add_column("P90 (ms)", justify="right")
        table.add_column("P99 (ms)", justify="right")

        for entry in cls.results():
            table.add_row(
                entry.name, f"{entry.count}", f"{entry.P50:.2f}", f"{entry.P90:.2f}", f"{entry.P99:.2f}",
            )

        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get()
