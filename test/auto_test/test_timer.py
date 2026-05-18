"""Regression tests for timer lifecycle behavior."""

from __future__ import annotations

import typing as T

from torch.cuda import Event

from src.utility.timer import Timer


class _DummyEvent:
    def __init__(self, elapsed_ms: float) -> None:
        self._elapsed_ms = elapsed_ms

    def elapsed_time(self, _: object) -> float:
        return self._elapsed_ms


def test_timer_reset_clears_accumulated_entries() -> None:
    Timer.reset()
    Timer._timer_queue["inference"] = T.cast(
        list[tuple[Event, Event]],
        [
            (_DummyEvent(10.0), _DummyEvent(0.0)),
            (_DummyEvent(20.0), _DummyEvent(0.0)),
        ],
    )

    entries_before_reset = Timer.results()

    assert len(entries_before_reset) == 1
    assert entries_before_reset[0].name == "inference"
    assert entries_before_reset[0].count == 2

    Timer.reset()

    assert Timer.results() == []
