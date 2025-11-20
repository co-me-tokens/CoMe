from math import isfinite
from typing import TypeVar, Generic
from dataclasses import dataclass
from statistics import mean, median, stdev

T = TypeVar("T", int, float)

@dataclass
class Statistic(Generic[T]):
    avg: T
    std: T
    med: T
    
    @classmethod
    def from_data(cls, data: list[T]) -> "Statistic[T]":
        result = cls(
            avg=mean(filter(isfinite, data)), std=stdev(filter(isfinite, data)), med=median(filter(isfinite, data))
        )
        return result
    
    def __format__(self, format_spec: str) -> str:
        return f"avg: {self.avg:{format_spec}}, std: {self.std:{format_spec}}, med: {self.med:{format_spec}}"
