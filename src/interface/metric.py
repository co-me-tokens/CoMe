import typing as tp
from dataclasses import dataclass


T_Co = tp.TypeVar("T_Co", covariant=True)

@dataclass
class Metric(tp.Generic[T_Co]):
    good: T_Co
    bad : T_Co
    all : T_Co

    def to_dict(self) -> dict[str, T_Co]:
        return {"good": self.good, "bad": self.bad, "all": self.all}


@dataclass
class DepthEvaluation:
    metric_scale: bool
    mask_ratio  : float
    
    # Metrics
    avg_l1        : Metric[float]
    avg_delta_1_25: Metric[float]

    def to_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            "mask_ratio"    : self.mask_ratio,
            "avg_l1"        : self.avg_l1.to_dict(),
            "avg_delta_1_25": self.avg_delta_1_25.to_dict(),
        }


@dataclass
class IntrinsicEvaluation:
    metric_scale: bool
    mask_ratio  : float
