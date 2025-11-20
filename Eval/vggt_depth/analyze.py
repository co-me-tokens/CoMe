import json
from pathlib import Path
from ..common.statistic import Statistic

def analyze(file: Path):
    with open(file, "r") as f: data = json.load(f)
    
    return {
        "Runtime_ms": Statistic.from_data([d['runtime_ms'] for d in data]),
        "Memory_gb ": Statistic.from_data([d['memory_gb']  for d in data]),
        "L1_meter  ": Statistic.from_data([d['avg_l1'] for d in data]),
        "RMSE_meter": Statistic.from_data([d['avg_rmse'] for d in data]),
        "AbsRel    ": Statistic.from_data([d['avg_absrel'] for d in data]),
        "δ_1.25    ": Statistic.from_data([d['avg_δ1_25'] for d in data]),
    }
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    args   = parser.parse_args()
    
    print(args.file)
    print("\n".join(map(lambda pair: f"\t{pair[0]}\t| {pair[1]:.6f}", analyze(args.file).items())))
