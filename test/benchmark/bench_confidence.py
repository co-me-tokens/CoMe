"""Benchmark confidence distillation model inference time.

Given a checkpoint directory (containing config.json and checkpoint.pth),
automatically instantiates the confidence network, loads weights, and
benchmarks forward-pass runtime across various (B, P, H, W) token shapes.

Usage:
    python -m test.benchmark.bench_confidence /path/to/step_00100
    python -m test.benchmark.bench_confidence /path/to/step_00100 --dtype fp16
    python -m test.benchmark.bench_confidence /path/to/step_00100 --min-run-time 5.0
"""

import json
import argparse
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark

from src.accelerate.common.confidence_predictor import GlobalAttentionConfidence, Conv2DConfidence


DEVICE = "cuda"

# (label, B, P, H, W)
#   B = batch size, P = number of views, H/W = patch-grid spatial dims
#   total sequence length per sample = P * H * W
SHAPE_CONFIGS = [
    # --- Vary batch size (P=2, 16x16 spatial = 512 tokens) ---
    ("B1_P2_16x16",     1,   2,  16,  16),
    ("B2_P2_16x16",     2,   2,  16,  16),
    ("B4_P2_16x16",     4,   2,  16,  16),
    ("B8_P2_16x16",     8,   2,  16,  16),
    # --- Vary views / sequence (B=1, 16x16 spatial) ---
    ("B1_P4_16x16",     1,   4,  16,  16),
    ("B1_P8_16x16",     1,   8,  16,  16),
    ("B1_P16_16x16",    1,  16,  16,  16),
    # --- Vary spatial resolution (B=1, P=2) ---
    ("B1_P2_24x24",     1,   2,  24,  24),
    ("B1_P2_32x32",     1,   2,  32,  32),
    # --- Realistic VGGT shapes (patch_size=14) ---
    ("B1_P2_37x37",     1,   2,  37,  37),
    ("B1_P4_37x37",     1,   4,  37,  37),
    ("B1_P2_23x37",     1,   2,  23,  37),
]

_STATE_DICT_PREFIX_TO_CLASS: dict[str, type[torch.nn.Module]] = {
    "proj1":  GlobalAttentionConfidence,
    "module": Conv2DConfidence,
}


def _infer_model_class(state_dict: dict[str, torch.Tensor]) -> type[torch.nn.Module]:
    for prefix, cls in _STATE_DICT_PREFIX_TO_CLASS.items():
        if any(k.startswith(f"{prefix}.") for k in state_dict):
            return cls
    raise ValueError(
        f"Cannot infer model class from state dict keys: {sorted(state_dict.keys())[:10]}"
    )


def load_from_checkpoint(
    ckpt_dir: Path,
    device: str = DEVICE,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, dict]:
    """Load confidence predictor from a training checkpoint directory.

    Args:
        ckpt_dir: Directory containing config.json and checkpoint.pth.
        device:   Target device.
        dtype:    Target dtype.

    Returns:
        (model, config_dict) — model in eval mode on the target device.
    """
    config_path = ckpt_dir / "config.json"
    weight_path = ckpt_dir / "checkpoint.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {ckpt_dir}")
    if not weight_path.exists():
        raise FileNotFoundError(f"checkpoint.pth not found in {ckpt_dir}")

    with open(config_path) as f:
        config = json.load(f)

    dims       = config["predictor_dims"]
    activation = config["predictor_active"]

    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model_cls  = _infer_model_class(state_dict)

    model = model_cls(dims, activation)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device=device, dtype=dtype).eval(), config


@torch.inference_mode()
def run_benchmark(
    model: torch.nn.Module,
    C: int,
    dtype: torch.dtype,
    min_run_time: float,
) -> list[benchmark.Measurement]:
    model_name = type(model).__name__
    results: list[benchmark.Measurement] = []

    for label, B, P, H, W in SHAPE_CONFIGS:
        seq_len = P * H * W
        tokens  = torch.randn(B * P, H * W, C, device=DEVICE, dtype=dtype)
        BPHWC   = (B, P, H, W, C)

        t = benchmark.Timer(
            stmt="model(tokens, BPHWC)",
            globals={"model": model, "tokens": tokens, "BPHWC": BPHWC},
            label="confidence_forward",
            sub_label=f"{label}  B={B} seq={seq_len}",
            description=model_name,
        )
        r = t.blocked_autorange(min_run_time=min_run_time)
        results.append(r)

        print(f"  {label:20s}  B={B:<3d}  P={P:<3d}  HxW={H:>2d}x{W:<2d}  "
              f"seq={seq_len:<6d}  median={r.median * 1e3:.3f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark confidence distillation model inference time."
    )
    parser.add_argument(
        "checkpoint", type=Path,
        help="Checkpoint directory (containing config.json and checkpoint.pth)",
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
        help="Inference dtype (default: bf16)",
    )
    parser.add_argument(
        "--min-run-time", type=float, default=2.0,
        help="Minimum wall-clock seconds per config (default: 2.0)",
    )
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype     = dtype_map[args.dtype]

    model, config = load_from_checkpoint(args.checkpoint, dtype=dtype)
    C = config["predictor_dims"][0]

    print(f"Model     : {type(model).__name__}")
    print(f"Dims      : {config['predictor_dims']}")
    print(f"Activation: {config['predictor_active']}")
    print(f"Dtype     : {args.dtype}")
    print(f"Device    : {DEVICE}")

    print("\n" + "=" * 90)
    print("CONFIDENCE MODEL INFERENCE BENCHMARK")
    print("=" * 90)

    results = run_benchmark(model, C, dtype, args.min_run_time)

    print("\n" + "=" * 90)
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()
